#!/usr/bin/env node
/**
 * dsha-zhipu-mcp — 智谱视觉模型 MCP server for DSH
 *
 * 零依赖实现：仅用 Node 内置模块 + 全局 fetch。
 * 传输层：stdio，消息帧为换行分隔的 JSON-RPC 2.0（NDJSON）。
 *
 * 设计要点：
 *  - stdout 只输出协议消息；所有日志走 stderr（否则会污染协议流）。
 *  - 密钥从环境变量 ZHIPU_API_KEY 读取。DSH 的 stdio 子进程会清洗掉名字含
 *    KEY/TOKEN/SECRET/PASSWORD 的变量，所以必须在 cordis.patch.yml 的 env: 里显式传。
 *  - 核心特性：多模型轮换容错。免费模型频繁限流（错误码 1305），
 *    遇限流自动切换下一个模型，走完一轮再从头发起，直到成功或超出预算。
 *  - 缺密钥时不崩溃：initialize / tools/list 正常返回，工具调用返回可读错误。
 */

import process from 'node:process';
import { readFile } from 'node:fs/promises';
import { basename, extname } from 'node:path';

// ---------------------------------------------------------------- 配置

const API_KEY = (process.env.ZHIPU_API_KEY ?? '').trim();
const API_BASE = (process.env.ZHIPU_API_BASE ?? 'https://open.bigmodel.cn/api/paas/v4').replace(/\/+$/, '');
const REQUEST_TIMEOUT_MS = clampInt(process.env.ZHIPU_TIMEOUT_MS, 180_000, 5_000, 600_000);
const MAX_IMAGE_BYTES = clampInt(process.env.ZHIPU_MAX_IMAGE_BYTES, 20 * 1024 * 1024, 1024, 100 * 1024 * 1024);
const SERVER_NAME = 'zhipu';
const SERVER_VERSION = '1.0.0';
const USER_AGENT = `dsha-zhipu-mcp/${SERVER_VERSION}`;

/**
 * 模型池（按尝试优先级排列）。
 *
 * 顺序依据 2026-10-14 本机实测（同一张模糊 MiFlash 照片，要求逐字读出路径）：
 *   glm-4.6v-flash            唯一全对（最强，但最容易限流 —— 实测第二张图就报了 1305）
 *   glm-4v-flash              漏字符但稳定，额度宽松
 *   glm-4.1v-thinking-flash   会输出思维链，字符级易错
 * 注意：顺序不代表"越好越先试"，而是"越值得优先尝试的越靠前"；
 *       限流时自动降级到下一个，故最差情况也能出结果。
 */
const MODELS = [
  { id: 'glm-4.6v-flash', label: 'GLM-4.6V-Flash', thinking: false },
  { id: 'glm-4v-flash', label: 'GLM-4V-Flash', thinking: false },
  { id: 'glm-4.1v-thinking-flash', label: 'GLM-4.1V-Thinking-Flash', thinking: true },
];

/** 可在参数中显式指定的模型白名单（防止模型自己编造模型名） */
const MODEL_IDS = MODELS.map((m) => m.id);

/** 默认最大轮数：3 个模型 × 2 轮 = 每张图最多 6 次尝试 */
const DEFAULT_MAX_ROUNDS = 2;

/** 轮与轮之间的等待（毫秒），给限流一点恢复时间 */
const ROUND_DELAY_MS = 1_500;

function clampInt(raw, fallback, min, max) {
  const n = Number.parseInt(String(raw ?? ''), 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(Math.max(n, min), max);
}

function log(...args) {
  process.stderr.write(`[zhipu-mcp] ${args.join(' ')}\n`);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// ---------------------------------------------------------------- 工具定义

const IMAGE_SOURCE_PROPS = {
  path: {
    type: 'string',
    description:
      '本地图片文件的绝对路径，例如 /sdcard/DCIM/Camera/a.jpg。与 url、base64 三选一。适合读取手机上的截图或照片。',
  },
  url: {
    type: 'string',
    description: '远程图片的 http(s) 地址。与 path、base64 三选一。适合网络图片。',
  },
  base64: {
    type: 'string',
    description:
      '图片的 base64 编码（不含 data: 前缀）。与 path、url 三选一。一般不需要手填，优先用 path。',
  },
  mime: {
    type: 'string',
    description: "配合 base64 使用，指定图片类型，如 image/jpeg、image/png。默认 image/jpeg。",
  },
};

const TOOLS = [
  {
    name: 'zhipu_vision',
    description:
      '用智谱免费视觉模型识别图片内容（看图、读图上的字、描述画面、解析截图/照片）。' +
      '内置多模型轮换：遇到限流（错误码 1305）会自动切换到下一个模型并循环重试，直到成功。' +
      '适合读取手机截图、照片中的文字与界面元素。',
    inputSchema: {
      type: 'object',
      properties: {
        prompt: {
          type: 'string',
          description:
            '给视觉模型的指令，用中文描述你要它做什么，例如"读出图中输入框里的完整路径，逐字符准确输出"。',
        },
        ...IMAGE_SOURCE_PROPS,
        model: {
          type: 'string',
          description:
            `指定使用哪个模型（可选）。可选值：${MODEL_IDS.join(' / ')}。` +
            '省略则按内置优先级自动选择并在限流时轮换。',
        },
        maxRounds: {
          type: 'integer',
          minimum: 1,
          maximum: 10,
          description: '最多轮换几轮（每轮尝试所有模型一次）。默认 2。遇到持续限流可调大。',
        },
        maxTokens: {
          type: 'integer',
          minimum: 16,
          maximum: 8192,
          description: '回复最大 token 数。默认 1024；要读长文本可调到 2048~4096。',
        },
      },
      required: ['prompt'],
      additionalProperties: false,
    },
  },
  {
    name: 'zhipu_models',
    description:
      '列出智谱视觉模型池的当前状态：模型 id、优先级顺序、以及一次轻量探测的可用性（是否限流）。' +
      '用于判断此刻该用哪个模型，或排查配额问题。',
    inputSchema: {
      type: 'object',
      properties: {
        probe: {
          type: 'boolean',
          description: '是否实际发起一次轻量探测请求验证可用性。默认 true（true 会消耗极少额度）。',
          default: true,
        },
      },
      required: [],
      additionalProperties: false,
    },
  },
];

// ---------------------------------------------------------------- 错误分类

class ZhipuError extends Error {
  constructor(message, { status, code, retryable = false, detail } = {}) {
    super(message);
    this.name = 'ZhipuError';
    this.status = status;
    this.code = code;
    this.retryable = retryable;
    this.detail = detail;
  }
}

/**
 * 判断错误是否值得换模型重试。
 *
 * 实测智谱免费模型的限流返回：HTTP 429 或 HTTP 200/400 带 body.code === '1305'
 * （message: "该模型当前访问量过大，请您稍后再试"）。
 * 这类错误是"这一时刻这个模型忙"，换一个模型/过一会儿就好 → 可重试。
 *
 * 相反，401/403（密钥错）、1301（参数错）重试多少次都没用 → 不重试，直接抛。
 */
function isRetryable(status, code, message) {
  if (status === 429) return true;
  if (status === 408 || status === 500 || status === 502 || status === 503 || status === 504) return true;
  const c = String(code ?? '');
  if (c === '1305') return true; // 访问量过大
  if (/访问量过大|稍后再试|rate\s*limit|too\s*many\s*requests|busy/i.test(String(message ?? ''))) return true;
  return false;
}

// ---------------------------------------------------------------- 图片处理

const MIME_BY_EXT = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.bmp': 'image/bmp',
};

async function resolveImage(args) {
  const path = nonEmptyString(args?.path);
  const url = nonEmptyString(args?.url);
  const b64 = nonEmptyString(args?.base64);

  const given = [path, url, b64].filter(Boolean);
  if (given.length === 0) {
    throw new ZhipuError('缺少图片来源：请提供 path（本地路径）、url（网络地址）或 base64 三者之一。');
  }
  if (given.length > 1) {
    throw new ZhipuError('图片来源只能指定一个：path / url / base64 请只填其中一项。');
  }

  if (url) {
    if (!/^https?:\/\//i.test(url)) throw new ZhipuError('参数 url 必须以 http:// 或 https:// 开头。');
    return { kind: 'url', value: url, note: `网络图片 ${url.slice(0, 80)}` };
  }

  if (b64) {
    const mime = nonEmptyString(args?.mime) ?? 'image/jpeg';
    const cleaned = b64.replace(/^data:[^;]+;base64,/, '');
    const bytes = Math.floor((cleaned.length * 3) / 4);
    if (bytes > MAX_IMAGE_BYTES) {
      throw new ZhipuError(
        `图片过大：约 ${(bytes / 1048576).toFixed(1)} MB，上限 ${(MAX_IMAGE_BYTES / 1048576).toFixed(0)} MB。请压缩后再试。`,
      );
    }
    return { kind: 'base64', value: cleaned, mime, note: `内联 base64（约 ${(bytes / 1024).toFixed(0)} KB）` };
  }

  // 本地文件
  let buf;
  try {
    buf = await readFile(path);
  } catch (err) {
    const reason = err?.code === 'ENOENT' ? '文件不存在' : err?.code === 'EACCES' ? '无读取权限' : String(err?.message ?? err);
    throw new ZhipuError(`读取本地图片失败（${path}）：${reason}。请确认路径正确且可读。`);
  }

  if (buf.length > MAX_IMAGE_BYTES) {
    throw new ZhipuError(
      `图片过大：${(buf.length / 1048576).toFixed(1)} MB，上限 ${(MAX_IMAGE_BYTES / 1048576).toFixed(0)} MB。请压缩后再试。`,
    );
  }

  const ext = extname(path).toLowerCase();
  const mime = nonEmptyString(args?.mime) ?? MIME_BY_EXT[ext] ?? 'image/jpeg';
  return {
    kind: 'base64',
    value: buf.toString('base64'),
    mime,
    note: `本地文件 ${basename(path)}（${(buf.length / 1024).toFixed(0)} KB, ${mime}）`,
  };
}

function imagePart(img) {
  const url = img.kind === 'url' ? img.value : `data:${img.mime};base64,${img.value}`;
  return { type: 'image_url', image_url: { url } };
}

// ---------------------------------------------------------------- HTTP 调用

async function callModel(modelId, image, prompt, maxTokens) {
  if (!API_KEY) {
    throw new ZhipuError(
      '缺少智谱密钥：环境变量 ZHIPU_API_KEY 为空。请在 $DSH_HOME/.env 中设置 ZHIPU_API_KEY，' +
        '并确认 cordis.patch.yml 的 mcp-zhipu 条目在 env: 里显式传递了该变量。',
    );
  }

  const payload = {
    model: modelId,
    // image 为 null 时走纯文本模式（用于 zhipu_models 的轻量探测），
    // 这样探测结果只反映"模型是否可用/限流"，不受图片格式问题干扰。
    messages: [
      {
        role: 'user',
        content: image ? [imagePart(image), { type: 'text', text: prompt }] : [{ type: 'text', text: prompt }],
      },
    ],
    max_tokens: maxTokens,
  };

  let res;
  try {
    res = await fetch(`${API_BASE}/chat/completions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${API_KEY}`,
        'Content-Type': 'application/json',
        Accept: 'application/json',
        'User-Agent': USER_AGENT,
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (err) {
    const reason = err?.name === 'TimeoutError' ? `请求超时（${REQUEST_TIMEOUT_MS}ms）` : String(err?.message ?? err);
    throw new ZhipuError(`调用智谱失败：${reason}`, { retryable: true });
  }

  const raw = await res.text();

  let parsed = null;
  try {
    parsed = JSON.parse(raw);
  } catch {
    /* 非 JSON 响应 */
  }

  if (!res.ok || parsed?.error) {
    const errObj = parsed?.error ?? {};
    const code = errObj.code;
    const message = errObj.message ?? raw.slice(0, 300);
    const retryable = isRetryable(res.status, code, message);
    const hint =
      res.status === 401 || res.status === 403 || code === '1000' || code === '1001'
        ? '（密钥无效或权限不足，重试无用，请检查 ZHIPU_API_KEY）'
        : retryable
          ? '（限流/服务繁忙，可切换模型重试）'
          : '';
    throw new ZhipuError(`HTTP ${res.status}${code ? ` code=${code}` : ''}${hint}：${message}`, {
      status: res.status,
      code,
      retryable,
      detail: message,
    });
  }

  const choice = parsed?.choices?.[0];
  const content = choice?.message?.content;
  let text = '';
  if (typeof content === 'string') {
    text = content;
  } else if (Array.isArray(content)) {
    text = content.map((c) => (typeof c === 'string' ? c : (c?.text ?? ''))).join('');
  }

  if (!text.trim()) {
    throw new ZhipuError(`模型 ${modelId} 返回了空内容（finish_reason=${choice?.finish_reason ?? '未知'}）。`, {
      retryable: true,
    });
  }

  // 思考类模型会把思维链包在 <think>…</think> 里 —— 提取出来单独返回，
  // 避免正文被思维链淹没（实测 glm-4.1v-thinking-flash 就是这种格式）。
  let thinking = '';
  const m = text.match(/<think>([\s\S]*?)<\/think>/i);
  if (m) {
    thinking = m[1].trim();
    text = text.replace(m[0], '').trim();
  }
  // 清掉回答框标记
  text = text.replace(/<\|begin_of_box\|>/g, '').replace(/<\|end_of_box\|>/g, '').trim();

  return { text, thinking, usage: parsed?.usage ?? null, model: parsed?.model ?? modelId };
}

// ---------------------------------------------------------------- 轮换执行

/**
 * 多模型轮换执行：按模型池顺序逐个尝试，遇可重试错误就换下一个；
 * 走完一轮还没成功就等一会儿再来一轮，直到成功或超出 maxRounds。
 *
 * 返回 { result, attempts, winner }，attempts 记录每次尝试的模型与结果，
 * 便于模型/用户看清"到底切了几次、为什么切"。
 */
async function runWithRotation({ image, prompt, maxTokens, only, maxRounds }) {
  const pool = only ? MODELS.filter((m) => m.id === only) : MODELS;
  if (pool.length === 0) {
    throw new ZhipuError(`指定的模型不在支持列表中。可用：${MODEL_IDS.join(' / ')}`);
  }

  const attempts = [];
  let lastError = null;

  for (let round = 1; round <= maxRounds; round++) {
    for (const model of pool) {
      const started = Date.now();
      try {
        const r = await callModel(model.id, image, prompt, maxTokens);
        attempts.push({
          round,
          model: model.id,
          ok: true,
          ms: Date.now() - started,
          usage: r.usage,
        });
        return { result: r, attempts, winner: model, rounds: round };
      } catch (err) {
        const ms = Date.now() - started;
        const retryable = err instanceof ZhipuError && err.retryable;
        attempts.push({
          round,
          model: model.id,
          ok: false,
          ms,
          retryable,
          status: err?.status ?? null,
          code: err?.code ?? null,
          error: String(err?.message ?? err).slice(0, 200),
        });
        lastError = err;
        log(`模型 ${model.id} 第 ${round} 轮失败（${retryable ? '可重试' : '不可重试'}）：${err?.message ?? err}`);

        // 不可重试的错误（如密钥错、参数错）—— 继续换模型也没意义，直接放弃
        if (!retryable) {
          const e = new ZhipuError(
            `模型 ${model.id} 返回不可重试的错误，已中止（换模型也无法解决）：${err?.message ?? err}`,
            { status: err?.status, code: err?.code, detail: err?.detail },
          );
          e.attempts = attempts;
          throw e;
        }
      }
    }

    if (round < maxRounds) {
      log(`第 ${round} 轮全部模型均失败，等待 ${ROUND_DELAY_MS}ms 后重试…`);
      await sleep(ROUND_DELAY_MS);
    }
  }

  // 全部轮次都没成功
  const e = new ZhipuError(
    `已尝试 ${maxRounds} 轮 × ${pool.length} 个模型，全部失败。最后一次错误：${lastError?.message ?? '未知'}` +
      (lastError?.code === '1305' ? '（所有模型都在限流，请稍后再试或加大 maxRounds）' : ''),
    { status: lastError?.status, code: lastError?.code },
  );
  e.attempts = attempts;
  throw e;
}

// ---------------------------------------------------------------- 工具执行

function nonEmptyString(v) {
  return typeof v === 'string' && v.trim().length > 0 ? v.trim() : undefined;
}

function toInt(v) {
  const n = typeof v === 'number' ? v : Number.parseInt(String(v ?? ''), 10);
  return Number.isFinite(n) ? Math.trunc(n) : undefined;
}

function formatAttempts(attempts) {
  return attempts
    .map((a) => {
      const mark = a.ok ? '✅' : a.retryable ? '↻' : '✖';
      const why = a.ok ? `成功 ${a.ms}ms` : `${a.code ? `code=${a.code} ` : ''}${a.error ?? ''}`;
      return `  ${mark} 第${a.round}轮 ${a.model} — ${why}`;
    })
    .join('\n');
}

async function runTool(name, args) {
  if (name === 'zhipu_models') {
    const probe = args?.probe !== false;
    const lines = [];
    for (const m of MODELS) {
      let status = '未探测（probe=false）';
      if (probe) {
        const t0 = Date.now();
        try {
          // 纯文本探测：只关心"模型当前是否可用"，不涉及图片，
          // 避免把图片格式问题误报成限流。
          await callModel(m.id, null, '回复一个字：好', 8);
          status = '✅ 可用';
        } catch (err) {
          const code = err?.code;
          if (code === '1305' || err?.status === 429) status = '⛔ 限流中';
          else if (err?.status === 401 || err?.status === 403) status = '⛔ 密钥无效';
          else status = `⚠️ 异常（${code ? `code=${code}` : (err?.status ?? '未知')}）`;
        }
        status += `（${Date.now() - t0}ms）`;
      }
      lines.push(`  ${MODELS.indexOf(m) + 1}. ${m.id}  [${m.label}]  ${status}`);
    }
    return [
      '智谱视觉模型池（按尝试优先级）：',
      ...lines,
      '',
      '说明：遇到限流（code 1305）时，zhipu_vision 会自动切换到下一个模型并循环重试。',
      `模型名白名单：${MODEL_IDS.join(' / ')}`,
      `当前每轮尝试 ${MODELS.length} 个模型，默认 ${DEFAULT_MAX_ROUNDS} 轮。`,
    ].join('\n');
  }

  if (name !== 'zhipu_vision') throw new ZhipuError(`未知工具：${name}`);

  const prompt = nonEmptyString(args?.prompt);
  if (!prompt) throw new ZhipuError('参数 prompt 不能为空，请描述你要模型做什么。');

  const image = await resolveImage(args);

  const requestedModel = nonEmptyString(args?.model);
  if (requestedModel && !MODEL_IDS.includes(requestedModel)) {
    throw new ZhipuError(`不支持的模型 "${requestedModel}"。可用：${MODEL_IDS.join(' / ')}`);
  }

  const maxRounds = Math.min(Math.max(toInt(args?.maxRounds) ?? DEFAULT_MAX_ROUNDS, 1), 10);
  const maxTokens = Math.min(Math.max(toInt(args?.maxTokens) ?? 1024, 16), 8192);

  const started = Date.now();
  const { result, attempts, winner, rounds } = await runWithRotation({
    image,
    prompt,
    maxTokens,
    only: requestedModel,
    maxRounds,
  });
  const totalMs = Date.now() - started;

  const head = [
    `【模型】${winner.id}（${winner.label}）`,
    `【图片】${image.note}`,
    `【耗时】${totalMs}ms（第 ${rounds} 轮命中）`,
    attempts.length > 1 ? `【轮换记录】共尝试 ${attempts.length} 次：\n${formatAttempts(attempts)}` : '【轮换记录】一次成功，无需切换',
  ].join('\n');

  const body = result.thinking
    ? `${result.text}\n\n—— 附：模型思维链 ——\n${result.thinking}`
    : result.text;

  return `${head}\n\n—— 识别结果 ——\n${body}`;
}

// ---------------------------------------------------------------- JSON-RPC 协议层

const OUR_PROTOCOL_VERSIONS = ['2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05', '2024-10-07'];
const FALLBACK_PROTOCOL_VERSION = '2024-11-05';

function negotiateVersion(requested) {
  const want = typeof requested === 'string' ? requested : '';
  for (const v of OUR_PROTOCOL_VERSIONS) {
    if (v <= want) return v;
  }
  return FALLBACK_PROTOCOL_VERSION;
}

function send(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function sendResult(id, result) {
  send({ jsonrpc: '2.0', id, result });
}

function sendError(id, code, message, data) {
  const error = { code, message };
  if (data !== undefined) error.data = data;
  send({ jsonrpc: '2.0', id, error });
}

async function handleRequest(msg) {
  const { id, method, params } = msg;

  switch (method) {
    case 'initialize': {
      const version = negotiateVersion(params?.protocolVersion);
      log(`initialize: 客户端请求 ${params?.protocolVersion ?? '(未指定)'} → 协商为 ${version}`);
      sendResult(id, {
        protocolVersion: version,
        capabilities: { tools: { listChanged: false } },
        serverInfo: { name: 'Zhipu Vision MCP Server', version: SERVER_VERSION },
      });
      return;
    }

    case 'ping':
      sendResult(id, {});
      return;

    case 'tools/list':
      sendResult(id, {
        tools: TOOLS.map(({ name, description, inputSchema }) => ({ name, description, inputSchema })),
      });
      return;

    case 'tools/call': {
      const toolName = params?.name;
      const args = params?.arguments ?? {};
      try {
        const text = await runTool(toolName, args);
        sendResult(id, { content: [{ type: 'text', text }], isError: false });
      } catch (err) {
        let message = err instanceof ZhipuError ? err.message : `执行工具失败：${String(err?.message ?? err)}`;
        if (Array.isArray(err?.attempts) && err.attempts.length > 0) {
          message += `\n\n【轮换记录】\n${formatAttempts(err.attempts)}`;
        }
        log(`tools/call ${toolName} 失败: ${message.slice(0, 300)}`);
        sendResult(id, { content: [{ type: 'text', text: message }], isError: true });
      }
      return;
    }

    case 'resources/list':
      sendResult(id, { resources: [] });
      return;
    case 'resources/templates/list':
      sendResult(id, { resourceTemplates: [] });
      return;
    case 'prompts/list':
      sendResult(id, { prompts: [] });
      return;

    default:
      sendError(id, -32601, `Method not found: ${method}`);
  }
}

const inFlight = new Set();

function handleMessage(msg) {
  if (!msg || typeof msg !== 'object') return;

  if (msg.id === undefined || msg.id === null) {
    if (msg.method === 'notifications/initialized') log('客户端已初始化');
    return;
  }

  const task = handleRequest(msg).catch((err) => {
    log(`处理 ${msg.method} 时发生未捕获错误: ${String(err?.stack ?? err)}`);
    sendError(msg.id, -32603, `Internal error: ${String(err?.message ?? err)}`);
  });
  inFlight.add(task);
  task.finally(() => inFlight.delete(task));
}

// ---------------------------------------------------------------- stdio 主循环

let buffer = '';
process.stdin.setEncoding('utf8');

process.stdin.on('data', (chunk) => {
  buffer += chunk;
  let idx;
  while ((idx = buffer.indexOf('\n')) !== -1) {
    const line = buffer.slice(0, idx).trim();
    buffer = buffer.slice(idx + 1);
    if (!line) continue;
    try {
      handleMessage(JSON.parse(line));
    } catch (err) {
      log(`无法解析的消息（已忽略）: ${line.slice(0, 200)} — ${String(err?.message ?? err)}`);
    }
  }
});

process.stdin.on('end', () => {
  void Promise.allSettled([...inFlight]).then(() => process.exit(0));
});
process.stdin.on('error', (err) => {
  log(`stdin 错误: ${String(err?.message ?? err)}`);
  process.exit(1);
});

if (!API_KEY) {
  log('警告：ZHIPU_API_KEY 未设置，工具将返回错误信息。');
}
log(`已启动 v${SERVER_VERSION}，模型池 ${MODELS.length} 个，提供 ${TOOLS.length} 个工具。`);
