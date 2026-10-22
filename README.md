# dsha-zhipu-mcp

A zero-dependency MCP server exposing **Zhipu AI (智谱 GLM)** vision models to AI agents.

Built for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) but works with any MCP client that supports stdio.

## Tools

| Tool | Purpose |
| --- | --- |
| `zhipu_vision` | Analyze an image: read text off a screenshot, describe a photo, parse UI elements. Accepts a local path, a remote URL, or raw base64. |
| `zhipu_models` | List the vision model pool with priority order, plus a lightweight availability probe to detect rate limiting. |

## Why this exists

Coding agents constantly need to look at screenshots. Most vision APIs either
require an SDK, choke on rate limits, or return the wrong shape. This server:

- Uses **only Node.js built-in modules** — no `npm install`, no lockfile drift.
- **Rotates across a model pool** when a model reports rate limiting (error `1305`),
  instead of failing the whole request.
- Reads its key from `ZHIPU_API_KEY` — nothing is hardcoded.

## Configuration

```yaml
- insert:
    - id: mcp-zhipu
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: zhipu
        transport: stdio
        command: /usr/local/bin/node
        args:
          - /path/to/server.mjs
        env:
          ZHIPU_API_KEY: !!js 'process.env.ZHIPU_API_KEY ?? ""'
```

> **Note on env cleaning.** DSH strips variables whose names contain
> `KEY` / `TOKEN` / `SECRET` / `PASSWORD` from stdio subprocesses.
> `ZHIPU_API_KEY` matches, so it must be passed explicitly via `env:`
> as shown above — inheriting it from the parent process will not work.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `ZHIPU_API_KEY` | — | **Required.** API key. |
| `ZHIPU_API_BASE` | `https://open.bigmodel.cn/api/paas/v4` | Endpoint override. |
| `ZHIPU_TIMEOUT_MS` | `180000` | Per-request timeout. Vision models are slow; don't set it low. |
| `ZHIPU_MAX_IMAGE_BYTES` | `20971520` (20 MB) | Reject larger images before upload. |

## Behavior notes

- Passing an explicit `model` argument pins that model: no rotation. Rotation only
  applies when the caller lets the pool decide.
- Non-retryable errors (invalid key → 401/403) abort immediately rather than
  burning the whole pool.

## License

MIT — see [LICENSE](./LICENSE).
