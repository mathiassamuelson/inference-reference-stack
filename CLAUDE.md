# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reference deployment of a production-pattern LLM inference stack sized for modest hardware (4× consumer RTX 3090, single workstation, team-of-one operator). It is **configuration only** — there is no application source code, no build step, and no test suite. Every service runs from an upstream image; there are deliberately no custom Dockerfiles. "Working on this repo" means editing Compose, nginx, Prometheus, and Grafana config files and reasoning about how the containers compose.

The stack accompanies a separate inference-optimization training program. Commit messages and config comments reference "Week N" recipes and specific upstream bug numbers (e.g. `vllm-project/vllm#39133`, a KV-sizing bug). That provenance is why versions are pinned hard — see below.

## Common commands

```bash
cp .env.example .env          # set HF_CACHE_DIR (abs path) and GRAFANA_ADMIN_PASSWORD
docker compose up -d          # bring up the whole stack
docker compose ps             # service health
docker compose logs -f vllm   # follow one service (vllm | nginx | prometheus | grafana | dcgm-exporter | node-exporter)
docker compose up -d vllm     # recreate a single service after editing its config
docker compose down           # stop; add -v to also wipe the named data volumes

docker compose exec nginx nginx -t   # validate nginx.conf before reload
docker compose exec nginx nginx -s reload

curl http://localhost:8000/v1/models       # vLLM directly (localhost-only bind)
curl localhost:9090/-/reload               # hot-reload Prometheus (web.enable-lifecycle is on)
docker logs irs-nginx                      # access log shows $upstream_addr per request
```

There is no lint/test. The "verification instrument" for routing is the nginx access log: it records `upstream=$upstream_addr` per request to stdout, which is how pool distribution and named-route pinning are confirmed.

## Port and bind conventions

Most service ports are bound to `127.0.0.1` on purpose — nginx is meant to be the only public ingress. When adding or moving a service, preserve this: only the front door listens on `0.0.0.0`. Current binds: vLLM `127.0.0.1:8000`, Prometheus `127.0.0.1:9090`, dcgm-exporter `127.0.0.1:9400`, node-exporter `127.0.0.1:9100`. Grafana (`3000`) and nginx (`80`) are exposed on all interfaces.

## Architecture notes that span files

- **Service discovery is split-brain by design.** Prometheus scrapes services by their Compose **service name** on the `irs` bridge network (`vllm:8000`, `dcgm-exporter:9400`, `node-exporter:9100`) — container-to-container DNS. Operators and `curl` reach the same services via the host-published `127.0.0.1:PORT`. Use service names inside config that runs in a container, host ports from the shell.

- **nginx.conf is ahead of docker-compose.yml — a real, intentional drift.** `docker-compose.yml` currently runs a *single* vLLM service (the 26B MoE model on `:8000`). `nginx/nginx.conf` already encodes a *multi-tier* routing contract that assumes backends that Compose does not yet define: a `least_conn` pool over two 12B workers at `:8001`/`:8003` plus a 31B "orchestrator" at `:8000`. nginx also listens on `:8080` there, not the `:80` Compose publishes. Treat nginx.conf as the target topology; don't "fix" it to match Compose without confirming intent.

- **The routing contract (nginx).** The path number is an **instance index within a tier**, never a GPU id — hardware is abstracted out, and worker→port is an explicit map, not derived. `/v1/` is an API-version namespace preserved on every proxied route (so a future breaking change can add `/v2/` alongside). Routes: `/v1/...` → worker pool (no pinning); `/v1/worker/N/...` and `/v1/orchestrator/N/...` → a specific instance (the location strips `/v1/worker/N/` and the `proxy_pass` URI re-adds `/v1/` for the backend); `/healthz` → `ok`. Known gap, flagged in-file: an unknown instance like `/v1/worker/9/` falls through to the pool (backend 404) rather than the catch-all 404.

- **LLM-serving proxy tuning is load-bearing.** `proxy_buffering off` is required for token-streaming SSE (`stream=true`) to flow immediately; read/send timeouts are 3600s because generations run for minutes; keepalive to upstreams needs `proxy_http_version 1.1` + cleared `Connection` header. Keep these whenever touching proxy locations.

- **Grafana is fully provisioned**, no UI clicks: the Prometheus datasource and a dashboard provider are wired via `observability/grafana/provisioning/`. Drop dashboard JSON into `observability/grafana/dashboards/` and it loads automatically.

- **node-exporter runs in the host PID namespace with host mounts** (`/proc`, `/sys`, `/`) so it reports true host CPU/RAM/disk, not the container's — needed for the boot-choreography host-contention measurements. dcgm-exporter needs `SYS_ADMIN` and `count: all` GPUs.

## Version-pinning discipline

Image versions are pinned hard and the vLLM image is pinned **by digest**, not tag. The stated rationale: when an upstream fix lands, swap *only* that one service, re-run the benchmark, and attribute the change unambiguously to the fix rather than version drift. Do not loosen a pin or bump an image opportunistically — a bump is its own deliberate change with a benchmark behind it.

## GPU topology assumption

The vLLM service reserves `device_ids: ["0", "2"]` — the NVLink-connected pair on the target host — with `--tensor-parallel-size 2`. NVLink-paired GPUs are preferred over wider tensor parallelism. Adjust `device_ids` in `docker-compose.yml` for different hardware; the model is swappable by editing the positional model arg in the vLLM `command`, no other changes required.

## Known stale references

`.env.example` still says the HF cache is "Mounted into the Triton container" — Triton was dropped early (NGC's Triton vLLM image bundled a vLLM too old for Gemma 4's AWQ-INT4). The empty `triton/model_repository/` tree is a leftover of that abandoned plan. The active engine is vLLM's built-in OpenAI-compatible server.
