# inference-reference-stack

A reference deployment of a production-pattern LLM inference service on modest hardware.

Most published inference-serving guides assume datacenter GPUs (H100s, A100s) and hyperscaler-grade orchestration. This repository aims for the opposite end of the spectrum: what does a credible production inference stack look like when the hardware is four consumer RTX 3090s on a single workstation, and the operator is a team of one? The answer, it turns out, uses most of the same components — vLLM, nginx, Prometheus, Grafana — just sized and configured differently.

## Status

Early construction. This is a living reference deployment that will grow over time. See [Roadmap](#roadmap) for what's here now and what's coming.

## Architecture

```
┌──────────────┐      ┌───────────┐      ┌─────────────────────┐
│  Client app  │─────▶│   nginx   │─────▶│        vLLM         │
│ (statmon-ai) │ HTTPS│  (proxy,  │ HTTP │  (OpenAI-compatible │
└──────────────┘      │   TLS)    │      │        server)      │
                      └─────┬─────┘      └──────────┬──────────┘
                            │                       │
                            │ access logs           │ /metrics
                            ▼                       ▼
                      ┌─────────────────────────────────┐
                      │  Prometheus  ──▶  Grafana       │
                      └─────────────────────────────────┘
```

All services run as containers, orchestrated via Docker Compose. No custom Dockerfiles — every image is pulled from an upstream registry (Docker Hub for nginx/Prometheus/Grafana, vLLM's official image for the engine).

### Components

- **vLLM** — the inference engine, running as a standalone OpenAI-compatible HTTP server. Exposes Prometheus metrics on the same port as the API.
- **nginx** — reverse proxy and TLS terminator. Streaming-aware configuration for server-sent events.
- **Prometheus** — metrics collection from vLLM's metrics endpoint and (later) nginx access logs.
- **Grafana** — dashboards for request rate, latency, GPU utilization, and KV cache pressure.

### Model

Currently serving **Gemma 4 26B A4B MoE** (`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`) — AWQ-INT4 quantized, running on two NVLink-connected RTX 3090s via tensor parallelism. Other models can be swapped in by editing the vLLM service command in `docker-compose.yml`; no other changes required.

### Why not Triton?

The original plan was to front vLLM with NVIDIA Triton Inference Server. That was abandoned at Day 1 after discovering that the current NGC Triton vLLM image bundles a vLLM version too old to support Gemma 4's AWQ-INT4 quantization via the compressed-tensors path. Rather than maintain a custom Dockerfile to layer a newer vLLM on top of Triton, the stack uses vLLM's built-in OpenAI-compatible server directly. The nginx / observability / gateway components above this layer are architecture-level concerns that would be the same regardless of engine choice. Triton can be reintroduced later if and when multi-model serving or ensemble workflows become a requirement.

## Hardware target

- 4× NVIDIA RTX 3090 (24 GB each)
- GPUs 0 and 2 connected via NVLink (~100 GB/s bidirectional)
- Ubuntu 24.04, CUDA 12.6
- Docker with NVIDIA Container Toolkit

The Compose setup assumes this topology (GPUs 0 and 2 for the serving model). Adjust `device_ids` in `docker-compose.yml` for different hardware.

## Quick start

_Not yet available — coming as the stack lands._

The intended flow will be:

```bash
git clone https://github.com/<owner>/inference-reference-stack.git
cd inference-reference-stack
cp .env.example .env     # edit HF_CACHE_DIR and admin password
docker compose up -d
```

Point your OpenAI-compatible client at `https://<host>/v1/` once TLS is configured, or `http://<host>/v1/` for local development.

## Roadmap

Tracked by deployment phase rather than calendar week.

- [ ] vLLM serving Gemma 4 26B MoE in a container
- [ ] Prometheus scraping vLLM metrics
- [ ] Grafana dashboards (request rate, latency, KV cache, GPU)
- [ ] nginx reverse proxy with streaming-aware config
- [ ] HTTPS via Let's Encrypt or self-signed for local
- [ ] API key authentication and per-client rate limiting
- [ ] Per-token usage metering
- [ ] Slack-integrated alerting on SLO breaches
- [ ] Model hot-swap without full stack restart

## Repository layout

```
.
├── docker-compose.yml       # orchestration entry point
├── nginx/                   # reverse proxy config, streaming-aware
├── observability/
│   ├── prometheus/          # scrape config
│   └── grafana/             # dashboards, datasources
├── docs/                    # architecture notes, operational runbooks
└── README.md
```

## License

Apache License 2.0. See [LICENSE](./LICENSE).

## Related work

This repository is the deployment companion to a broader training program exploring inference optimization on commodity GPUs. Some design choices here reflect findings from that work — in particular, the preference for NVLink-connected GPU pairs over wider tensor parallelism, and sensitivity to KV cache sizing behavior in vLLM on models with hybrid attention.
