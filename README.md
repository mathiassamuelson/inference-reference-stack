# inference-reference-stack

A reference deployment of a production-pattern LLM inference service on modest hardware.

Most published inference-serving guides assume datacenter GPUs (H100s, A100s) and hyperscaler-grade orchestration. This repository aims for the opposite end of the spectrum: what does a credible production inference stack look like when the hardware is four consumer RTX 3090s on a single workstation, and the operator is a team of one? The answer, it turns out, uses most of the same components — Triton Inference Server, vLLM, nginx, Prometheus, Grafana — just sized and configured differently.

## Status

Early construction. This is a living reference deployment that will grow over time. See [Roadmap](#roadmap) for what's here now and what's coming.

## Architecture

```
┌──────────────┐      ┌───────────┐      ┌─────────────────────┐
│  Client app  │─────▶│   nginx   │─────▶│  Triton Inference   │
│ (statmon-ai) │ HTTPS│  (proxy,  │ HTTP │  Server + vLLM      │
└──────────────┘      │   TLS)    │      │  backend            │
                      └─────┬─────┘      └──────────┬──────────┘
                            │                       │
                            │ access logs           │ /metrics
                            ▼                       ▼
                      ┌─────────────────────────────────┐
                      │  Prometheus  ──▶  Grafana       │
                      └─────────────────────────────────┘
```

All services run as containers, orchestrated via Docker Compose.

### Components

- **Triton Inference Server** with the vLLM backend — serves the model over an OpenAI-compatible API and exposes Prometheus metrics.
- **vLLM** — the inference engine inside Triton. Handles KV cache management, continuous batching, and PagedAttention.
- **nginx** — reverse proxy and TLS terminator. Streaming-aware configuration for server-sent events.
- **Prometheus** — metrics collection from Triton's metrics endpoint and nginx access logs.
- **Grafana** — dashboards for request rate, latency, GPU utilization, and KV cache pressure.

### Model

Currently serving **Gemma 4 26B A4B MoE** (`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`) — AWQ-INT4 quantized, running on two NVLink-connected RTX 3090s via tensor parallelism. Other models can be swapped in by editing the Triton model config; no code changes required.

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
cp .env.example .env     # edit as needed
docker compose up -d
```

Point your OpenAI-compatible client at `https://<host>/v1/` once TLS is configured, or `http://<host>:8000/v1/` for local development.

## Roadmap

Tracked by deployment phase rather than calendar week.

- [ ] Triton + vLLM serving Gemma 4 26B MoE in a container
- [ ] Prometheus scraping Triton metrics
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
├── triton/                  # Triton model repository + vLLM backend config
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
