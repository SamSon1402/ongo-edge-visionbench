# OngoEdge-VisionBench

> Edge CV benchmark suite for the Ongo companion robot.
> Built for InteractionLabs by Sameer M. — Paris, May 2026.

A reproducible benchmark harness that sweeps small CV models across edge
runtimes (ONNX Runtime / TFLite / TensorRT) and edge targets (Jetson Orin,
Raspberry Pi 5, Coral, RK3588) and produces a Pareto-front report:
latency × power × accuracy.

Designed to answer the only question that matters for an always-on,
privacy-first robot: **which model do we actually ship, on which device,
at which precision, today?**

---

## Why this exists

> *"Edge ML for Robotics: fine-tuning and deploying computer vision and
> audio models on-device for real-world robotics use cases."*
> — InteractionLabs, Founding ML Engineer JD

Ongo's on-device privacy story (magnetic sunglasses, no cloud video)
only holds if perception actually fits the SoC. This repo is the
measurement layer for that promise.

---

## What's in this drop

This is a **walking skeleton** — the module boundaries, type contracts,
core abstractions, FastAPI surface, and tests are real and pass. The
heavy lifting (TRT engine builder, full INT8 calibration) is stubbed
behind protocols so a TRT engineer can drop in without touching the
benchmark logic.

```
src/ongoedge/
  core/        # domain types, errors, protocols (no I/O)
  runtimes/    # ONNXRuntime, TFLite, TensorRT adapters
  models/      # registry: YOLO11n, MobileViT, EfficientViT, NanoDet
  bench/       # the sweep engine + telemetry
  api/         # FastAPI: /run, /results, /ws (live stream)
  reports/     # markdown + html generators
```

---

## Quick start

```bash
make install        # poetry / pip install -e ".[dev]"
make test           # pytest -q
make bench          # run a small smoke sweep on the host CPU
make api            # uvicorn ongoedge.api:app
make docker         # build Jetson-targeted image
```

A 30-second smoke run on a laptop:

```bash
ongoedge run --target host-cpu --models yolo11n --precision fp16 --frames 100
```

Produces `reports/2026_05_18_host-cpu.html` and `reports/2026_05_18_host-cpu.json`.

---

## Design notes

**No globals, no singletons.** Every benchmark run is a value of
`BenchPlan`, executed by a `Runner` against a `Target`. The same plan
is byte-identical reproducible across machines.

**Hardware via dependency injection.** `TelemetryProbe` is a protocol;
on Jetson it reads `tegrastats`, on Pi it reads `vcgencmd`, in CI it
returns a deterministic fixture. Tests never touch hardware.

**Latency is the contract.** Every model exposes `infer(frame) -> Detection`
with a per-call timing tuple. p50 and p99 are computed from the raw
samples, not the rolling mean — because Ongo's user experience is set by
the bad tail, not the average.

---

## Status

| Area | State |
|---|---|
| Core types & runner | ✅ done |
| ONNX Runtime adapter | ✅ done |
| TFLite adapter | 🟡 interface done, kernel binding TODO |
| TensorRT adapter | 🟡 interface done, builder TODO |
| Telemetry — Jetson | ✅ tegrastats parser done |
| Telemetry — Pi 5 | 🟡 vcgencmd parser TODO |
| FastAPI surface | ✅ done |
| WS live stream | ✅ done |
| HTML report | ✅ done |
| Calibration (INT8) | 🟡 plumbing only |

The 🟡 items are deliberate — they're hardware-specific and need real
silicon to verify. The structure is in place so they're each ~half a
day's work once we have the targets in hand.

---

— Sameer M. · samson1402.github.io · sameer@…
