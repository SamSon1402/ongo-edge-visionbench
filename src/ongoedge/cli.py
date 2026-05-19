"""Command-line interface.

Usage:
    ongoedge run --target host-cpu --models yolo11n,nanodet-plus \
                 --precision fp16 --frames 200

Pretty output with rich. Same code paths as the API → no skew.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from ongoedge import models as model_registry
from ongoedge.bench import Runner
from ongoedge.core import BenchPlan, Frame, Precision, Target
from ongoedge.runtimes.mock import MockRuntime
from ongoedge.telemetry.fixture import FixtureProbe

app = typer.Typer(
    name="ongoedge",
    help="Edge CV benchmark suite for the Ongo companion robot.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    target: Target = typer.Option(Target.HOST_CPU, "--target", "-t"),
    models: str = typer.Option("yolo11n", "--models", "-m", help="comma-separated"),
    precision: Precision = typer.Option(Precision.FP16, "--precision", "-p"),
    frames: int = typer.Option(200, "--frames", "-n", min=10, max=5000),
    warmup: int = typer.Option(10, "--warmup", "-w", min=0, max=200),
    out: Path = typer.Option(Path("reports"), "--out", "-o"),
) -> None:
    """Run a benchmark sweep and print a Pareto-sorted table."""
    model_ids = [m.strip() for m in models.split(",") if m.strip()]
    unknown = [m for m in model_ids if m not in model_registry.REGISTRY]
    if unknown:
        console.print(f"[red]unknown models:[/red] {unknown}")
        console.print(f"available: {sorted(model_registry.REGISTRY)}")
        sys.exit(1)

    results = []
    for model_id in model_ids:
        plan = BenchPlan(
            target=target,
            model_id=model_id,
            precision=precision,
            input_h=480,
            input_w=640,
            num_frames=frames,
            warmup_frames=warmup,
        )
        console.print(f"[cyan]▶[/cyan] {plan.plan_id}")
        runner = Runner(
            runtime=MockRuntime(
                model_id=model_id,
                target=target,
                precision=precision,
                baseline_ms=_baseline(model_id, precision),
            ),
            probe=FixtureProbe(target=target),
            frame_source=_make_frame,
        )
        results.append(runner.execute(plan))

    _print_table(results)
    out.mkdir(parents=True, exist_ok=True)
    # JSON dump for downstream tooling
    import json

    (out / "results.json").write_text(
        json.dumps(
            [
                {
                    "plan_id": r.plan.plan_id,
                    "p50_ms": r.p50_ms,
                    "p99_ms": r.p99_ms,
                    "fps": r.fps,
                    "peak_ram_mb": r.peak_ram_mb,
                    "mean_power_w": r.mean_power_w,
                }
                for r in results
            ],
            indent=2,
        )
    )
    console.print(f"\n[green]✓[/green] wrote {out / 'results.json'}")


@app.command()
def models_list() -> None:
    """List registered models."""
    t = Table(title="Registered models")
    t.add_column("model_id", style="cyan")
    t.add_column("shape")
    t.add_column("description")
    for m in model_registry.REGISTRY.values():
        t.add_row(m.model_id, f"{m.input_w}×{m.input_h}", m.description)
    console.print(t)


# ── helpers ─────────────────────────────────────────────────────────


def _baseline(model_id: str, precision: Precision) -> float:
    base = {
        "yolo11n": 11.5,
        "yolov8n": 12.7,
        "nanodet-plus": 9.0,
        "mobilevit-xs": 18.0,
        "efficientvit-b0": 15.5,
    }.get(model_id, 14.0)
    mult = {Precision.FP32: 1.6, Precision.FP16: 1.0, Precision.INT8: 0.7, Precision.INT4: 0.55}
    return base * mult[precision]


def _make_frame(plan: BenchPlan) -> Frame:
    px = np.full((plan.input_h, plan.input_w, 3), 96, dtype=np.uint8)
    return Frame(width=plan.input_w, height=plan.input_h, pixels=px)


def _print_table(results: list) -> None:
    t = Table(title="Sweep results (Pareto-sorted)")
    t.add_column("model", style="cyan")
    t.add_column("precision")
    t.add_column("p50 ms", justify="right")
    t.add_column("p99 ms", justify="right")
    t.add_column("fps", justify="right", style="green")
    t.add_column("ram MB", justify="right")
    t.add_column("power W", justify="right")
    t.add_column("score", justify="right")
    for r in sorted(results, key=lambda x: x.frontier_score(), reverse=True):
        t.add_row(
            r.plan.model_id,
            r.plan.precision.value.upper(),
            f"{r.p50_ms:.1f}",
            f"{r.p99_ms:.1f}",
            f"{r.fps:.0f}",
            f"{r.peak_ram_mb:.0f}",
            f"{r.mean_power_w:.2f}",
            f"{r.frontier_score():.3f}",
        )
    console.print(t)


if __name__ == "__main__":
    app()
