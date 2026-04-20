from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from .datasets import KNOWN_DATASETS, build_loader
from .engines import DEFAULT_SWEEP, KNOWN_ENGINES, SWEEP_PARAM, build_engine
from .envutil import load_dotenv
from .runner import aggregate, dump_json, run_engine_on_loader
from .sweep_runner import dump_sweep, run_sweep


def _cmd_run(args: argparse.Namespace) -> int:
    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())

    datasets = args.dataset or list(KNOWN_DATASETS)
    engines = args.engine or list(KNOWN_ENGINES)

    all_metrics = []
    for engine_name in engines:
        engine = build_engine(engine_name)
        try:
            for ds_name in datasets:
                if ds_name not in cfg:
                    print(f"[warn] dataset {ds_name} not in {args.config}", file=sys.stderr)
                    continue
                loader = build_loader(ds_name, cfg[ds_name])
                results = list(
                    run_engine_on_loader(engine, loader, args.max_seconds_per_dataset)
                )
                m = aggregate(results, engine_name)
                m.dataset = ds_name
                all_metrics.append(m)
        finally:
            engine.close()

    dump_json(all_metrics, Path(args.out))
    _render_table(all_metrics)
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    datasets = args.dataset or list(KNOWN_DATASETS)

    all_results = []
    for spec in args.engine_spec:
        if ":" in spec:
            engine_name, values_str = spec.split(":", 1)
            values = [float(x) for x in values_str.split(",")]
        else:
            engine_name = spec
            values = DEFAULT_SWEEP[engine_name]
        if engine_name not in SWEEP_PARAM:
            print(f"[warn] {engine_name} does not support sweep, skipping", file=sys.stderr)
            continue
        print(f"[sweep] {engine_name} over {SWEEP_PARAM[engine_name]}={values}")
        results = run_sweep(
            engine_name=engine_name,
            values=values,
            datasets=datasets,
            config=cfg,
            max_seconds=args.max_seconds_per_dataset,
        )
        all_results.extend(results)

    dump_sweep(all_results, Path(args.out))
    _render_sweep_table(all_results)
    return 0


def _render_sweep_table(results) -> None:
    console = Console()
    for r in results:
        title = f"{r.engine} / {r.dataset}   param={r.param_name}   trap-AUC={_fmt(r.auc_trap)}"
        table = Table(title=title)
        for col in ("value", "FPR", "TPR(recall)", "precision", "F1", "acc"):
            table.add_column(col)
        for p in r.points:
            table.add_row(
                f"{p.param_value:g}",
                f"{p.false_positive_rate:.3f}",
                f"{p.true_positive_rate:.3f}",
                _fmt(p.precision),
                _fmt(p.f1),
                f"{p.accuracy:.3f}",
            )
        console.print(table)


def _fmt(v):
    return "-" if v is None else f"{v:.3f}"


def _cmd_report(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.input).read_text())
    from .runner import DatasetMetrics

    metrics = [DatasetMetrics(**d) for d in data]
    _render_table(metrics)
    return 0


def _render_table(metrics: list) -> None:
    console = Console()
    table = Table(title="VAD benchmark")
    cols = (
        "dataset", "engine", "clips", "seconds", "speech%",
        "ROC-AUC", "acc", "precision", "recall", "F1", "FPR",
    )
    for col in cols:
        table.add_column(col)

    def fmt(v):
        return "-" if v is None else f"{v:.3f}"

    for m in metrics:
        speech_pct = (m.n_speech_frames / m.n_frames * 100) if m.n_frames else 0.0
        table.add_row(
            m.dataset,
            m.engine,
            str(m.n_clips),
            f"{m.total_seconds:.0f}",
            f"{speech_pct:.1f}",
            fmt(m.roc_auc),
            f"{m.accuracy:.3f}",
            fmt(m.precision),
            fmt(m.recall),
            fmt(m.f1),
            f"{m.false_positive_rate:.3f}",
        )
    console.print(table)


def main() -> int:
    p = argparse.ArgumentParser(prog="vad-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run benchmark")
    pr.add_argument("--config", default="configs/datasets.yaml")
    pr.add_argument("--dataset", nargs="+", choices=KNOWN_DATASETS)
    pr.add_argument("--engine", nargs="+", choices=KNOWN_ENGINES)
    pr.add_argument("--max-seconds-per-dataset", type=float, default=None)
    pr.add_argument("--out", default="results/run.json")
    pr.set_defaults(func=_cmd_run)

    prep = sub.add_parser("report", help="pretty-print a results JSON")
    prep.add_argument("input")
    prep.set_defaults(func=_cmd_report)

    ps = sub.add_parser(
        "sweep",
        help="run parameter sweep for binary-output engines (webrtc, aicoustics) "
        "and report trapezoidal AUC",
    )
    ps.add_argument("--config", default="configs/datasets.yaml")
    ps.add_argument("--dataset", nargs="+", choices=KNOWN_DATASETS)
    ps.add_argument(
        "engine_spec",
        nargs="+",
        help="engine[:v1,v2,...] e.g. aicoustics:2,4,6,8,10 webrtc:0,1,2,3 (defaults used if omitted)",
    )
    ps.add_argument("--max-seconds-per-dataset", type=float, default=None)
    ps.add_argument("--out", default="results/sweep.json")
    ps.set_defaults(func=_cmd_sweep)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
