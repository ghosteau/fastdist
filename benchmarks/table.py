"""
Render one benchmark report as a Markdown table, for pasting into BENCHMARKS.md.

    python benchmarks/table.py benchmarks/results/<report>.json
    python benchmarks/table.py --latest

Exists so entries in the evidence log are generated from the recorded JSON
rather than transcribed by hand -- a typo in a number nobody can check later is
worse than no number at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _fmt_time(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds * 1e6:.2f} us"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--group", help="only this group (batch, scalar, sample)")
    args = parser.parse_args()

    if args.latest:
        reports = sorted(RESULTS_DIR.glob("*.json"))
        if not reports:
            print(f"error: no reports in {RESULTS_DIR}")
            return 2
        path = reports[-1]
    elif args.report:
        path = args.report
    else:
        print("error: pass a report path, or --latest")
        return 2

    payload = json.loads(path.read_text(encoding="utf-8"))
    env = payload["environment"]

    print(f"<!-- generated from {path.name} by benchmarks/table.py -->")
    print()
    print(f"- **Version** {env['fastdist_version']} "
          f"(`{(env.get('git_commit') or '')[:10]}` on `{env.get('git_branch')}`"
          f"{', working tree dirty' if env.get('git_dirty') else ''})")
    print(f"- **Measured** {env['timestamp_utc']}")
    print(f"- **CPU** {env['processor']}")
    print(f"- **Platform** {env['platform']}")
    print(f"- **Toolchain** Python {env['python']}, numpy {env['numpy']}, scipy {env['scipy']}")
    print(f"- **CUDA** {'available' if env.get('cuda_available') else 'not built'}")
    print()

    groups = {}
    for r in payload["results"]:
        if args.group and r["group"] != args.group:
            continue
        groups.setdefault(r["group"], []).append(r)

    for group, rows in groups.items():
        print(f"### {group}")
        print()
        print("| case | n | fastdist | baseline | speedup | max abs diff |")
        print("|---|---:|---:|---:|---:|---:|")
        for r in rows:
            base = _fmt_time(r["baseline_s"]) if r["baseline_s"] is not None else "-"
            if r["baseline_name"]:
                base += f" ({r['baseline_name']})"
            speed = f"{r['speedup']:.2f}x" if r.get("speedup") else "-"
            diff = "-" if r["max_abs_diff"] is None else f"{r['max_abs_diff']:.1e}"
            print(f"| `{r['case']}` | {r['n']:,} | {_fmt_time(r['fastdist_s'])} "
                  f"| {base} | **{speed}** | {diff} |")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
