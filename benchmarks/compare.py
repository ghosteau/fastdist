"""
Compare two benchmark reports and print a regression table.

    python benchmarks/compare.py BEFORE.json AFTER.json
    python benchmarks/compare.py --latest              # newest two reports

Reads the JSON written by run.py. Cases are matched on (group, case, n), so
adding or removing benchmarks between runs is fine -- unmatched cases are
listed separately rather than silently dropped.

Reading the output
------------------
`change` is the change in fastdist's own time: negative is faster. It is the
number to look at when judging a code change.

`speedup` columns are fastdist against the baseline library in each report. A
change there can come from either side, so a moved speedup with an unmoved
`change` means the baseline moved, not this library.

The threshold for calling something a regression defaults to 5%, which is
comfortably above the noise on a quiet machine. Check the `noise_pct` field in
the reports before trusting a smaller difference: if either run was noisy, the
comparison is not meaningful at that resolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    index = {(r["group"], r["case"], r["n"]): r for r in payload["results"]}
    return payload["environment"], index


def _fmt_time(seconds: float) -> str:
    if seconds >= 1e-3:
        return f"{seconds * 1e3:8.2f}ms"
    return f"{seconds * 1e6:8.2f}us"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", nargs="?", type=Path)
    parser.add_argument("after", nargs="?", type=Path)
    parser.add_argument("--latest", action="store_true", help="use the newest two reports")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="percent change before a case is called out (default 5)")
    args = parser.parse_args()

    if args.latest:
        reports = sorted(RESULTS_DIR.glob("*.json"))
        if len(reports) < 2:
            return _fail(f"need two reports in {RESULTS_DIR}, found {len(reports)}")
        before_path, after_path = reports[-2], reports[-1]
    elif args.before and args.after:
        before_path, after_path = args.before, args.after
    else:
        return _fail("pass two report paths, or --latest")

    before_env, before = load(before_path)
    after_env, after = load(after_path)

    print(f"before  {before_path.name}")
    print(f"        {before_env['fastdist_version']} @ {(before_env.get('git_commit') or '')[:10]}"
          f"  {before_env['timestamp_utc']}")
    print(f"after   {after_path.name}")
    print(f"        {after_env['fastdist_version']} @ {(after_env.get('git_commit') or '')[:10]}"
          f"  {after_env['timestamp_utc']}")

    if before_env.get("processor") != after_env.get("processor"):
        print("\n!! different CPUs -- these reports are not comparable")
        print(f"   before: {before_env.get('processor')}")
        print(f"   after:  {after_env.get('processor')}")

    print()
    header = f"{'case':<34} {'before':>11} {'after':>11} {'change':>9}  {'speedup':>16}"
    print(header)
    print("-" * len(header))

    regressions, improvements = [], []

    for key in sorted(before.keys() & after.keys()):
        b, a = before[key], after[key]
        group, case, n = key

        change = (a["fastdist_s"] - b["fastdist_s"]) / b["fastdist_s"] * 100.0
        speed = ""
        if b.get("speedup") and a.get("speedup"):
            speed = f"{b['speedup']:6.2f}x -> {a['speedup']:6.2f}x"

        label = f"{group}/{case} n={n:,}"
        flag = ""
        if change > args.threshold:
            flag, _ = " REGRESSED", regressions.append((label, change))
        elif change < -args.threshold:
            flag, _ = " faster", improvements.append((label, change))

        print(f"{label:<34} {_fmt_time(b['fastdist_s']):>11} {_fmt_time(a['fastdist_s']):>11} "
              f"{change:+8.1f}%  {speed:>16}{flag}")

    only_before = before.keys() - after.keys()
    only_after = after.keys() - before.keys()
    for label, keys in (("only in before", only_before), ("only in after", only_after)):
        if keys:
            print(f"\n{label}:")
            for group, case, n in sorted(keys):
                print(f"  {group}/{case} n={n:,}")

    print()
    if regressions:
        print(f"{len(regressions)} regression(s) beyond {args.threshold:g}%:")
        for label, change in sorted(regressions, key=lambda t: -t[1]):
            print(f"  {label}  {change:+.1f}%")
    if improvements:
        print(f"{len(improvements)} improvement(s) beyond {args.threshold:g}%:")
        for label, change in sorted(improvements, key=lambda t: t[1]):
            print(f"  {label}  {change:+.1f}%")
    if not regressions and not improvements:
        print(f"no case moved more than {args.threshold:g}%")

    return 1 if regressions else 0


def _fail(message: str) -> int:
    print(f"error: {message}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
