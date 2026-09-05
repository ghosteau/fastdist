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

The threshold for calling something a regression defaults to 5%, but a case is
only flagged when the change also exceeds the two runs' combined `noise_pct`
(the gap between each run's minimum and median round). A change smaller than
the noise says nothing, and the scalar cases -- dominated by Python call
overhead -- routinely swing 10% between runs on an otherwise idle machine.
Changes above the threshold but inside the noise are printed and marked rather
than counted.

`noise_pct` measures spread *within* a run, so it does not catch a run that was
uniformly slow. Treat a flagged case as a prompt to re-run rather than a
verdict; if it does not reproduce, it was the machine.
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
                        help="minimum percent change before a case is called out "
                             "(default 5); the run's measured noise raises this "
                             "further when the machine was busy")
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

        # A run is only as trustworthy as it was quiet. Each report records
        # noise_pct -- the gap between the minimum and median round -- and a
        # change smaller than the noise in the two runs combined says nothing.
        # Without this the scalar cases, which are dominated by Python call
        # overhead and routinely swing 10%, produce phantom regressions.
        noise = (b.get("fastdist_noise_pct") or 0.0) + (a.get("fastdist_noise_pct") or 0.0)
        limit = max(args.threshold, noise)

        flag = ""
        if change > limit:
            flag, _ = " REGRESSED", regressions.append((label, change))
        elif change < -limit:
            flag, _ = " faster", improvements.append((label, change))
        elif abs(change) > args.threshold:
            flag = f" (within noise, +/-{noise:.0f}%)"

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
