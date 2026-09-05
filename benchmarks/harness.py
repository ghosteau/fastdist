"""
Timing harness for the fastdist benchmark suite.

Separated from the benchmark definitions so the measurement methodology lives
in one place and can be reviewed on its own.

Methodology
-----------
Each case is timed as `repeat` independent rounds of `inner` calls. The round
total is divided by `inner` to get a per-call time, and the reported figure is
the **minimum** round rather than the mean.

Minimum is the right summary here. The quantity of interest is how long the
work takes; every source of noise on a shared machine (scheduler preemption,
frequency scaling, another process touching the cache) can only ever add time,
never remove it. The mean estimates "time under typical interference", which is
a property of the machine that day. The minimum estimates the work itself,
which is the thing a regression would move.

The spread between the minimum and the median is reported alongside as
`noise_pct`. It is not an error bar on the measurement -- it is a measure of
how quiet the machine was. A large value means the numbers are still usable but
the machine was busy; comparisons across runs with very different noise_pct
deserve suspicion.

A warmup round runs before timing and is discarded, so first-call costs (lazy
imports, page faults on the output buffer, branch predictor cold start) do not
land in the result.
"""

from __future__ import annotations

import json
import platform
import subprocess
import statistics
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Result:
    """One measured case."""

    group: str
    case: str
    n: int
    # Seconds per call, minimum over rounds.
    fastdist_s: float
    baseline_s: float | None
    baseline_name: str | None
    # How quiet the machine was, as (median - min) / min, in percent.
    fastdist_noise_pct: float
    baseline_noise_pct: float | None
    # Largest absolute difference between the two implementations' outputs.
    # None when there is no baseline to compare against.
    max_abs_diff: float | None

    @property
    def speedup(self) -> float | None:
        if self.baseline_s is None or self.fastdist_s == 0.0:
            return None
        return self.baseline_s / self.fastdist_s


def _time_one(fn: Callable[[], object], repeat: int, inner: int) -> tuple[float, float]:
    """Return (min seconds per call, noise percent)."""
    # Warmup, discarded.
    for _ in range(inner):
        fn()

    rounds = []
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        rounds.append((time.perf_counter() - start) / inner)

    best = min(rounds)
    median = statistics.median(rounds)
    noise = ((median - best) / best * 100.0) if best > 0 else 0.0
    return best, noise


def measure(
    group: str,
    case: str,
    n: int,
    fastdist_fn: Callable[[], object],
    baseline_fn: Callable[[], object] | None = None,
    baseline_name: str | None = None,
    repeat: int = 7,
    inner: int = 1,
) -> Result:
    """Time one case against an optional baseline and check they agree."""
    fd_s, fd_noise = _time_one(fastdist_fn, repeat, inner)

    base_s = base_noise = max_abs_diff = None
    if baseline_fn is not None:
        base_s, base_noise = _time_one(baseline_fn, repeat, inner)

        # A speedup only means something if both sides computed the same thing.
        # Any case whose outputs disagree is a bug in the benchmark or the
        # library, and reporting its timing would be misleading either way.
        max_abs_diff = _max_abs_diff(fastdist_fn(), baseline_fn())

    return Result(
        group=group,
        case=case,
        n=n,
        fastdist_s=fd_s,
        baseline_s=base_s,
        baseline_name=baseline_name,
        fastdist_noise_pct=fd_noise,
        baseline_noise_pct=base_noise,
        max_abs_diff=max_abs_diff,
    )


def _max_abs_diff(a, b) -> float:
    """Largest absolute elementwise difference, ignoring matching non-finites."""
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("inf")

    finite = np.isfinite(a) & np.isfinite(b)
    # Disagreeing on *where* the non-finites are is a real mismatch.
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return float("inf")
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(a[finite] - b[finite])))


def environment() -> dict:
    """Everything needed to judge whether two runs are comparable."""
    import numpy

    try:
        import scipy

        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = None

    sys.path.insert(0, str(REPO_ROOT / "python"))
    import fastdist
    import fastdist._fastdist as core

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fastdist_version": core.__version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "cuda_available": hasattr(core, "normal_pdf_cuda"),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy_version,
        "platform": platform.platform(),
        "processor": _cpu_name(),
        "machine": platform.machine(),
    }


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu_name() -> str:
    """platform.processor() is often empty or useless; try harder on each OS.

    Every probe here is best-effort. A missing CPU name is cosmetic -- it must
    never take the benchmark run down with it -- so each branch swallows its
    own failures and the function always returns a string.
    """
    if sys.platform == "win32":
        # wmic was removed in recent Windows 11 builds, so try the registry
        # first and only then fall back to the (vaguer) environment variable.
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            with key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass

        import os

        return os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "unknown"

    if sys.platform == "darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return platform.processor() or "unknown"

    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def write_report(results: list[Result], env: dict, out_dir: Path) -> Path:
    """Write one run to a JSON file named for the version and commit."""
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = (env.get("git_commit") or "nocommit")[:10]
    stamp = env["timestamp_utc"].replace(":", "").replace("-", "")
    path = out_dir / f"{env['fastdist_version']}_{stamp}_{commit}.json"

    payload = {
        "environment": env,
        "results": [
            {**asdict(r), "speedup": r.speedup} for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
