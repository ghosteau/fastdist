"""
The fastdist benchmark suite.

Run from the repo root:

    python benchmarks/run.py                 # full suite, writes a JSON report
    python benchmarks/run.py --quick         # fewer sizes, for a fast check
    python benchmarks/run.py --no-write      # print only, write nothing

Results land in benchmarks/results/ as one JSON file per run, tagged with the
version, commit and machine. benchmarks/compare.py turns two of those into a
regression table, and BENCHMARKS.md is the curated log across releases.

What is being compared
----------------------
The baseline is SciPy, because that is what someone reaching for this library
would otherwise use. Comparisons are grouped by how fair they are:

  batch   fastdist's *_cpu entry points against the equivalent vectorised SciPy
          call. Both take a numpy array and return one, both do the loop in
          compiled code, and neither pays per-element Python overhead. This is
          the honest headline comparison.

  scalar  fastdist's *_scalar entry points against SciPy called on one value at
          a time. Both sides pay Python call overhead per element, so this
          measures the cost of a single call rather than throughput. It is
          reported because users do write scalar loops, but it flatters
          whichever library has the thinner binding layer and should not be
          quoted as a throughput number.

  cuda    The *_cuda entry points against the *_cpu ones on the same input,
          so the number is the speedup the GPU backend buys over this library's
          own CPU path -- the decision a caller actually faces. Skipped
          entirely unless the extension was built with FASTDIST_ENABLE_CUDA.
          GPU timings include the host-to-device copy and the copy back,
          because a caller cannot avoid those.

  sample  Drawing variates. fastdist samples one value per call, while numpy
          fills an array in one call, so numpy is expected to win by a wide
          margin. It is measured anyway: this is a real gap in the library and
          the log should record it rather than quietly omit it.

Every case with a baseline also checks that the two implementations agree
numerically (see harness.max_abs_diff). A speedup on a wrong answer is not a
speedup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import Result, environment, measure, write_report  # noqa: E402

try:
    from scipy import stats as sps
except ImportError:  # pragma: no cover
    sys.exit("benchmarks require scipy: pip install scipy")

import fastdist._fastdist as core  # noqa: E402

SIZES = (1_000, 100_000, 1_000_000)
QUICK_SIZES = (100_000,)
SCALAR_N = 20_000


# ---------------------------------------------------------------------------
# Batch: fastdist *_cpu vs vectorised scipy
# ---------------------------------------------------------------------------
def batch_cases(sizes):
    """(case name, fastdist callable factory, scipy callable factory).

    The trailing 0.0 on every *_cpu call is step_size. The C++ headers declare
    a default for it but the pybind11 bindings do not expose one, so it has to
    be passed explicitly; 0.0 means "evaluate x as given".
    """
    rng = np.random.default_rng(20260905)

    for n in sizes:
        x_real = rng.normal(0.0, 1.0, n)
        x_pos = np.abs(rng.normal(2.0, 1.0, n)) + 0.05
        x_unit = rng.uniform(0.01, 0.99, n)
        k_count = rng.integers(0, 20, n).astype(float)
        # bernoulli_pmf_batch takes int32, and poisson takes an int step_size:
        # the discrete batch entry points are not typed uniformly with the
        # continuous ones.
        k_binary = rng.integers(0, 2, n).astype(np.int32)

        yield from [
            ("normal_pdf", n,
             lambda x=x_real: core.normal_pdf_cpu(x, 0.0, 1.0, 0.0),
             lambda x=x_real: sps.norm.pdf(x, 0.0, 1.0)),
            ("normal_cdf", n,
             lambda x=x_real: core.normal_cdf_cpu(x, 0.0, 1.0, 0.0),
             lambda x=x_real: sps.norm.cdf(x, 0.0, 1.0)),
            ("normal_logpdf", n,
             lambda x=x_real: core.normal_logpdf_cpu(x, 0.0, 1.0, 0.0),
             lambda x=x_real: sps.norm.logpdf(x, 0.0, 1.0)),
            ("exponential_pdf", n,
             lambda x=x_pos: core.exponential_pdf_cpu(x, 2.0, 0.0),
             lambda x=x_pos: sps.expon.pdf(x, scale=0.5)),
            ("exponential_cdf", n,
             lambda x=x_pos: core.exponential_cdf_cpu(x, 2.0, 0.0),
             lambda x=x_pos: sps.expon.cdf(x, scale=0.5)),
            ("uniform_pdf", n,
             lambda x=x_real: core.uniform_pdf_cpu(x, -3.0, 3.0, 0.0),
             lambda x=x_real: sps.uniform.pdf(x, loc=-3.0, scale=6.0)),
            ("uniform_cdf", n,
             lambda x=x_real: core.uniform_cdf_cpu(x, -3.0, 3.0, 0.0),
             lambda x=x_real: sps.uniform.cdf(x, loc=-3.0, scale=6.0)),
            ("poisson_pmf", n,
             lambda x=k_count: core.poisson_pmf_cpu(x, 4.0, 0),
             lambda x=k_count: sps.poisson.pmf(x, 4.0)),
            ("poisson_cdf", n,
             lambda x=k_count: core.poisson_cdf_cpu(x, 4.0, 0),
             lambda x=k_count: sps.poisson.cdf(x, 4.0)),
            ("bernoulli_pmf", n,
             lambda x=k_binary: core.bernoulli_pmf_cpu(x, 0.3, 0),
             lambda x=k_binary: sps.bernoulli.pmf(x, 0.3)),
        ]


# ---------------------------------------------------------------------------
# Scalar: per-call cost
# ---------------------------------------------------------------------------
def scalar_cases():
    xs = np.random.default_rng(7).normal(0.0, 1.0, SCALAR_N)

    def fd_loop(fn, *args):
        return lambda: [fn(float(v), *args) for v in xs]

    def sp_loop(fn, *args, **kw):
        return lambda: [float(fn(float(v), *args, **kw)) for v in xs]

    return [
        ("normal_pdf", SCALAR_N,
         fd_loop(core.normal_pdf_scalar, 0.0, 1.0),
         sp_loop(sps.norm.pdf, 0.0, 1.0)),
        ("normal_cdf", SCALAR_N,
         fd_loop(core.normal_cdf_scalar, 0.0, 1.0),
         sp_loop(sps.norm.cdf, 0.0, 1.0)),
    ]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_cases(sizes):
    for n in sizes:
        rng = np.random.default_rng(11)
        yield ("normal_sample", n,
               lambda n=n: [core.normal_sample(0.0, 1.0) for _ in range(n)],
               lambda n=n, rng=rng: rng.normal(0.0, 1.0, n))
        yield ("uniform_sample", n,
               lambda n=n: [core.uniform_sample(0.0, 1.0) for _ in range(n)],
               lambda n=n, rng=rng: rng.uniform(0.0, 1.0, n))


# ---------------------------------------------------------------------------
# CUDA: *_cuda against this library's own *_cpu path
# ---------------------------------------------------------------------------
def cuda_cases(sizes):
    """Empty unless the extension was built with the CUDA backend."""
    if not hasattr(core, "normal_pdf_cuda"):
        return

    rng = np.random.default_rng(31337)
    for n in sizes:
        x_real = rng.normal(0.0, 1.0, n)
        x_pos = np.abs(rng.normal(2.0, 1.0, n)) + 0.05

        yield ("normal_pdf", n,
               lambda x=x_real: core.normal_pdf_cuda(x, 0.0, 1.0, 0.0),
               lambda x=x_real: core.normal_pdf_cpu(x, 0.0, 1.0, 0.0))
        yield ("normal_cdf", n,
               lambda x=x_real: core.normal_cdf_cuda(x, 0.0, 1.0, 0.0),
               lambda x=x_real: core.normal_cdf_cpu(x, 0.0, 1.0, 0.0))
        yield ("normal_logpdf", n,
               lambda x=x_real: core.normal_logpdf_cuda(x, 0.0, 1.0),
               lambda x=x_real: core.normal_logpdf_cpu(x, 0.0, 1.0, 0.0))
        yield ("exponential_pdf", n,
               lambda x=x_pos: core.exponential_pdf_cuda(x, 2.0, 0.0),
               lambda x=x_pos: core.exponential_pdf_cpu(x, 2.0, 0.0))
        yield ("uniform_pdf", n,
               lambda x=x_real: core.uniform_pdf_cuda(x, -3.0, 3.0, 0.0),
               lambda x=x_real: core.uniform_pdf_cpu(x, -3.0, 3.0, 0.0))


def run(sizes, sample_sizes) -> list[Result]:
    results: list[Result] = []

    for case, n, fd, sp in batch_cases(sizes):
        # Big arrays are slow enough that one call per round is plenty; small
        # ones need repetition to rise above timer resolution.
        inner = 50 if n <= 1_000 else 1
        results.append(measure("batch", case, n, fd, sp, "scipy", inner=inner))
        print(f"  batch  {case:<18} n={n:<9,} {_fmt(results[-1])}")

    for case, n, fd, sp in scalar_cases():
        results.append(measure("scalar", case, n, fd, sp, "scipy"))
        print(f"  scalar {case:<18} n={n:<9,} {_fmt(results[-1])}")

    for case, n, gpu, cpu in cuda_cases(sizes):
        # The "fastdist" column is the GPU path and the baseline is the CPU
        # path, so `speedup` reads as "how much the GPU buys over the CPU".
        # measure() also checks the two agree numerically, which is the part
        # worth having: a kernel that is fast and wrong is the failure mode.
        results.append(measure("cuda", case, n, gpu, cpu, "fastdist-cpu"))
        print(f"  cuda   {case:<18} n={n:<9,} {_fmt(results[-1])}")

    for case, n, fd, np_fn in sample_cases(sample_sizes):
        # No correctness check: both draw from their own RNG, so the outputs
        # are different random numbers by construction. measure() would flag
        # that as a mismatch, so the baseline is timed as a separate case.
        fd_result = measure("sample", case, n, fd)
        np_result = measure("sample", case, n, np_fn)
        fd_result.baseline_s = np_result.fastdist_s
        fd_result.baseline_name = "numpy"
        fd_result.baseline_noise_pct = np_result.fastdist_noise_pct
        results.append(fd_result)
        print(f"  sample {case:<18} n={n:<9,} {_fmt(fd_result)}")

    return results


def _fmt(r: Result) -> str:
    speed = r.speedup
    verdict = "-" if speed is None else f"{speed:6.2f}x"
    diff = "" if r.max_abs_diff is None else f"  maxdiff={r.max_abs_diff:.2e}"
    return f"fastdist={r.fastdist_s * 1e6:10.2f}us  {verdict}{diff}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="one array size only")
    parser.add_argument("--no-write", action="store_true", help="do not save a report")
    args = parser.parse_args()

    env = environment()
    print("fastdist benchmark suite")
    print(f"  version   {env['fastdist_version']}  ({env['git_branch']}@{(env['git_commit'] or '')[:10]}"
          f"{', dirty' if env['git_dirty'] else ''})")
    print(f"  cpu       {env['processor']}")
    print(f"  python    {env['python']}  numpy {env['numpy']}  scipy {env['scipy']}")
    print(f"  cuda      {'available' if env['cuda_available'] else 'not built'}")
    print()

    sizes = QUICK_SIZES if args.quick else SIZES
    sample_sizes = (100_000,) if args.quick else (100_000, 1_000_000)
    results = run(sizes, sample_sizes)

    mismatched = [r for r in results if r.max_abs_diff is not None and r.max_abs_diff > 1e-9]
    if mismatched:
        print("\nWARNING: outputs disagree with the baseline, timings below are not comparable:")
        for r in mismatched:
            print(f"  {r.group}/{r.case} n={r.n}: max abs diff {r.max_abs_diff:.3e}")

    if not args.no_write:
        path = write_report(results, env, REPO_ROOT / "benchmarks" / "results")
        print(f"\nwrote {path.relative_to(REPO_ROOT)}")

    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
