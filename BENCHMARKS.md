# fastdist benchmarks

Performance evidence log. One entry per release, plus entries for changes made
specifically to move performance.

The point of this file is to be checkable. Every number here was produced by
`benchmarks/run.py`, saved as JSON under `benchmarks/results/`, and rendered by
`benchmarks/table.py` rather than typed in by hand. The raw reports are
committed alongside, so any claim below can be traced to the run that produced
it, on a named CPU, at a named commit.

Results that are unflattering are recorded too. A log that only contains wins
is marketing, not evidence, and it would not catch a regression.

---

## Running the suite

```bash
pip install scipy          # baseline, not needed to build or use fastdist
python benchmarks/run.py
```

The run writes `benchmarks/results/<version>_<timestamp>_<commit>.json` and
prints a summary. `--quick` uses one array size for a fast check; `--no-write`
prints without saving.

To compare two runs:

```bash
python benchmarks/compare.py --latest
```

`compare.py` exits non-zero if any case regressed by more than 5%, so it can
gate a change. To render a report for this log:

```bash
python benchmarks/table.py --latest
```

---

## Method, and what the numbers do not say

The baseline is SciPy, because that is the realistic alternative for someone
who would otherwise use this library.

Each case is timed as several independent rounds, and the **minimum** round is
reported. Noise on a shared machine can only ever add time, so the minimum is
the best available estimate of the work itself; the mean would measure how busy
the machine was. Each report also records `noise_pct`, the gap between the
minimum and median, as a check on how quiet the run was.

Every case with a baseline verifies that both implementations produce the same
numbers before either is timed. The `max abs diff` column carries that
agreement, and it is the reason the speedups can be taken at face value: at
1e-16 the two are computing the same function.

Three caveats worth stating plainly:

- **These are single-machine numbers.** Every figure below is one desktop CPU
  on Windows. They are directionally useful, not a portable claim.
- **`scalar` is not a throughput number.** It measures the cost of one Python
  call into each library. fastdist wins by ~60x there, but that mostly reflects
  a thinner binding layer than SciPy's dispatch machinery, not faster math.
  Quote the `batch` numbers instead.
- **The comparison is single-threaded on both sides.** Neither library is
  parallelising these calls.

---

## v0.1.0 — initial baseline

First recorded measurement, taken at the point the benchmark suite was added so
that later work has something to be compared against.

<!-- generated from 0.1.0_20260905T061458+0000_7671393bc4.json by benchmarks/table.py -->
- **Version** 0.1.0 (`7671393bc4` on `fix/flaky-rng-tolerances`, working tree dirty)
- **Measured** 2026-09-05T06:14:58+00:00
- **CPU** AMD Ryzen 7 7700 8-Core Processor
- **Platform** Windows-11-10.0.26200-SP0
- **Toolchain** Python 3.14.2, numpy 2.5.2, scipy 1.18.1
- **CUDA** not built

### batch (vs vectorised SciPy)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_pdf` | 1,000 | 5.70 us | 31.52 us (scipy) | **5.53x** | 1.1e-16 |
| `normal_cdf` | 1,000 | 7.76 us | 29.88 us (scipy) | **3.85x** | 2.2e-16 |
| `normal_logpdf` | 1,000 | 5.01 us | 32.29 us (scipy) | **6.44x** | 8.9e-16 |
| `exponential_pdf` | 1,000 | 4.49 us | 29.98 us (scipy) | **6.68x** | 0.0e+00 |
| `exponential_cdf` | 1,000 | 4.40 us | 31.13 us (scipy) | **7.07x** | 8.3e-17 |
| `uniform_pdf` | 1,000 | 2.37 us | 33.17 us (scipy) | **13.98x** | 0.0e+00 |
| `uniform_cdf` | 1,000 | 2.41 us | 31.55 us (scipy) | **13.07x** | 0.0e+00 |
| `poisson_pmf` | 1,000 | 47.22 us | 38.13 us (scipy) | **0.81x** | 2.0e-19 |
| `poisson_cdf` | 1,000 | 425.69 us | 70.40 us (scipy) | **0.17x** | 3.3e-16 |
| `bernoulli_pmf` | 1,000 | 1.92 us | 51.16 us (scipy) | **26.67x** | 2.2e-16 |
| `normal_pdf` | 100,000 | 462.80 us | 1.92 ms (scipy) | **4.16x** | 1.1e-16 |
| `normal_cdf` | 100,000 | 711.40 us | 2.32 ms (scipy) | **3.26x** | 2.2e-16 |
| `normal_logpdf` | 100,000 | 393.50 us | 2.03 ms (scipy) | **5.16x** | 8.9e-16 |
| `exponential_pdf` | 100,000 | 345.00 us | 1.81 ms (scipy) | **5.24x** | 0.0e+00 |
| `exponential_cdf` | 100,000 | 336.00 us | 1.99 ms (scipy) | **5.93x** | 1.1e-16 |
| `uniform_pdf` | 100,000 | 129.00 us | 1.95 ms (scipy) | **15.11x** | 0.0e+00 |
| `uniform_cdf` | 100,000 | 131.20 us | 1.86 ms (scipy) | **14.18x** | 0.0e+00 |
| `poisson_pmf` | 100,000 | 4.72 ms | 3.58 ms (scipy) | **0.76x** | 2.0e-19 |
| `poisson_cdf` | 100,000 | 43.47 ms | 6.45 ms (scipy) | **0.15x** | 3.3e-16 |
| `bernoulli_pmf` | 100,000 | 290.60 us | 3.59 ms (scipy) | **12.37x** | 2.2e-16 |
| `normal_pdf` | 1,000,000 | 5.34 ms | 19.98 ms (scipy) | **3.74x** | 1.1e-16 |
| `normal_cdf` | 1,000,000 | 7.67 ms | 21.87 ms (scipy) | **2.85x** | 2.2e-16 |
| `normal_logpdf` | 1,000,000 | 4.47 ms | 21.47 ms (scipy) | **4.80x** | 8.9e-16 |
| `exponential_pdf` | 1,000,000 | 4.45 ms | 16.78 ms (scipy) | **3.77x** | 0.0e+00 |
| `exponential_cdf` | 1,000,000 | 4.36 ms | 22.64 ms (scipy) | **5.20x** | 1.7e-16 |
| `uniform_pdf` | 1,000,000 | 1.88 ms | 20.21 ms (scipy) | **10.75x** | 0.0e+00 |
| `uniform_cdf` | 1,000,000 | 2.02 ms | 20.22 ms (scipy) | **10.03x** | 0.0e+00 |
| `poisson_pmf` | 1,000,000 | 50.15 ms | 38.98 ms (scipy) | **0.78x** | 2.0e-19 |
| `poisson_cdf` | 1,000,000 | 451.33 ms | 65.91 ms (scipy) | **0.15x** | 3.3e-16 |
| `bernoulli_pmf` | 1,000,000 | 3.66 ms | 40.59 ms (scipy) | **11.08x** | 2.2e-16 |

### scalar (per-call cost, not throughput)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_pdf` | 20,000 | 8.55 ms | 488.55 ms (scipy) | **57.13x** | 1.1e-16 |
| `normal_cdf` | 20,000 | 7.15 ms | 477.79 ms (scipy) | **66.78x** | 2.2e-16 |

### sample (vs numpy)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_sample` | 100,000 | 27.29 ms | 871.20 us (numpy) | **0.03x** | - |
| `uniform_sample` | 100,000 | 23.74 ms | 227.30 us (numpy) | **0.01x** | - |
| `normal_sample` | 1,000,000 | 303.10 ms | 10.19 ms (numpy) | **0.03x** | - |
| `uniform_sample` | 1,000,000 | 273.73 ms | 3.30 ms (numpy) | **0.01x** | - |

### Reading of this baseline

**Where the library is genuinely fast.** The continuous PDF/CDF batch paths beat
SciPy by 3–15x, and agreement to ~1e-16 confirms both sides compute the same
function. The largest margins are on the cheapest distributions — `uniform` at
10–15x, `bernoulli` at 11–27x — which is what one would expect: when the math
per element is trivial, the fixed overhead SciPy pays per call dominates, and
fastdist has less of it. The margin narrows as arrays grow (`uniform_pdf` falls
from 14x at n=1,000 to 10.75x at n=1,000,000), which is the same effect seen
from the other side — at a million elements the actual arithmetic starts to
dominate the fixed cost.

**Where it is slower, and why.** `poisson_cdf` is 6.6x *slower* than SciPy, and
`poisson_pmf` about 25% slower. `poisson_cdf_scalar` sums the PMF from 0 to k,
calling `poisson_pmf_scalar` once per term, and each of those recomputes
`log(lambda)`, an `lgamma`, and an `exp`. For the benchmark's counts that is on
the order of twenty transcendental calls per element where a recurrence needs
none. This is a real defect, not a measurement artifact, and it is the clearest
optimisation target in the library.

**Sampling is 30–100x slower than numpy.** This is a structural gap, not a
tuning problem: fastdist crosses the Python/C++ boundary once per variate,
while numpy fills an entire array per call. Closing it needs a batch sampling
entry point — `normal_sample_batch(n)` returning an array — which does not
exist yet. Until it does, the honest statement is that this library is for
evaluating distribution functions, not for bulk variate generation.

---

## Changes to record here

Add an entry when a release ships, or when a change is made specifically to
move performance. Each entry should carry the generated table, the commit, and
a short reading of what moved and why. `compare.py` output makes a good basis
for the reading.
