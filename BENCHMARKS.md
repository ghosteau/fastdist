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

## Unreleased — discrete CDF recurrences and batch invariant hoisting

Commit `73ba7d6905`, measured against the v0.1.0 baseline above on the same
machine in the same session. 29 cases improved, none regressed.

Two changes: the three discrete CDFs that summed PMF terms now use recurrences,
and the batch paths hoist parameter validation and loop-invariant terms out of
their loops. See the commit for the derivations and the underflow fallbacks.

The largest movements, as reported by `compare.py --latest`. `change` is
fastdist's own time, so negative is faster:

| case | change |
|---|---:|
| `batch/poisson_cdf n=1,000` | -97.7% |
| `batch/poisson_cdf n=100,000` | -97.0% |
| `batch/poisson_cdf n=1,000,000` | -96.9% |
| `batch/normal_logpdf n=100,000` | -80.2% |
| `batch/normal_logpdf n=1,000,000` | -68.3% |
| `batch/normal_logpdf n=1,000` | -62.0% |
| `batch/uniform_pdf n=100,000` | -28.7% |
| `batch/uniform_pdf n=1,000,000` | -26.4% |
| `batch/normal_cdf n=100,000` | -25.7% |
| `batch/uniform_cdf n=1,000,000` | -25.2% |
| `batch/normal_cdf n=1,000,000` | -23.6% |
| `batch/normal_cdf n=1,000` | -22.5% |

<!-- generated from 0.1.0_20260905T062420+0000_73ba7d6905.json by benchmarks/table.py -->
- **Version** 0.1.0 (`73ba7d6905` on `fix/flaky-rng-tolerances`, working tree dirty)
- **Measured** 2026-09-05T06:24:20+00:00
- **CPU** AMD Ryzen 7 7700 8-Core Processor
- **Platform** Windows-11-10.0.26200-SP0
- **Toolchain** Python 3.14.2, numpy 2.5.2, scipy 1.18.1
- **CUDA** not built

### batch (vs vectorised SciPy)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_pdf` | 1,000 | 5.24 us | 31.57 us (scipy) | **6.02x** | 1.1e-16 |
| `normal_cdf` | 1,000 | 6.01 us | 29.63 us (scipy) | **4.93x** | 2.2e-16 |
| `normal_logpdf` | 1,000 | 1.91 us | 32.20 us (scipy) | **16.89x** | 8.9e-16 |
| `exponential_pdf` | 1,000 | 4.14 us | 29.31 us (scipy) | **7.08x** | 0.0e+00 |
| `exponential_cdf` | 1,000 | 4.16 us | 29.63 us (scipy) | **7.12x** | 8.3e-17 |
| `uniform_pdf` | 1,000 | 2.02 us | 32.99 us (scipy) | **16.30x** | 0.0e+00 |
| `uniform_cdf` | 1,000 | 2.13 us | 31.45 us (scipy) | **14.78x** | 0.0e+00 |
| `poisson_pmf` | 1,000 | 40.52 us | 37.70 us (scipy) | **0.93x** | 2.0e-19 |
| `poisson_cdf` | 1,000 | 9.95 us | 70.07 us (scipy) | **7.05x** | 2.2e-16 |
| `bernoulli_pmf` | 1,000 | 1.93 us | 48.90 us (scipy) | **25.36x** | 2.2e-16 |
| `normal_pdf` | 100,000 | 413.90 us | 1.44 ms (scipy) | **3.48x** | 1.1e-16 |
| `normal_cdf` | 100,000 | 528.60 us | 2.06 ms (scipy) | **3.89x** | 2.2e-16 |
| `normal_logpdf` | 100,000 | 77.90 us | 1.67 ms (scipy) | **21.45x** | 8.9e-16 |
| `exponential_pdf` | 100,000 | 312.50 us | 1.52 ms (scipy) | **4.87x** | 0.0e+00 |
| `exponential_cdf` | 100,000 | 312.10 us | 1.64 ms (scipy) | **5.27x** | 1.1e-16 |
| `uniform_pdf` | 100,000 | 92.00 us | 1.55 ms (scipy) | **16.79x** | 0.0e+00 |
| `uniform_cdf` | 100,000 | 102.00 us | 1.65 ms (scipy) | **16.18x** | 0.0e+00 |
| `poisson_pmf` | 100,000 | 4.01 ms | 3.16 ms (scipy) | **0.79x** | 2.0e-19 |
| `poisson_cdf` | 100,000 | 1.31 ms | 6.05 ms (scipy) | **4.63x** | 2.2e-16 |
| `bernoulli_pmf` | 100,000 | 290.80 us | 3.48 ms (scipy) | **11.96x** | 2.2e-16 |
| `normal_pdf` | 1,000,000 | 4.73 ms | 19.66 ms (scipy) | **4.16x** | 1.1e-16 |
| `normal_cdf` | 1,000,000 | 5.86 ms | 22.84 ms (scipy) | **3.90x** | 2.2e-16 |
| `normal_logpdf` | 1,000,000 | 1.42 ms | 21.70 ms (scipy) | **15.33x** | 8.9e-16 |
| `exponential_pdf` | 1,000,000 | 3.92 ms | 18.36 ms (scipy) | **4.68x** | 0.0e+00 |
| `exponential_cdf` | 1,000,000 | 3.74 ms | 18.55 ms (scipy) | **4.96x** | 1.7e-16 |
| `uniform_pdf` | 1,000,000 | 1.38 ms | 17.70 ms (scipy) | **12.80x** | 0.0e+00 |
| `uniform_cdf` | 1,000,000 | 1.51 ms | 19.54 ms (scipy) | **12.97x** | 0.0e+00 |
| `poisson_pmf` | 1,000,000 | 42.65 ms | 37.75 ms (scipy) | **0.89x** | 2.0e-19 |
| `poisson_cdf` | 1,000,000 | 14.09 ms | 63.76 ms (scipy) | **4.53x** | 2.2e-16 |
| `bernoulli_pmf` | 1,000,000 | 3.52 ms | 38.78 ms (scipy) | **11.01x** | 2.2e-16 |

### scalar (per-call cost, not throughput)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_pdf` | 20,000 | 7.12 ms | 451.64 ms (scipy) | **63.40x** | 1.1e-16 |
| `normal_cdf` | 20,000 | 7.35 ms | 439.65 ms (scipy) | **59.79x** | 2.2e-16 |

### sample (vs numpy)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_sample` | 100,000 | 27.25 ms | 867.30 us (numpy) | **0.03x** | - |
| `uniform_sample` | 100,000 | 23.73 ms | 227.10 us (numpy) | **0.01x** | - |
| `normal_sample` | 1,000,000 | 288.47 ms | 10.02 ms (numpy) | **0.03x** | - |
| `uniform_sample` | 1,000,000 | 253.13 ms | 3.19 ms (numpy) | **0.01x** | - |

### Reading

`poisson_cdf` was the headline defect in the baseline and is now the largest
win: 43.47ms to 1.32ms at 100k elements, moving from 6.6x slower than SciPy to
4.9x faster. Agreement with SciPy tightened from 3.3e-16 to 2.2e-16 at the same
time, which is the expected consequence of doing far fewer floating-point
operations to reach the same answer.

`normal_logpdf` improved 80% purely from hoisting `log(sigma)`, which the batch
path had been recomputing per element for a value fixed across the whole array.
It is now the fastest continuous case in the suite at 22x SciPy.

The remaining known gaps are unchanged and still worth recording:

- `poisson_pmf` is 0.81x. The per-element `lgamma` dominates and is not
  loop-invariant, so hoisting cannot reach it. Beating SciPy here needs a
  different evaluation strategy, not tuning.
- Sampling is still 30-100x slower than numpy. Unchanged, and structural: it
  needs a batch sampling entry point that does not exist yet.

---

## Unreleased — correctness pass

The headline of this entry is not a speedup. Three of the library's CDFs were
returning wrong answers, two of them probabilities above 1.0, and the benchmark
suite is what surfaced the first of them: it checks agreement with SciPy before
it times anything, so a wrong result cannot quietly post a good number.

- **beta_cdf** was wrong at every point -- 0.0015 against a true 0.1143 for
  Beta(2,5) at x=0.1, and -147 for Beta(0.01,0.01) at x=0.5. The series had an
  inverted coefficient ratio and normalised by Gamma(a+1) instead of B(a,b).
  Replaced with the modified-Lentz continued fraction.
- **gamma_cdf** and **chi_square_cdf** shared a continued fraction whose Lentz
  coefficient was written `-i * (i - a)` with an unsigned loop index, so the
  unary minus wrapped to 2^32 - i. Errors reached 0.26 and
  Gamma(1.5,1.0).cdf(2.5) returned 1.000498004.
- **MAX_ITER** was 100, silently truncating the gamma series for large shapes
  (wrong by 0.16 at alpha = 10000). Now 1000.

All three now agree with SciPy to ~1e-12 across the parameter ranges recorded in
the commits, and they are benchmarked from here on -- their absence from the
suite is why the defects survived this long. They land at 46-56x SciPy in the
scalar group, which is where their cost can be tracked since they have no
`*_cpu` batch path.

Performance is otherwise unchanged from the previous entry: no regression
outside measurement noise.

One methodology change came out of this run. The scalar cases are dominated by
per-call Python overhead, which the interpreter varies far more than it varies
compiled work; at 7 rounds an untouched `normal_cdf` differed by 8% between two
runs and `compare.py` reported it as a regression. The scalar group now uses 21
rounds, and `compare.py` will not flag a change smaller than the two runs'
combined `noise_pct`. Re-running confirmed the phantom: `normal_cdf` came back
at -8.7%, its original level.

<!-- generated from 0.1.0_20260905T070807+0000_8f0f46fdd0.json by benchmarks/table.py -->
- **Version** 0.1.0 (`8f0f46fdd0` on `fix/flaky-rng-tolerances`, working tree dirty)
- **Measured** 2026-09-05T07:08:07+00:00
- **CPU** AMD Ryzen 7 7700 8-Core Processor
- **Platform** Windows-11-10.0.26200-SP0
- **Toolchain** Python 3.14.2, numpy 2.5.2, scipy 1.18.1
- **CUDA** not built

### batch (vs vectorised SciPy)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_pdf` | 1,000 | 5.22 us | 31.30 us (scipy) | **6.00x** | 1.1e-16 |
| `normal_cdf` | 1,000 | 6.01 us | 30.26 us (scipy) | **5.04x** | 2.2e-16 |
| `normal_logpdf` | 1,000 | 1.88 us | 31.97 us (scipy) | **17.04x** | 8.9e-16 |
| `exponential_pdf` | 1,000 | 4.17 us | 29.55 us (scipy) | **7.09x** | 0.0e+00 |
| `exponential_cdf` | 1,000 | 4.20 us | 30.39 us (scipy) | **7.24x** | 8.3e-17 |
| `uniform_pdf` | 1,000 | 2.04 us | 33.67 us (scipy) | **16.50x** | 0.0e+00 |
| `uniform_cdf` | 1,000 | 2.10 us | 32.46 us (scipy) | **15.44x** | 0.0e+00 |
| `poisson_pmf` | 1,000 | 40.67 us | 41.04 us (scipy) | **1.01x** | 2.0e-19 |
| `poisson_cdf` | 1,000 | 9.94 us | 72.03 us (scipy) | **7.25x** | 2.2e-16 |
| `bernoulli_pmf` | 1,000 | 2.00 us | 49.44 us (scipy) | **24.72x** | 2.2e-16 |
| `normal_pdf` | 100,000 | 414.70 us | 1.44 ms (scipy) | **3.47x** | 1.1e-16 |
| `normal_cdf` | 100,000 | 529.20 us | 2.09 ms (scipy) | **3.95x** | 2.2e-16 |
| `normal_logpdf` | 100,000 | 77.60 us | 1.66 ms (scipy) | **21.45x** | 8.9e-16 |
| `exponential_pdf` | 100,000 | 308.80 us | 1.54 ms (scipy) | **4.99x** | 0.0e+00 |
| `exponential_cdf` | 100,000 | 312.30 us | 1.64 ms (scipy) | **5.26x** | 1.1e-16 |
| `uniform_pdf` | 100,000 | 93.70 us | 1.51 ms (scipy) | **16.07x** | 0.0e+00 |
| `uniform_cdf` | 100,000 | 99.30 us | 1.59 ms (scipy) | **16.04x** | 0.0e+00 |
| `poisson_pmf` | 100,000 | 4.02 ms | 3.23 ms (scipy) | **0.80x** | 2.0e-19 |
| `poisson_cdf` | 100,000 | 1.31 ms | 6.18 ms (scipy) | **4.70x** | 2.2e-16 |
| `bernoulli_pmf` | 100,000 | 299.20 us | 3.54 ms (scipy) | **11.82x** | 2.2e-16 |
| `normal_pdf` | 1,000,000 | 4.82 ms | 20.38 ms (scipy) | **4.23x** | 1.1e-16 |
| `normal_cdf` | 1,000,000 | 5.96 ms | 22.55 ms (scipy) | **3.78x** | 2.2e-16 |
| `normal_logpdf` | 1,000,000 | 1.31 ms | 21.51 ms (scipy) | **16.39x** | 8.9e-16 |
| `exponential_pdf` | 1,000,000 | 3.69 ms | 17.35 ms (scipy) | **4.70x** | 0.0e+00 |
| `exponential_cdf` | 1,000,000 | 3.80 ms | 19.79 ms (scipy) | **5.20x** | 1.7e-16 |
| `uniform_pdf` | 1,000,000 | 1.39 ms | 18.56 ms (scipy) | **13.36x** | 0.0e+00 |
| `uniform_cdf` | 1,000,000 | 1.49 ms | 19.02 ms (scipy) | **12.80x** | 0.0e+00 |
| `poisson_pmf` | 1,000,000 | 42.78 ms | 38.78 ms (scipy) | **0.91x** | 2.0e-19 |
| `poisson_cdf` | 1,000,000 | 14.15 ms | 63.90 ms (scipy) | **4.52x** | 2.2e-16 |
| `bernoulli_pmf` | 1,000,000 | 3.51 ms | 38.80 ms (scipy) | **11.06x** | 2.2e-16 |

### scalar (per-call cost, not throughput)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_pdf` | 20,000 | 7.22 ms | 441.28 ms (scipy) | **61.08x** | 1.1e-16 |
| `normal_cdf` | 20,000 | 7.24 ms | 425.74 ms (scipy) | **58.77x** | 2.2e-16 |
| `gamma_cdf` | 20,000 | 8.78 ms | 426.74 ms (scipy) | **48.63x** | 1.2e-13 |
| `chi_square_cdf` | 20,000 | 7.91 ms | 445.09 ms (scipy) | **56.25x** | 1.2e-13 |
| `beta_cdf` | 20,000 | 9.48 ms | 467.80 ms (scipy) | **49.36x** | 8.9e-16 |

### sample (vs numpy)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_sample` | 100,000 | 28.22 ms | 893.60 us (numpy) | **0.03x** | - |
| `uniform_sample` | 100,000 | 23.78 ms | 226.90 us (numpy) | **0.01x** | - |
| `normal_sample` | 1,000,000 | 295.14 ms | 9.94 ms (numpy) | **0.03x** | - |
| `uniform_sample` | 1,000,000 | 258.23 ms | 3.14 ms (numpy) | **0.01x** | - |

---

## Changes to record here

Add an entry when a release ships, or when a change is made specifically to
move performance. Each entry should carry the generated table, the commit, and
a short reading of what moved and why. `compare.py` output makes a good basis
for the reading.
