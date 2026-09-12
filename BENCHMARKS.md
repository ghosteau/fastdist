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

There are two baselines. `scipy.stats` is what a user would otherwise call,
and it is the comparison most people mean. The `primitives` group is the same
quantity written directly with numpy or `scipy.special`, which is a much harder
target: most of the margin over `scipy.stats` is that library's generic
distribution machinery rather than faster arithmetic. Both are reported for
every case, because quoting only the first would oversell the library.

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

## Unreleased — CUDA backend measured, CDF tail accuracy

Commit `f98eb9f142`. The first run of the CUDA backend on real hardware, and the
speed cost of making two CDFs accurate in their tails. Compared against the
correctness-pass report above, taken on the same machine.

### The GPU path is slower than the CPU path at every size

**Superseded.** These GPU figures were measured on an idle, downclocked card and
are wrong. See the correction entry below.

Until this branch the CUDA backend did not build on Windows, and once built it
could not be imported, and once imported it crashed on its first call. With
those fixed it runs, agrees with the CPU path to machine precision, and loses
to it everywhere: 0.06x to 0.89x the CPU path's speed across
the cases and sizes below.

The kernels are not the bottleneck. GPU time barely depends on which function
runs, and `normal_pdf` at n = 1,000,000 moves 16 MB through the device in
11.1 ms -- about 1.4 GB/s, a small fraction of what the bus sustains. The
executor copies from pageable host memory across four streams; pinned buffers
and fewer, larger transfers are the obvious next step, and should be measured
rather than assumed.

This matters beyond the benchmark: the Python classes dispatch to CUDA
automatically from n = 100,000, so on a CUDA build they currently choose the
slower path.

### Tail accuracy cost normal_cdf most of its lead

`normal_cdf` returned exactly 0 below about -10 sigma and `exponential_cdf`
returned 0 for very small arguments. Both now use erfc / expm1 where the
naive form cancels, and the cheaper form elsewhere. Against the pre-branch
report:

- `normal_cdf` n=100,000: 529 us -> 892 us (+69%)
- `normal_cdf` n=1,000,000: 5963 us -> 9663 us (+62%)
- `exponential_cdf` n=100,000: 312 us -> 360 us (+15%)

`normal_cdf` is still faster than SciPy, and now correct where p-values live.
It recovered much less than expected when the tail-safe form was restricted to
the lower tail, which suggests the cost is not erfc itself; lost loop
auto-vectorisation is a plausible cause, not a confirmed one.

### Discarded runs

Two runs on this branch were disturbed by other load on the machine and are
not recorded. In the first, untouched functions came out 45-89% slower while
reporting within-run noise under 5% -- the blind spot `compare.py` documents,
since noise_pct cannot see a run that is uniformly slow. In the second, only
the cheapest 1M-element cases (uniform, ~1.5 ms a call) drifted, by 15-25%.
Both were re-measured case by case and confirmed as noise. Batch cases now
take the minimum of 15 rounds rather than 7, so a brief burst of background
load is less likely to cover every round of a short case. Before this entry
was written, every batch case this branch did not touch was checked against
the pre-branch report; the largest drift was +4.9% (uniform_pdf, n=1,000,000).

<!-- generated from 0.1.0_20260911T024546+0000_f98eb9f142.json by benchmarks/table.py -->
- **Version** 0.1.0 (`f98eb9f142` on `chore/release-prep`, working tree dirty)
- **Measured** 2026-09-11T02:45:46+00:00
- **CPU** AMD Ryzen 7 7700 8-Core Processor
- **Platform** Windows-11-10.0.26200-SP0
- **Toolchain** Python 3.14.2, numpy 2.5.2, scipy 1.18.1
- **CUDA** available

### batch (vs vectorised SciPy)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_pdf` | 1,000 | 5.23 us | 31.01 us (scipy) | **5.93x** | 1.1e-16 |
| `normal_cdf` | 1,000 | 6.79 us | 29.65 us (scipy) | **4.37x** | 2.2e-16 |
| `normal_logpdf` | 1,000 | 1.89 us | 31.27 us (scipy) | **16.56x** | 8.9e-16 |
| `exponential_pdf` | 1,000 | 4.16 us | 28.87 us (scipy) | **6.94x** | 0.0e+00 |
| `exponential_cdf` | 1,000 | 4.45 us | 30.02 us (scipy) | **6.74x** | 1.1e-16 |
| `uniform_pdf` | 1,000 | 2.06 us | 32.63 us (scipy) | **15.86x** | 0.0e+00 |
| `uniform_cdf` | 1,000 | 2.10 us | 32.09 us (scipy) | **15.25x** | 0.0e+00 |
| `poisson_pmf` | 1,000 | 40.62 us | 37.69 us (scipy) | **0.93x** | 2.0e-19 |
| `poisson_cdf` | 1,000 | 9.95 us | 71.15 us (scipy) | **7.15x** | 2.2e-16 |
| `bernoulli_pmf` | 1,000 | 2.00 us | 48.84 us (scipy) | **24.42x** | 2.2e-16 |
| `normal_pdf` | 100,000 | 414.20 us | 1.44 ms (scipy) | **3.48x** | 1.1e-16 |
| `normal_cdf` | 100,000 | 892.20 us | 2.15 ms (scipy) | **2.41x** | 2.2e-16 |
| `normal_logpdf` | 100,000 | 77.90 us | 1.59 ms (scipy) | **20.38x** | 8.9e-16 |
| `exponential_pdf` | 100,000 | 312.40 us | 1.37 ms (scipy) | **4.37x** | 0.0e+00 |
| `exponential_cdf` | 100,000 | 360.10 us | 1.65 ms (scipy) | **4.58x** | 1.7e-16 |
| `uniform_pdf` | 100,000 | 93.60 us | 1.55 ms (scipy) | **16.57x** | 0.0e+00 |
| `uniform_cdf` | 100,000 | 99.30 us | 1.63 ms (scipy) | **16.43x** | 0.0e+00 |
| `poisson_pmf` | 100,000 | 4.02 ms | 3.23 ms (scipy) | **0.80x** | 2.0e-19 |
| `poisson_cdf` | 100,000 | 1.31 ms | 6.17 ms (scipy) | **4.70x** | 2.2e-16 |
| `bernoulli_pmf` | 100,000 | 292.70 us | 3.52 ms (scipy) | **12.03x** | 2.2e-16 |
| `normal_pdf` | 1,000,000 | 4.82 ms | 20.68 ms (scipy) | **4.29x** | 1.1e-16 |
| `normal_cdf` | 1,000,000 | 9.66 ms | 23.05 ms (scipy) | **2.39x** | 2.2e-16 |
| `normal_logpdf` | 1,000,000 | 1.34 ms | 22.24 ms (scipy) | **16.60x** | 8.9e-16 |
| `exponential_pdf` | 1,000,000 | 3.81 ms | 17.79 ms (scipy) | **4.67x** | 0.0e+00 |
| `exponential_cdf` | 1,000,000 | 4.31 ms | 20.35 ms (scipy) | **4.72x** | 1.7e-16 |
| `uniform_pdf` | 1,000,000 | 1.46 ms | 19.01 ms (scipy) | **13.05x** | 0.0e+00 |
| `uniform_cdf` | 1,000,000 | 1.48 ms | 20.21 ms (scipy) | **13.67x** | 0.0e+00 |
| `poisson_pmf` | 1,000,000 | 42.48 ms | 38.83 ms (scipy) | **0.91x** | 2.0e-19 |
| `poisson_cdf` | 1,000,000 | 14.32 ms | 64.10 ms (scipy) | **4.48x** | 2.2e-16 |
| `bernoulli_pmf` | 1,000,000 | 3.55 ms | 39.58 ms (scipy) | **11.16x** | 2.2e-16 |

### scalar (per-call cost, not throughput)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_pdf` | 20,000 | 7.40 ms | 466.93 ms (scipy) | **63.09x** | 1.1e-16 |
| `normal_cdf` | 20,000 | 7.67 ms | 443.36 ms (scipy) | **57.81x** | 2.2e-16 |
| `gamma_cdf` | 20,000 | 12.23 ms | 444.12 ms (scipy) | **36.33x** | 1.2e-13 |
| `chi_square_cdf` | 20,000 | 11.84 ms | 447.19 ms (scipy) | **37.76x** | 1.2e-13 |
| `beta_cdf` | 20,000 | 9.90 ms | 484.99 ms (scipy) | **48.98x** | 8.9e-16 |

### cuda (GPU path vs the CPU path; below 1x means the GPU is slower)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_pdf` | 1,000 | 42.20 us | 5.30 us (fastdist-cpu) | **0.13x** | 5.6e-17 |
| `normal_cdf` | 1,000 | 39.80 us | 7.20 us (fastdist-cpu) | **0.18x** | 2.2e-16 |
| `normal_logpdf` | 1,000 | 34.30 us | 2.00 us (fastdist-cpu) | **0.06x** | 0.0e+00 |
| `exponential_pdf` | 1,000 | 33.30 us | 4.30 us (fastdist-cpu) | **0.13x** | 1.1e-16 |
| `uniform_pdf` | 1,000 | 39.30 us | 3.60 us (fastdist-cpu) | **0.09x** | 0.0e+00 |
| `normal_pdf` | 100,000 | 1.11 ms | 413.90 us (fastdist-cpu) | **0.37x** | 5.6e-17 |
| `normal_cdf` | 100,000 | 1.13 ms | 888.00 us (fastdist-cpu) | **0.78x** | 2.2e-16 |
| `normal_logpdf` | 100,000 | 1.12 ms | 77.80 us (fastdist-cpu) | **0.07x** | 0.0e+00 |
| `exponential_pdf` | 100,000 | 1.11 ms | 311.80 us (fastdist-cpu) | **0.28x** | 2.2e-16 |
| `uniform_pdf` | 100,000 | 1.10 ms | 94.10 us (fastdist-cpu) | **0.09x** | 0.0e+00 |
| `normal_pdf` | 1,000,000 | 11.08 ms | 5.00 ms (fastdist-cpu) | **0.45x** | 5.6e-17 |
| `normal_cdf` | 1,000,000 | 11.23 ms | 9.98 ms (fastdist-cpu) | **0.89x** | 2.2e-16 |
| `normal_logpdf` | 1,000,000 | 11.06 ms | 1.39 ms (fastdist-cpu) | **0.13x** | 0.0e+00 |
| `exponential_pdf` | 1,000,000 | 10.85 ms | 3.74 ms (fastdist-cpu) | **0.35x** | 2.2e-16 |
| `uniform_pdf` | 1,000,000 | 10.84 ms | 1.55 ms (fastdist-cpu) | **0.14x** | 0.0e+00 |

### sample (vs numpy)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_sample` | 100,000 | 27.79 ms | 871.10 us (numpy) | **0.03x** | - |
| `uniform_sample` | 100,000 | 25.48 ms | 226.80 us (numpy) | **0.01x** | - |
| `normal_sample` | 1,000,000 | 294.54 ms | 10.22 ms (numpy) | **0.03x** | - |
| `uniform_sample` | 1,000,000 | 273.14 ms | 3.17 ms (numpy) | **0.01x** | - |

---

## Unreleased — correction: the CUDA figures above were measured on a cold GPU

The previous entry concluded that the GPU path loses to the CPU path at every
size. That conclusion was an artifact of how it was measured, not a property of
the backend, and it is withdrawn.

### What went wrong

An idle NVIDIA card drops its clocks and downtrains its PCIe link. On this
machine that is 210 MHz and Gen1, against 2865 MHz and Gen4 under load. The
benchmark runs its CPU groups first, which takes minutes, so the GPU was cold
by the time the `cuda` group ran, and each case is far too short to train it
back up. The single warmup call the harness makes is nowhere near enough.

Same call, same machine, `normal_pdf` at n = 1,000,000, minimum of 7 rounds:

| GPU state before timing | clocks / link | time |
|---|---|---:|
| idle during the CPU groups | 210 MHz, Gen1 | 10.16 ms |
| after 3 s of GPU work | 2865 MHz, Gen4 | 2.47 ms |

A standalone CUDA probe confirms the hardware was never the problem: 8 MB
copies sustain 22.7 GB/s pageable and 25.8 GB/s pinned, the kernel alone takes
0.19 ms, and the full round trip the executor performs takes 1.44 ms.

`benchmarks/run.py` now warms the GPU before timing the `cuda` group and
records the device state it reached, which is printed in the run and shown
below.

### The corrected picture

The GPU wins where there is real arithmetic per element, and loses where the
CPU path is already very fast and the transfer dominates. At n = 1,000,000 it
wins for `exponential_pdf`, `normal_cdf`, `normal_pdf` and loses for `normal_logpdf`, `uniform_pdf`.

| case | n | cold (withdrawn) | warm (this run) |
|---|---:|---:|---:|
| `normal_cdf` | 1,000,000 | 0.89x | 4.59x |
| `normal_pdf` | 1,000,000 | 0.45x | 2.58x |
| `exponential_pdf` | 1,000,000 | 0.35x | 2.05x |
| `normal_logpdf` | 1,000,000 | 0.13x | 0.68x |
| `uniform_pdf` | 1,000,000 | 0.14x | 0.77x |

At n = 1,000 the GPU loses every case: launch and transfer overhead swamps a
few microseconds of work.

This bears directly on the auto-dispatch thresholds, which default to 100,000
for every function. That is about right for `normal_pdf`, `normal_cdf` and
`exponential_pdf`, which are 2.2x to 4.2x faster on the GPU there, and wrong for
`normal_logpdf` and `uniform_pdf`, which are still slower on the GPU at
n = 1,000,000. The thresholds want to be per function, which is what
`config.auto_tune` is for -- though its search starts at 500,000 and cannot
return "never", so it cannot express the `uniform_pdf` case today.

One more thing this run shows: the first call to each kernel costs about 12.8 ms
against 2.7 ms afterwards, because `CMAKE_CUDA_ARCHITECTURES` is never set. The
binary carries PTX for compute_52 and the driver JIT-compiles it for the actual
card on first use.

<!-- generated from 0.1.0_20260912T012110+0000_b7752f22e7.json by benchmarks/table.py -->
- **Version** 0.1.0 (`b7752f22e7` on `chore/release-prep`, working tree dirty)
- **Measured** 2026-09-12T01:21:10+00:00
- **CPU** AMD Ryzen 7 7700 8-Core Processor
- **Platform** Windows-11-10.0.26200-SP0
- **Toolchain** Python 3.14.2, numpy 2.5.2, scipy 1.18.1
- **CUDA** available

### cuda (GPU path vs the CPU path; below 1x means the GPU is slower)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_pdf` | 1,000 | 32.40 us | 6.70 us (fastdist-cpu) | **0.21x** | 5.6e-17 |
| `normal_cdf` | 1,000 | 25.30 us | 7.00 us (fastdist-cpu) | **0.28x** | 2.2e-16 |
| `normal_logpdf` | 1,000 | 27.30 us | 4.40 us (fastdist-cpu) | **0.16x** | 0.0e+00 |
| `exponential_pdf` | 1,000 | 27.80 us | 5.30 us (fastdist-cpu) | **0.19x** | 1.1e-16 |
| `uniform_pdf` | 1,000 | 29.90 us | 3.50 us (fastdist-cpu) | **0.12x** | 0.0e+00 |
| `normal_pdf` | 100,000 | 193.40 us | 419.10 us (fastdist-cpu) | **2.17x** | 5.6e-17 |
| `normal_cdf` | 100,000 | 211.90 us | 896.70 us (fastdist-cpu) | **4.23x** | 2.2e-16 |
| `normal_logpdf` | 100,000 | 200.20 us | 78.30 us (fastdist-cpu) | **0.39x** | 0.0e+00 |
| `exponential_pdf` | 100,000 | 188.20 us | 311.90 us (fastdist-cpu) | **1.66x** | 2.2e-16 |
| `uniform_pdf` | 100,000 | 184.70 us | 95.50 us (fastdist-cpu) | **0.52x** | 0.0e+00 |
| `normal_pdf` | 1,000,000 | 1.96 ms | 5.05 ms (fastdist-cpu) | **2.58x** | 5.6e-17 |
| `normal_cdf` | 1,000,000 | 2.17 ms | 9.96 ms (fastdist-cpu) | **4.59x** | 2.2e-16 |
| `normal_logpdf` | 1,000,000 | 2.02 ms | 1.37 ms (fastdist-cpu) | **0.68x** | 0.0e+00 |
| `exponential_pdf` | 1,000,000 | 1.92 ms | 3.94 ms (fastdist-cpu) | **2.05x** | 2.2e-16 |
| `uniform_pdf` | 1,000,000 | 1.86 ms | 1.43 ms (fastdist-cpu) | **0.77x** | 0.0e+00 |

---

## Unreleased -- a second baseline, so the SciPy comparison is not oversold

Every entry above compares against `scipy.stats`. That is what a user would
replace, but it is a soft target: `scipy.stats.norm.pdf` carries argument
validation, broadcasting and masking that a direct expression does not, and
most of the margin recorded above is that machinery rather than faster
arithmetic.

The suite now also measures a `primitives` group: the same quantity written
directly with numpy or `scipy.special`, which is what a competent user would
write if they cared about speed. It is the harder baseline and the honest
ceiling.

| case | vs `scipy.stats` (100k) | vs primitives (100k) | vs `scipy.stats` (1M) | vs primitives (1M) |
|---|---:|---:|---:|---:|
| `bernoulli_pmf` | 12.02x | 0.16x | 10.88x | 0.31x |
| `exponential_cdf` | 4.67x | 1.00x | 4.50x | 1.46x |
| `exponential_pdf` | 4.34x | 0.82x | 4.50x | 1.31x |
| `normal_cdf` | 2.42x | 0.94x | 2.29x | 0.95x |
| `normal_logpdf` | 18.28x | 0.56x | 15.72x | 2.25x |
| `normal_pdf` | 3.63x | 0.68x | 4.15x | 1.25x |
| `poisson_cdf` | 4.63x | 3.72x | 4.44x | 3.58x |
| `poisson_pmf` | 0.78x | 0.58x | 0.90x | 0.65x |
| `uniform_cdf` | 17.07x | 0.51x | 12.86x | 1.98x |
| `uniform_pdf` | 16.60x | 0.64x | 12.84x | 0.99x |

### Reading

Against `scipy.stats` the library is 2.3x to 16x faster. Against the
primitives it is roughly at parity: behind on most cases at n = 100,000, ahead
on six of ten at n = 1,000,000.

The size dependence is the interesting part. A numpy expression materialises a
temporary array per operation; fastdist makes one pass and writes one output.
At 100,000 elements those temporaries still sit in cache and numpy wins. At
1,000,000 they do not, and the single pass pulls ahead.

Two cases stand out at each end. `poisson_cdf` is 3.58x faster than
`scipy.special.pdtr`, because summing the terms by recurrence beats a general
implementation. `bernoulli_pmf` is 0.16x at 100,000: the primitive is a single
`np.where`, and nothing in a C++ loop can beat one vectorised select.

What this means for how the numbers get quoted:

- "3x to 16x faster than SciPy" is true only of `scipy.stats`, and needs the
  reason attached, or it is a misleading claim.
- "Faster than hand-written numpy" is true only at a million elements and only
  for some functions. It is not a general claim.
- The scalar group remains the library's strongest honest result: calling into
  fastdist from a Python loop costs far less than calling `scipy.stats`.

<!-- generated from 0.1.0_20260912T013406+0000_da5f274059.json by benchmarks/table.py -->
- **Version** 0.1.0 (`da5f274059` on `chore/release-prep`, working tree dirty)
- **Measured** 2026-09-12T01:34:06+00:00
- **CPU** AMD Ryzen 7 7700 8-Core Processor
- **Platform** Windows-11-10.0.26200-SP0
- **Toolchain** Python 3.14.2, numpy 2.5.2, scipy 1.18.1
- **CUDA** available

### primitives (vs the numpy / scipy.special expression)

| case | n | fastdist | baseline | speedup | max abs diff |
|---|---:|---:|---:|---:|---:|
| `normal_pdf` | 1,000 | 5.24 us | 4.59 us (numpy/scipy.special) | **0.88x** | 1.1e-16 |
| `normal_cdf` | 1,000 | 6.80 us | 4.32 us (numpy/scipy.special) | **0.63x** | 2.2e-16 |
| `normal_logpdf` | 1,000 | 1.88 us | 1.83 us (numpy/scipy.special) | **0.97x** | 8.9e-16 |
| `exponential_pdf` | 1,000 | 4.16 us | 4.03 us (numpy/scipy.special) | **0.97x** | 0.0e+00 |
| `exponential_cdf` | 1,000 | 4.49 us | 4.67 us (numpy/scipy.special) | **1.04x** | 0.0e+00 |
| `uniform_pdf` | 1,000 | 2.07 us | 2.91 us (numpy/scipy.special) | **1.41x** | 0.0e+00 |
| `uniform_cdf` | 1,000 | 2.12 us | 3.35 us (numpy/scipy.special) | **1.58x** | 0.0e+00 |
| `poisson_pmf` | 1,000 | 40.82 us | 15.11 us (numpy/scipy.special) | **0.37x** | 2.0e-19 |
| `poisson_cdf` | 1,000 | 10.05 us | 36.52 us (numpy/scipy.special) | **3.63x** | 2.2e-16 |
| `bernoulli_pmf` | 1,000 | 1.96 us | 2.04 us (numpy/scipy.special) | **1.04x** | 0.0e+00 |
| `normal_pdf` | 100,000 | 415.90 us | 281.70 us (numpy/scipy.special) | **0.68x** | 1.1e-16 |
| `normal_cdf` | 100,000 | 894.80 us | 836.80 us (numpy/scipy.special) | **0.94x** | 2.2e-16 |
| `normal_logpdf` | 100,000 | 77.60 us | 43.60 us (numpy/scipy.special) | **0.56x** | 8.9e-16 |
| `exponential_pdf` | 100,000 | 311.50 us | 255.00 us (numpy/scipy.special) | **0.82x** | 0.0e+00 |
| `exponential_cdf` | 100,000 | 366.40 us | 367.00 us (numpy/scipy.special) | **1.00x** | 0.0e+00 |
| `uniform_pdf` | 100,000 | 94.70 us | 60.70 us (numpy/scipy.special) | **0.64x** | 0.0e+00 |
| `uniform_cdf` | 100,000 | 100.50 us | 50.90 us (numpy/scipy.special) | **0.51x** | 0.0e+00 |
| `poisson_pmf` | 100,000 | 4.02 ms | 2.34 ms (numpy/scipy.special) | **0.58x** | 2.0e-19 |
| `poisson_cdf` | 100,000 | 1.30 ms | 4.84 ms (numpy/scipy.special) | **3.72x** | 2.2e-16 |
| `bernoulli_pmf` | 100,000 | 293.00 us | 45.90 us (numpy/scipy.special) | **0.16x** | 0.0e+00 |
| `normal_pdf` | 1,000,000 | 4.77 ms | 5.96 ms (numpy/scipy.special) | **1.25x** | 1.1e-16 |
| `normal_cdf` | 1,000,000 | 9.61 ms | 9.15 ms (numpy/scipy.special) | **0.95x** | 2.2e-16 |
| `normal_logpdf` | 1,000,000 | 1.28 ms | 2.89 ms (numpy/scipy.special) | **2.25x** | 8.9e-16 |
| `exponential_pdf` | 1,000,000 | 3.73 ms | 4.88 ms (numpy/scipy.special) | **1.31x** | 0.0e+00 |
| `exponential_cdf` | 1,000,000 | 4.26 ms | 6.22 ms (numpy/scipy.special) | **1.46x** | 0.0e+00 |
| `uniform_pdf` | 1,000,000 | 1.41 ms | 1.40 ms (numpy/scipy.special) | **0.99x** | 0.0e+00 |
| `uniform_cdf` | 1,000,000 | 1.47 ms | 2.91 ms (numpy/scipy.special) | **1.98x** | 0.0e+00 |
| `poisson_pmf` | 1,000,000 | 41.82 ms | 27.22 ms (numpy/scipy.special) | **0.65x** | 2.0e-19 |
| `poisson_cdf` | 1,000,000 | 14.09 ms | 50.43 ms (numpy/scipy.special) | **3.58x** | 2.2e-16 |
| `bernoulli_pmf` | 1,000,000 | 3.51 ms | 1.07 ms (numpy/scipy.special) | **0.31x** | 0.0e+00 |

---

## Changes to record here

Add an entry when a release ships, or when a change is made specifically to
move performance. Each entry should carry the generated table, the commit, and
a short reading of what moved and why. `compare.py` output makes a good basis
for the reading.
