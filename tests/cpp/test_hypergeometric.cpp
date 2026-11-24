// Unit tests for the Hypergeometric distribution
//
// Successes in a draw made WITHOUT replacement:
//
//     P(X = k) = C(K, k) C(N - K, n - k) / C(N, n)
//
// with N = population, K = successes in the population, n = draws. The
// references below are built from exact integer binomial coefficients where
// the numbers are small enough to hold in a double, and from closed forms
// otherwise.
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <fastdist/math/hypergeometric.h>
#include <fastdist/math/rng.h>

namespace {

    bool close(const double got, const double want, const double rel = 1e-11) {
        if (want == 0.0) return std::fabs(got) < 1e-300;
        return std::fabs(got - want) / std::fabs(want) < rel;
    }

    // Exact C(n, r) by the multiplicative recurrence. Every partial product is
    // itself a binomial coefficient, so each division is exact while the result
    // stays inside 2^53.
    double choose_exact(const int n, int r) {
        if (r < 0 || r > n) return 0.0;
        if (r > n - r) r = n - r;
        double result = 1.0;
        for (int i = 1; i <= r; ++i) {
            result *= static_cast<double>(n - r + i);
            result /= static_cast<double>(i);
        }
        return result;
    }

    double reference_pmf(const int k, const int N, const int K, const int n) {
        const int lo = std::max(0, n - (N - K));
        const int hi = std::min(n, K);
        if (k < lo || k > hi) return 0.0;
        return choose_exact(K, k) * choose_exact(N - K, n - k) / choose_exact(N, n);
    }

} // namespace

void test_hypergeometric() {
    std::cout << "Running hypergeometric distribution tests...\n";

    using namespace fastdist::math;

    // A spread of shapes, including ones whose support does not start at zero.
    const int grid[][3] = {{20, 7, 5}, {50, 10, 10}, {10, 9, 9}, {100, 50, 30}, {6, 3, 3},
                           {2, 1, 1},  {15, 15, 7},  {15, 0, 7}, {12, 4, 0},    {12, 4, 12}};

    // -------------------------------------------------------------------
    // PMF against exact binomial coefficients, and the CDF against the
    // running sum of that same reference
    // -------------------------------------------------------------------
    for (const auto& c: grid) {
        const int N = c[0], K = c[1], n = c[2];
        const int lo = std::max(0, n - (N - K));
        const int hi = std::min(n, K);

        double total = 0.0;
        double running = 0.0;
        for (int k = lo; k <= hi; ++k) {
            const double want = reference_pmf(k, N, K, n);
            assert(close(hypergeometric_pmf_scalar(k, N, K, n), want));
            total += hypergeometric_pmf_scalar(k, N, K, n);

            running += want;
            assert(close(hypergeometric_cdf_scalar(k, N, K, n), running, 1e-10));
        }

        // The mass over the support is all of it.
        assert(close(total, 1.0));

        // Outside the support: no mass, and the CDF is pinned to its ends.
        assert(hypergeometric_pmf_scalar(lo - 1, N, K, n) == 0.0);
        assert(hypergeometric_pmf_scalar(hi + 1, N, K, n) == 0.0);
        assert(hypergeometric_cdf_scalar(lo - 1, N, K, n) == 0.0);
        assert(hypergeometric_cdf_scalar(hi, N, K, n) == 1.0);
        assert(hypergeometric_cdf_scalar(hi + 100, N, K, n) == 1.0);

        // The CDF never decreases and never leaves [0, 1].
        double previous = -1.0;
        for (int k = lo - 1; k <= hi + 1; ++k) {
            const double current = hypergeometric_cdf_scalar(k, N, K, n);
            assert(current >= previous - 1e-15);
            assert(current >= 0.0 && current <= 1.0);
            previous = current;
        }
    }

    // -------------------------------------------------------------------
    // log PMF
    // -------------------------------------------------------------------
    {
        for (int k = 0; k <= 5; ++k) {
            const double pmf = hypergeometric_pmf_scalar(k, 20, 7, 5);
            assert(close(hypergeometric_logpmf_scalar(k, 20, 7, 5), std::log(pmf)));
        }
        assert(hypergeometric_logpmf_scalar(-1, 20, 7, 5) == -std::numeric_limits<double>::infinity());
        assert(hypergeometric_logpmf_scalar(6, 20, 7, 5) == -std::numeric_limits<double>::infinity());

        // A population whose factorials overflow a double many times over.
        // Evaluating in log space is what keeps this finite.
        const double big = hypergeometric_logpmf_scalar(250, 1000000, 500000, 500);
        assert(std::isfinite(big));
        const double big_pmf = hypergeometric_pmf_scalar(250, 1000000, 500000, 500);
        assert(big_pmf > 0.0 && big_pmf < 1.0);
    }

    // -------------------------------------------------------------------
    // Moments, against the sums they are defined by
    // -------------------------------------------------------------------
    for (const auto& c: grid) {
        const int N = c[0], K = c[1], n = c[2];
        const int lo = std::max(0, n - (N - K));
        const int hi = std::min(n, K);

        double mean = 0.0;
        for (int k = lo; k <= hi; ++k) {
            mean += k * reference_pmf(k, N, K, n);
        }
        double variance = 0.0;
        for (int k = lo; k <= hi; ++k) {
            const double d = k - mean;
            variance += d * d * reference_pmf(k, N, K, n);
        }

        assert(close(hypergeometric_mean(N, K, n), mean, 1e-10) ||
               (mean == 0.0 && hypergeometric_mean(N, K, n) == 0.0));
        assert(close(hypergeometric_variance(N, K, n), variance, 1e-9) ||
               (variance < 1e-15 && hypergeometric_variance(N, K, n) < 1e-15));
        assert(close(hypergeometric_stddev(N, K, n), std::sqrt(hypergeometric_variance(N, K, n)), 1e-12) ||
               hypergeometric_variance(N, K, n) == 0.0);
    }

    // Drawing the whole population leaves nothing to vary, and a population of
    // one is deterministic -- the 0/0 in the finite population correction.
    assert(hypergeometric_variance(12, 4, 12) == 0.0);
    assert(hypergeometric_variance(1, 1, 1) == 0.0);
    assert(hypergeometric_variance(1, 0, 1) == 0.0);

    // Sampling without replacement can only narrow the spread.
    {
        constexpr int N = 100, K = 40, n = 20;
        constexpr double p = static_cast<double>(K) / N;
        assert(hypergeometric_variance(N, K, n) < n * p * (1.0 - p));
    }

    // -------------------------------------------------------------------
    // MGF and CGF, accumulated over the finite support
    // -------------------------------------------------------------------
    {
        for (const double t: {0.0, 0.3, -0.7, 1.5}) {
            for (const auto& c: grid) {
                const int N = c[0], K = c[1], n = c[2];
                const int lo = std::max(0, n - (N - K));
                const int hi = std::min(n, K);

                double want = 0.0;
                for (int k = lo; k <= hi; ++k) {
                    want += std::exp(t * k) * reference_pmf(k, N, K, n);
                }
                assert(close(hypergeometric_mgf_scalar(t, N, K, n), want, 1e-10));
            }
        }

        // M(0) = 1 for every shape.
        for (const auto& c: grid) {
            assert(close(hypergeometric_mgf_scalar(0.0, c[0], c[1], c[2]), 1.0));
        }

        // The CGF is the log of the MGF where both are representable, and
        // stays finite where the MGF overflows.
        assert(close(hypergeometric_cgf_scalar(0.3, 50, 20, 15), std::log(hypergeometric_mgf_scalar(0.3, 50, 20, 15))));
        const double cgf = hypergeometric_cgf_scalar(500.0, 1000, 400, 600);
        assert(std::isfinite(cgf));
        assert(std::isinf(hypergeometric_mgf_scalar(500.0, 1000, 400, 600)));

        // M'(0) = E[X], by a central difference.
        constexpr double h = 1e-5;
        const double derivative =
                (hypergeometric_mgf_scalar(h, 50, 20, 15) - hypergeometric_mgf_scalar(-h, 50, 20, 15)) / (2.0 * h);
        assert(close(derivative, hypergeometric_mean(50, 20, 15), 1e-7));
    }

    // -------------------------------------------------------------------
    // Identities
    // -------------------------------------------------------------------
    {
        // Which items are drawn and which are successes is the same experiment
        // seen two ways, so the PMF is symmetric in K and n.
        for (int k = 0; k <= 5; ++k) {
            assert(close(hypergeometric_pmf_scalar(k, 20, 7, 5), hypergeometric_pmf_scalar(k, 20, 5, 7)));
        }

        // One draw succeeds with probability K/N.
        assert(close(hypergeometric_pmf_scalar(1, 37, 11, 1), 11.0 / 37.0, 1e-14));
        assert(close(hypergeometric_pmf_scalar(0, 37, 11, 1), 26.0 / 37.0, 1e-14));

        // An empty draw, an all-success pool and a no-success pool are certain.
        assert(hypergeometric_pmf_scalar(0, 12, 4, 0) == 1.0);
        assert(close(hypergeometric_pmf_scalar(7, 15, 15, 7), 1.0));
        assert(close(hypergeometric_pmf_scalar(0, 15, 0, 7), 1.0));
    }

    // -------------------------------------------------------------------
    // Validation: parameters that cannot describe a draw
    // -------------------------------------------------------------------
    {
        const int bad[][3] = {{-1, 0, 0}, {10, -1, 5}, {10, 5, -1}, {10, 11, 5}, {10, 5, 11}};
        for (const auto& c: bad) {
            assert(std::isnan(hypergeometric_pmf_scalar(0, c[0], c[1], c[2])));
            assert(std::isnan(hypergeometric_cdf_scalar(0, c[0], c[1], c[2])));
            assert(std::isnan(hypergeometric_logpmf_scalar(0, c[0], c[1], c[2])));
            assert(std::isnan(hypergeometric_mean(c[0], c[1], c[2])));
            assert(std::isnan(hypergeometric_variance(c[0], c[1], c[2])));
            assert(std::isnan(hypergeometric_mgf_scalar(0.5, c[0], c[1], c[2])));
            assert(hypergeometric_sample(c[0], c[1], c[2]) == -1);
        }

        // With no population there is no distribution to take a mean of.
        assert(std::isnan(hypergeometric_mean(0, 0, 0)));
        assert(std::isnan(hypergeometric_variance(0, 0, 0)));

        // A non-finite t is rejected like everywhere else in the library.
        assert(std::isnan(hypergeometric_mgf_scalar(std::numeric_limits<double>::quiet_NaN(), 20, 7, 5)));
        assert(std::isnan(hypergeometric_cgf_scalar(std::numeric_limits<double>::infinity(), 20, 7, 5)));
    }

    // -------------------------------------------------------------------
    // Sampling. There is no std::hypergeometric_distribution, so the draw is
    // simulated directly and has to be held to the PMF it claims.
    // -------------------------------------------------------------------
    {
        seed_rng(20240101);
        for (const auto& c: grid) {
            const int N = c[0], K = c[1], n = c[2];
            const int lo = std::max(0, n - (N - K));
            const int hi = std::min(n, K);
            for (int i = 0; i < 200; ++i) {
                const int draw = hypergeometric_sample(N, K, n);
                assert(draw >= lo && draw <= hi);
            }
        }

        // Degenerate shapes are deterministic.
        for (int i = 0; i < 20; ++i) {
            assert(hypergeometric_sample(12, 4, 0) == 0);
            assert(hypergeometric_sample(15, 15, 7) == 7);
            assert(hypergeometric_sample(15, 0, 7) == 0);
            assert(hypergeometric_sample(12, 4, 12) == 4);
        }

        // The same seed replays the same draws.
        seed_rng(777);
        std::vector<int> first;
        for (int i = 0; i < 100; ++i) first.push_back(hypergeometric_sample(50, 20, 12));
        seed_rng(777);
        for (int i = 0; i < 100; ++i) assert(hypergeometric_sample(50, 20, 12) == first[i]);

        // Pearson's chi-square against the exact PMF. The seed is fixed, so
        // this is deterministic rather than a flaky statistical test; the
        // threshold sits above the 99.9th percentile of the null.
        {
            constexpr int N = 50, K = 20, n = 12;
            constexpr int trials = 200000;
            const int lo = std::max(0, n - (N - K));
            const int hi = std::min(n, K);

            std::vector<long long> counts(hi - lo + 1, 0);
            seed_rng(31337);
            for (int i = 0; i < trials; ++i) {
                ++counts[hypergeometric_sample(N, K, n) - lo];
            }

            double statistic = 0.0;
            int degrees = -1;
            for (int k = lo; k <= hi; ++k) {
                const double expected = reference_pmf(k, N, K, n) * trials;
                if (expected < 5.0) continue; // cells too thin to test
                const double diff = static_cast<double>(counts[k - lo]) - expected;
                statistic += diff * diff / expected;
                ++degrees;
            }
            assert(degrees >= 5);
            assert(statistic < 32.0 && "the sampler does not match the PMF");
        }

        // The sample mean lands within five standard errors of the analytic mean.
        {
            constexpr int N = 100, K = 30, n = 25;
            constexpr int trials = 100000;
            seed_rng(4242);
            long long total = 0;
            for (int i = 0; i < trials; ++i) total += hypergeometric_sample(N, K, n);

            const double observed = static_cast<double>(total) / trials;
            const double expected = hypergeometric_mean(N, K, n);
            const double standard_error = std::sqrt(hypergeometric_variance(N, K, n) / trials);
            assert(std::fabs(observed - expected) < 5.0 * standard_error);
        }
    }

    std::cout << "  hypergeometric distribution tests passed\n";
}
