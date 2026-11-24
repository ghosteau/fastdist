// Regression tests for defects found auditing the numerical core.
//
// Each block states what the backend did before the fix. The expected values
// are derived from closed forms or from numerically stable rearrangements of
// the same algebra, never captured from a previous run of this library.
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

#include <fastdist/math/beta.h>
#include <fastdist/math/binomial.h>
#include <fastdist/math/discrete_uniform.h>
#include <fastdist/math/gamma.h>
#include <fastdist/math/geometric.h>
#include <fastdist/math/negative_binomial.h>
#include <fastdist/math/normal.h>
#include <fastdist/math/poisson.h>
#include <fastdist/math/uniform.h>
#include <fastdist/math/utils.h>

namespace {

    constexpr int INT_MIN_V = std::numeric_limits<int>::min();
    constexpr int INT_MAX_V = std::numeric_limits<int>::max();

    // Relative difference, treating an exact zero expectation as absolute.
    bool close(const double got, const double want, const double rel = 1e-14) {
        if (want == 0.0) return std::fabs(got) < 1e-300;
        return std::fabs(got - want) / std::fabs(want) < rel;
    }

    // Milliseconds taken by fn().
    template<typename Fn>
    double millis(Fn&& fn) {
        const auto start = std::chrono::steady_clock::now();
        fn();
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

} // namespace

void test_regressions() {
    std::cout << "Running numerical regression tests...\n";

    // -------------------------------------------------------------------
    // Catastrophic cancellation
    //
    // 1 - (1-p)^k and exp(bt) - exp(at) each subtract two nearly equal
    // quantities. The stable forms below are algebraically identical.
    // -------------------------------------------------------------------
    {
        // geometric CDF: lost three of four significant digits at p = 1e-15.
        for (const double p: {1e-10, 1e-12, 1e-15, 1e-8}) {
            for (const int k: {1, 2, 3, 5}) {
                const double got = fastdist::math::geometric_cdf_scalar(k, p);
                const double want = -std::expm1(static_cast<double>(k) * std::log1p(-p));
                assert(close(got, want, 1e-15));
            }
        }

        // continuous uniform MGF: left four correct digits at t = 1e-13.
        const double pairs[][3] = {{1e-9, 0.0, 1.0}, {1e-11, 0.0, 1.0}, {1e-13, 2.0, 5.0}};
        for (const auto& c: pairs) {
            const double got = fastdist::math::uniform_mgf_scalar(c[0], c[1], c[2]);
            const double want = (std::expm1(c[2] * c[0]) - std::expm1(c[1] * c[0])) / (c[0] * (c[2] - c[1]));
            assert(close(got, want, 1e-15));
        }

        // discrete uniform MGF: canceled in both the numerator and denominator.
        {
            constexpr double t = 1e-10;
            constexpr int a = 0, b = 100;
            constexpr double n = b - a + 1;
            const double got = fastdist::math::discrete_uniform_mgf_scalar(t, a, b);
            const double want = std::exp(t * a) * std::expm1(t * n) / (n * std::expm1(t));
            assert(close(got, want, 1e-15));
        }
    }

    // -------------------------------------------------------------------
    // Integer overflow
    //
    // b - a + 1 was formed in int, so a range wider than INT_MAX wrapped:
    // the PMF came back infinite and the variance negative.
    // -------------------------------------------------------------------
    {
        const double pmf = fastdist::math::discrete_uniform_pmf_scalar(0, INT_MIN_V, INT_MAX_V);
        assert(close(pmf, 1.0 / 4294967296.0, 1e-15));

        const double var = fastdist::math::discrete_uniform_variance(INT_MIN_V, INT_MAX_V);
        assert(var > 0.0 && "a variance must never be negative");
        assert(close(var, (18446744073709551616.0 - 1.0) / 12.0, 1e-15));

        const double cdf = fastdist::math::discrete_uniform_cdf_scalar(-1, INT_MIN_V, INT_MAX_V);
        assert(close(cdf, 0.5, 1e-15));

        // poisson CDF: static_cast<int> of a double past INT_MAX is undefined.
        // It wrapped to INT_MIN, skipped the loop and returned exp(-lambda).
        for (const double x: {2147483648.0, 3e9, 1e18}) {
            assert(fastdist::math::poisson_cdf_scalar(x, 5.0) == 1.0);
        }
    }

    // -------------------------------------------------------------------
    // Unbounded summation
    //
    // These ran to their argument even once every remaining term was too
    // small to move the total.
    // -------------------------------------------------------------------
    {
        double result = 0.0;
        const double ms = millis([&] { result = fastdist::math::poisson_cdf_scalar(2e9, 5.0); });
        assert(result == 1.0);
        assert(ms < 1000.0 && "the poisson tail sum is not terminating early");

        const double ms_nb = millis([&] { result = fastdist::math::negative_binomial_cdf_scalar(200000000, 5, 0.5); });
        assert(result == 1.0);
        assert(ms_nb < 1000.0 && "the negative binomial tail sum is not terminating early");

        // factorial kept multiplying long after the product was infinite.
        const double ms_fact = millis([&] { result = fastdist::math::factorial(10000000u); });
        assert(std::isinf(result));
        assert(ms_fact < 500.0 && "factorial is still multiplying past infinity");
        assert(std::isfinite(fastdist::math::factorial(170u)));
        assert(std::isinf(fastdist::math::factorial(171u)));
    }

    // -------------------------------------------------------------------
    // CGF computed as log(MGF)
    //
    // The MGF raises a base to a large power and overflows, so log of it
    // reported NaN for a cumulant that is an ordinary number.
    // -------------------------------------------------------------------
    {
        const double cases[][3] = {{0.99, 1000.0, 1.0}, {0.5, 2000.0, 1.0}, {0.9, 5000.0, 1.0}};
        for (const auto& c: cases) {
            const double got = fastdist::math::gamma_cgf_scalar(c[0], c[1], c[2]);
            assert(std::isfinite(got) && "the gamma MGF overflowed and the CGF reported NaN");
            assert(close(got, -c[1] * std::log1p(-c[2] * c[0])));
        }

        const double nb_t[] = {0.5, 0.6, 0.1};
        const int nb_r[] = {1000, 2000, 5000};
        for (int i = 0; i < 3; ++i) {
            const double got = fastdist::math::negative_binomial_cgf_scalar(nb_t[i], nb_r[i], 0.5);
            const double want = nb_r[i] * (std::log(0.5) - std::log1p(-0.5 * std::exp(nb_t[i])));
            assert(std::isfinite(got) && "the negative binomial MGF overflowed and the CGF reported NaN");
            assert(close(got, want));
        }

        // Where the MGF is representable, both routes must still agree.
        for (const double alpha: {2.0, 3.0, 10.0}) {
            const double mgf = fastdist::math::gamma_mgf_scalar(0.1, alpha, 1.0);
            assert(close(fastdist::math::gamma_cgf_scalar(0.1, alpha, 1.0), std::log(mgf)));
        }
    }

    // -------------------------------------------------------------------
    // Missing validation
    //
    // Every entry point returns NaN outside its domain. These returned a
    // plausible-looking number: beta_variance(-1, -2) gave -0.111.
    // -------------------------------------------------------------------
    {
        const double bad[][2] = {{-1.0, -2.0}, {-5.0, 3.0}, {0.0, 1.0}, {1.0, 0.0}};
        for (const auto& c: bad) {
            assert(std::isnan(fastdist::math::beta_mean(c[0], c[1])));
            assert(std::isnan(fastdist::math::beta_variance(c[0], c[1])));
            assert(std::isnan(fastdist::math::beta_stddev(c[0], c[1])));
        }
        assert(close(fastdist::math::beta_mean(2.0, 3.0), 0.4, 1e-15));
        assert(close(fastdist::math::beta_variance(2.0, 3.0), 6.0 / (25.0 * 6.0), 1e-15));

        // z_score divided unchecked: sigma == 0 gave infinity, and a negative
        // sigma silently flipped the sign of the score.
        assert(std::isnan(fastdist::math::z_score(1.0, 0.0, 0.0)));
        assert(std::isnan(fastdist::math::z_score(1.0, 0.0, -2.0)));
        assert(std::isnan(fastdist::math::z_score(std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0)));
        assert(close(fastdist::math::z_score(3.0, 1.0, 2.0), 1.0, 1e-15));
    }

    // -------------------------------------------------------------------
    // Inconsistent parameter ranges
    //
    // mean and variance accepted p = 1 and returned 0, while the PMF and CDF
    // returned NaN for the same p.
    // -------------------------------------------------------------------
    {
        constexpr int r = 5;
        assert(fastdist::math::negative_binomial_pmf_scalar(0, r, 1.0) == 1.0);
        assert(fastdist::math::negative_binomial_pmf_scalar(1, r, 1.0) == 0.0);
        assert(fastdist::math::negative_binomial_cdf_scalar(0, r, 1.0) == 1.0);
        assert(fastdist::math::negative_binomial_mean(r, 1.0) == 0.0);
        assert(fastdist::math::negative_binomial_variance(r, 1.0) == 0.0);
        assert(close(fastdist::math::negative_binomial_mgf_scalar(0.5, r, 1.0), 1.0, 1e-15));
        assert(std::fabs(fastdist::math::negative_binomial_cgf_scalar(0.5, r, 1.0)) < 1e-15);
        assert(fastdist::math::negative_binomial_sample(r, 1.0) == 0);

        // p above 1 is still rejected everywhere.
        assert(std::isnan(fastdist::math::negative_binomial_pmf_scalar(0, r, 1.5)));
        assert(std::isnan(fastdist::math::negative_binomial_cdf_scalar(0, r, 1.5)));
        assert(std::isnan(fastdist::math::negative_binomial_mean(r, 1.5)));

        // std::geometric_distribution is undefined at p == 1, which the
        // validation admits; the first trial always succeeds there.
        for (int i = 0; i < 50; ++i) {
            assert(fastdist::math::geometric_sample(1.0) == 1);
        }
    }

    std::cout << "  numerical regression tests passed\n";
}
