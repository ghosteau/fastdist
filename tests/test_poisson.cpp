#include <cassert>
#include <cmath>
#include <fastdist/math/poisson.h>
#include <iostream>

void test_poisson() {
    std::cout << "Running Poisson distribution tests...\n";

    constexpr double tol = 1e-12;
    constexpr double lambda = 4.0;

    // -------------------------
    // PMF / logPMF / CDF tests
    // -------------------------
    {
        const double x = 3.0;
        const double pmf = fastdist::math::poisson_pmf_scalar(x, lambda);
        const double expected_pmf = std::exp(-lambda) * std::pow(lambda, x) / std::tgamma(x + 1.0);
        assert(std::abs(pmf - expected_pmf) < tol);

        // CDF
        const double cdf = fastdist::math::poisson_cdf_scalar(x, lambda);
        double sum = 0.0;
        for (int i = 0; i <= static_cast<int>(x); ++i) {
            sum += fastdist::math::poisson_pmf_scalar(i, lambda);
        }
        assert(std::abs(cdf - sum) < tol);
    }

    // -------------------------
    // Mean / Variance / Stddev
    // -------------------------
    {
        assert(std::abs(fastdist::math::poisson_mean(lambda) - lambda) < tol);
        assert(std::abs(fastdist::math::poisson_variance(lambda) - lambda) < tol);
        assert(std::abs(fastdist::math::poisson_stddev(lambda) - std::sqrt(lambda)) < tol);
    }

    // -------------------------
    // MGF / CGF tests
    // -------------------------
    {
        const double t = 0.25;
        assert(std::abs(fastdist::math::poisson_mgf_scalar(t, lambda) - std::exp(lambda * (std::exp(t) - 1.0))) < tol);
        assert(std::abs(fastdist::math::poisson_cgf_scalar(t, lambda) - lambda * (std::exp(t) - 1.0)) < tol);
    }

    // -------------------------
    // RNG sanity check
    // -------------------------
    {
        constexpr int N = 250000;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = 0; i < N; ++i) {
            const int x = fastdist::math::poisson_sample(lambda);
            sum += x;
            sumsq += x * x;
        }

        const double mean = sum / N;
        const double var = sumsq / N - mean * mean;

        assert(std::abs(mean - lambda) < 5e-2);
        assert(std::abs(var - lambda) < 5e-1);
    }

    std::cout << "Poisson distribution tests passed!\n";
}
