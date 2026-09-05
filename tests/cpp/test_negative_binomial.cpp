// Unit tests for Negative Binomial distribution
#include <cassert>
#include <cmath>
#include <iostream>

#include <fastdist/math/negative_binomial.h>

void test_negative_binomial() {
    std::cout << "Running Negative Binomial distribution tests...\n";

    constexpr double tol = 1e-12;
    constexpr int r = 4; // number of successes
    constexpr double p = 0.3; // success probability

    // -------------------------
    // PMF / CDF tests
    // -------------------------
    {
        constexpr int k = 3;

        const double pmf = fastdist::math::negative_binomial_pmf_scalar(k, r, p);

        const double expected =
                std::tgamma(k + r) / (std::tgamma(r) * std::tgamma(k + 1.0)) * std::pow(1.0 - p, k) * std::pow(p, r);

        assert(std::abs(pmf - expected) < tol);

        // Outside support
        assert(fastdist::math::negative_binomial_pmf_scalar(-1, r, p) == 0.0);

        // CDF sanity check
        double sum = 0.0;
        for (int i = 0; i <= k; ++i) {
            sum += fastdist::math::negative_binomial_pmf_scalar(i, r, p);
        }

        const double cdf = fastdist::math::negative_binomial_cdf_scalar(k, r, p);

        assert(std::abs(cdf - sum) < tol);
    }

    // -------------------------
    // Mean / Variance / Stddev
    // -------------------------
    {
        const double mean = fastdist::math::negative_binomial_mean(r, p);
        const double var = fastdist::math::negative_binomial_variance(r, p);
        const double std = fastdist::math::negative_binomial_stddev(r, p);

        const double expected_mean = r * (1.0 - p) / p;
        const double expected_var = r * (1.0 - p) / (p * p);

        assert(std::abs(mean - expected_mean) < tol);
        assert(std::abs(var - expected_var) < tol);
        assert(std::abs(std - std::sqrt(expected_var)) < tol);
    }

    // -------------------------
    // MGF / CGF tests
    // -------------------------
    {
        const double t = 0.25;

        const double mgf = fastdist::math::negative_binomial_mgf_scalar(t, r, p);

        const double expected_mgf = std::pow(p / (1.0 - (1.0 - p) * std::exp(t)), r);

        assert(std::abs(mgf - expected_mgf) < tol);

        const double cgf = fastdist::math::negative_binomial_cgf_scalar(t, r, p);

        assert(std::abs(cgf - std::log(expected_mgf)) < tol);

        // M'(0) = mean
        constexpr double dt = 1e-6;
        const double dM = (fastdist::math::negative_binomial_mgf_scalar(dt, r, p) -
                           fastdist::math::negative_binomial_mgf_scalar(-dt, r, p)) /
                          (2.0 * dt);

        assert(std::abs(dM - fastdist::math::negative_binomial_mean(r, p)) < 1e-6);
    }

    // -------------------------
    // RNG tests
    // -------------------------
    {
        constexpr int N = 250000;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = 0; i < N; ++i) {
            const int x = fastdist::math::negative_binomial_sample(r, p);

            // support check
            assert(x >= 0);

            sum += x;
            sumsq += static_cast<double>(x) * x;
        }

        const double mean = sum / N;
        const double var = sumsq / N - mean * mean;

        // SE(mean) = 0.011 and SE(var) = 0.117 at N, so these are ~9 and ~8.5
        // sigma; the previous values sat near 4.3 sigma.
        // See ghosteau/fastdist#2.
        assert(std::abs(mean - fastdist::math::negative_binomial_mean(r, p)) < 0.1);

        assert(std::abs(var - fastdist::math::negative_binomial_variance(r, p)) < 1.0);
    }

    std::cout << "Negative Binomial distribution tests passed!\n";
}
