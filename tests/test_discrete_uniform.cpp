// Unit tests for Discrete Uniform distribution
#include <cassert>
#include <cmath>
#include <iostream>

#include <fastdist/math/discrete_uniform.h>

void test_discrete_uniform() {
    std::cout << "Running discrete uniform tests...\n";

    constexpr double tol = 1e-12;
    constexpr int a = 1;
    constexpr int b = 6; // support = {1,2,3,4,5,6}
    constexpr int n = b - a + 1;

    // -------------------------
    // PMF / CDF tests
    // -------------------------
    {
        const double pmf3 = fastdist::math::discrete_uniform_pmf_scalar(3, a, b);
        assert(std::abs(pmf3 - 1.0 / n) < tol);

        const double pmf_out = fastdist::math::discrete_uniform_pmf_scalar(7, a, b);
        assert(pmf_out == 0.0);

        const double cdf4 = fastdist::math::discrete_uniform_cdf_scalar(4, a, b);
        assert(std::abs(cdf4 - 4.0 / n) < tol);

        const double cdf_low = fastdist::math::discrete_uniform_cdf_scalar(0, a, b);
        assert(cdf_low == 0.0);

        const double cdf_high = fastdist::math::discrete_uniform_cdf_scalar(10, a, b);
        assert(cdf_high == 1.0);
    }

    // -------------------------
    // Mean / Variance / Stddev
    // -------------------------
    {
        const double mean = fastdist::math::discrete_uniform_mean(a, b);
        const double var = fastdist::math::discrete_uniform_variance(a, b);
        const double stddev = fastdist::math::discrete_uniform_stddev(a, b);

        assert(std::abs(mean - 3.5) < tol);
        assert(std::abs(var - 35.0 / 12.0) < tol);
        assert(std::abs(stddev - std::sqrt(35.0 / 12.0)) < tol);
    }

    // -------------------------
    // MGF tests
    // -------------------------
    {
        // M(0) = 1
        assert(std::abs(fastdist::math::discrete_uniform_mgf_scalar(0.0, a, b) - 1.0) < tol);

        // M'(0) = mean
        constexpr double dt = 1e-4;
        const double dM = (fastdist::math::discrete_uniform_mgf_scalar(dt, a, b) -
                           fastdist::math::discrete_uniform_mgf_scalar(-dt, a, b)) /
                          (2.0 * dt);

        assert(std::abs(dM - fastdist::math::discrete_uniform_mean(a, b)) < 1e-4);
    }

    // -------------------------
    // CGF tests
    // -------------------------
    {
        // K(0) = 0
        assert(std::abs(fastdist::math::discrete_uniform_cgf_scalar(0.0, a, b)) < tol);

        // K'(0) = mean
        constexpr double dt = 1e-6;
        const double dK = (fastdist::math::discrete_uniform_cgf_scalar(dt, a, b) -
                           fastdist::math::discrete_uniform_cgf_scalar(-dt, a, b)) /
                          (2.0 * dt);

        assert(std::abs(dK - fastdist::math::discrete_uniform_mean(a, b)) < 1e-4);
    }

    // -------------------------
    // RNG tests
    // -------------------------
    {
        constexpr int N = 250000;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = 0; i < N; ++i) {
            const double x = fastdist::math::discrete_uniform_sample(a, b);

            // support check
            assert(x >= a);
            assert(x <= b);

            sum += x;
            sumsq += x * x;
        }

        const double mean = sum / N;
        const double var = sumsq / N - mean * mean;

        assert(std::abs(mean - fastdist::math::discrete_uniform_mean(a, b)) < 1e-2);
        assert(std::abs(var - fastdist::math::discrete_uniform_variance(a, b)) < 1e-2);
    }

    std::cout << "Discrete uniform tests passed.\n";
}
