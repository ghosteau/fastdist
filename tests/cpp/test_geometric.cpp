// Unit tests for Geometric distribution
#include <cassert>
#include <cmath>
#include <iostream>

#include <fastdist/math/geometric.h>

void test_geometric() {
    std::cout << "Running geometric distribution tests...\n";

    constexpr double tol = 1e-12;
    constexpr double p = 0.25; // success probability

    // -------------------------
    // PMF / CDF tests
    // Support: k = 1, 2, 3, ...
    // -------------------------
    {
        const double pmf1 = fastdist::math::geometric_pmf_scalar(1, p);
        assert(std::abs(pmf1 - p) < tol);

        const double pmf3 = fastdist::math::geometric_pmf_scalar(3, p);
        assert(std::abs(pmf3 - (p * std::pow(1.0 - p, 2))) < tol);

        const double pmf0 = fastdist::math::geometric_pmf_scalar(0, p);
        assert(pmf0 == 0.0);

        const double cdf1 = fastdist::math::geometric_cdf_scalar(1, p);
        assert(std::abs(cdf1 - p) < tol);

        const double cdf3 = fastdist::math::geometric_cdf_scalar(3, p);
        const double expected = 1.0 - std::pow(1.0 - p, 3);
        assert(std::abs(cdf3 - expected) < tol);

        const double cdf0 = fastdist::math::geometric_cdf_scalar(0, p);
        assert(cdf0 == 0.0);
    }

    // -------------------------
    // Mean / Variance / Stddev
    // -------------------------
    {
        const double mean = fastdist::math::geometric_mean(p);
        const double var = fastdist::math::geometric_variance(p);
        const double stddev = fastdist::math::geometric_stddev(p);

        assert(std::abs(mean - (1.0 / p)) < tol);
        assert(std::abs(var - ((1.0 - p) / (p * p))) < tol);
        assert(std::abs(stddev - std::sqrt((1.0 - p) / (p * p))) < tol);
    }

    // -------------------------
    // MGF tests
    // M(t) = p e^t / (1 - (1-p)e^t), t < -ln(1-p)
    // -------------------------
    {
        assert(std::abs(fastdist::math::geometric_mgf_scalar(0.0, p) - 1.0) < tol);

        constexpr double t = 0.2;
        const double et = std::exp(t);
        const double expected = (p * et) / (1.0 - (1.0 - p) * et);

        const double mgf = fastdist::math::geometric_mgf_scalar(t, p);
        assert(std::abs(mgf - expected) < tol);

        // M'(0) = mean
        constexpr double dt = 1e-6;
        const double dM = (fastdist::math::geometric_mgf_scalar(dt, p) - fastdist::math::geometric_mgf_scalar(-dt, p)) /
                          (2.0 * dt);

        assert(std::abs(dM - fastdist::math::geometric_mean(p)) < 1e-6);
    }

    // -------------------------
    // CGF tests
    // K(t) = log p + t - log(1 - (1-p)e^t)
    // -------------------------
    {
        assert(std::abs(fastdist::math::geometric_cgf_scalar(0.0, p)) < tol);

        constexpr double t = 0.2;
        const double expected = std::log(p) + t - std::log(1.0 - (1.0 - p) * std::exp(t));

        const double cgf = fastdist::math::geometric_cgf_scalar(t, p);
        assert(std::abs(cgf - expected) < tol);

        // K'(0) = mean
        const double dt = 1e-6;
        const double dK = (fastdist::math::geometric_cgf_scalar(dt, p) - fastdist::math::geometric_cgf_scalar(-dt, p)) /
                          (2.0 * dt);

        assert(std::abs(dK - fastdist::math::geometric_mean(p)) < 1e-6);
    }

    // -------------------------
    // RNG tests
    // -------------------------
    {
        constexpr int N = 250000;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = 0; i < N; ++i) {
            const int x = fastdist::math::geometric_sample(p);

            // support check
            assert(x >= 1);

            sum += x;
            sumsq += static_cast<double>(x) * x;
        }

        const double mean = sum / N;
        const double var = sumsq / N - mean * mean;

        assert(std::abs(mean - fastdist::math::geometric_mean(p)) < 0.15);
        // SE(var) = 0.068 at N, so 0.5 is ~7.3 sigma. The mean assertion above
        // shares the 0.15 literal but has SE 6.9e-3, which is ~22 sigma already.
        // See ghosteau/fastdist#2.
        assert(std::abs(var - fastdist::math::geometric_variance(p)) < 0.5);
    }

    std::cout << "Geometric distribution tests passed.\n";
}
