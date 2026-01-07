// Unit tests for Chi-square distribution
#include <cassert>
#include <cmath>
#include <iostream>

#include <fastdist/math/chi_square.h>

void test_chi_square() {
    std::cout << "Running Chi-square distribution tests...\n";

    constexpr double tol = 1e-12;
    constexpr double k = 6.0; // degrees of freedom

    // -------------------------
    // PDF / CDF tests
    // -------------------------
    {
        const double x = 4.0;

        const double pdf = fastdist::math::chi_square_pdf_scalar(x, k);

        // Expected PDF via Gamma(k/2, 2)
        const double expected_pdf =
                std::pow(x, k / 2.0 - 1.0) * std::exp(-x / 2.0) / (std::pow(2.0, k / 2.0) * std::tgamma(k / 2.0));

        assert(std::abs(pdf - expected_pdf) < tol);

        // Outside support
        assert(fastdist::math::chi_square_pdf_scalar(-1.0, k) == 0.0);

        // CDF sanity
        const double cdf1 = fastdist::math::chi_square_cdf_scalar(2.0, k);
        const double cdf2 = fastdist::math::chi_square_cdf_scalar(8.0, k);

        assert(cdf1 >= 0.0);
        assert(cdf2 <= 1.0);
        assert(cdf2 > cdf1);
    }

    // -------------------------
    // Mean / Variance / Stddev
    // -------------------------
    {
        const double mean = fastdist::math::chi_square_mean(k);
        const double var = fastdist::math::chi_square_variance(k);
        const double std = fastdist::math::chi_square_stddev(k);

        assert(std::abs(mean - k) < tol);
        assert(std::abs(var - 2.0 * k) < tol);
        assert(std::abs(std - std::sqrt(2.0 * k)) < tol);
    }

    // -------------------------
    // MGF / CGF tests
    // -------------------------
    {
        const double t = 0.1; // must satisfy t < 1/2

        const double mgf = fastdist::math::chi_square_mgf_scalar(t, k);

        const double expected_mgf = std::pow(1.0 - 2.0 * t, -k / 2.0);

        assert(std::abs(mgf - expected_mgf) < tol);

        const double cgf = fastdist::math::chi_square_cgf_scalar(t, k);

        assert(std::abs(cgf - std::log(expected_mgf)) < tol);

        // M'(0) = mean
        constexpr double dt = 1e-6;
        const double dM =
                (fastdist::math::chi_square_mgf_scalar(dt, k) - fastdist::math::chi_square_mgf_scalar(-dt, k)) /
                (2.0 * dt);

        assert(std::abs(dM - fastdist::math::chi_square_mean(k)) < 1e-6);
    }

    // -------------------------
    // RNG sanity check
    // -------------------------
    {
        constexpr int N = 250000;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = 0; i < N; ++i) {
            const double x = fastdist::math::chi_square_sample(k);

            assert(x >= 0.0);

            sum += x;
            sumsq += x * x;
        }

        const double mean = sum / N;
        const double var = sumsq / N - mean * mean;

        assert(std::abs(mean - fastdist::math::chi_square_mean(k)) < 5e-2);

        assert(std::abs(var - fastdist::math::chi_square_variance(k)) < 5e-1);
    }

    std::cout << "Chi-square distribution tests passed!\n";
}
