// Unit tests for Gamma distribution
#include <cassert>
#include <cmath>
#include <iostream>

#include <fastdist/math/gamma.h>

void test_gamma() {
    std::cout << "Running Gamma distribution tests...\n";

    constexpr double tol = 1e-12;
    constexpr double alpha = 3.0; // shape
    constexpr double theta = 2.0; // scale

    // -------------------------
    // PDF / CDF tests
    // -------------------------
    {
        const double x = 4.0;

        const double pdf = fastdist::math::gamma_pdf_scalar(x, alpha, theta);

        const double expected_pdf =
                std::pow(x, alpha - 1.0) * std::exp(-x / theta) / (std::tgamma(alpha) * std::pow(theta, alpha));

        assert(std::abs(pdf - expected_pdf) < tol);

        // Outside support
        assert(fastdist::math::gamma_pdf_scalar(-1.0, alpha, theta) == 0.0);

        // CDF monotonicity sanity
        const double cdf1 = fastdist::math::gamma_cdf_scalar(2.0, alpha, theta);
        const double cdf2 = fastdist::math::gamma_cdf_scalar(6.0, alpha, theta);

        assert(cdf1 >= 0.0);
        assert(cdf2 <= 1.0);
        assert(cdf2 > cdf1);
    }

    // -------------------------
    // Mean / Variance / Stddev
    // -------------------------
    {
        const double mean = fastdist::math::gamma_mean(alpha, theta);
        const double var = fastdist::math::gamma_variance(alpha, theta);
        const double std = fastdist::math::gamma_stddev(alpha, theta);

        assert(std::abs(mean - alpha * theta) < tol);
        assert(std::abs(var - alpha * theta * theta) < tol);
        assert(std::abs(std - std::sqrt(alpha * theta * theta)) < tol);
    }

    // -------------------------
    // MGF / CGF tests
    // -------------------------
    {
        const double t = 0.1; // must satisfy t < 1/theta

        const double mgf = fastdist::math::gamma_mgf_scalar(t, alpha, theta);

        const double expected_mgf = std::pow(1.0 - theta * t, -alpha);

        assert(std::abs(mgf - expected_mgf) < tol);

        const double cgf = fastdist::math::gamma_cgf_scalar(t, alpha, theta);

        assert(std::abs(cgf - std::log(expected_mgf)) < tol);

        // M'(0) = mean
        constexpr double dt = 1e-6;
        const double dM = (fastdist::math::gamma_mgf_scalar(dt, alpha, theta) -
                           fastdist::math::gamma_mgf_scalar(-dt, alpha, theta)) /
                          (2.0 * dt);

        assert(std::abs(dM - fastdist::math::gamma_mean(alpha, theta)) < 1e-6);
    }

    // -------------------------
    // RNG sanity check
    // -------------------------
    {
        constexpr int N = 250000;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = 0; i < N; ++i) {
            const double x = fastdist::math::gamma_sample(alpha, theta);

            // support check
            assert(x >= 0.0);

            sum += x;
            sumsq += x * x;
        }

        const double mean = sum / N;
        const double var = sumsq / N - mean * mean;

        assert(std::abs(mean - fastdist::math::gamma_mean(alpha, theta)) < 5e-2);

        assert(std::abs(var - fastdist::math::gamma_variance(alpha, theta)) < 5e-1);
    }

    std::cout << "Gamma distribution tests passed!\n";
}
