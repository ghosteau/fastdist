#include <cassert>
#include <cmath>
#include <fastdist/math/uniform.h>
#include <iostream>

void test_uniform() {
    std::cout << "Running continuous uniform distribution tests...\n";

    constexpr double tol = 1e-12;
    constexpr double a = 2.0;
    constexpr double b = 5.0;

    // -------------------------
    // PDF tests
    // -------------------------
    {
        // Inside [a,b]
        const double x = 3.0;
        const double pdf = fastdist::math::uniform_pdf_scalar(x, a, b);
        assert(std::abs(pdf - 1.0 / (b - a)) < tol);

        // At boundaries
        assert(std::abs(fastdist::math::uniform_pdf_scalar(a, a, b) - 1.0 / (b - a)) < tol);
        assert(std::abs(fastdist::math::uniform_pdf_scalar(b, a, b) - 1.0 / (b - a)) < tol);

        // Outside [a,b]
        assert(fastdist::math::uniform_pdf_scalar(a - 1.0, a, b) == 0.0);
        assert(fastdist::math::uniform_pdf_scalar(b + 1.0, a, b) == 0.0);
    }

    // -------------------------
    // CDF tests
    // -------------------------
    {
        const double x = 3.5;
        const double cdf = fastdist::math::uniform_cdf_scalar(x, a, b);
        assert(std::abs(cdf - (x - a) / (b - a)) < tol);

        // Boundary conditions
        assert(fastdist::math::uniform_cdf_scalar(a, a, b) == 0.0);
        assert(fastdist::math::uniform_cdf_scalar(b, a, b) == 1.0);
        assert(fastdist::math::uniform_cdf_scalar(a - 1.0, a, b) == 0.0);
        assert(fastdist::math::uniform_cdf_scalar(b + 1.0, a, b) == 1.0);
    }

    // -------------------------
    // Mean / Variance / Stddev
    // -------------------------
    {
        const double mean = fastdist::math::uniform_mean(a, b);
        const double var = fastdist::math::uniform_variance(a, b);
        const double stddev = fastdist::math::uniform_stddev(a, b);

        assert(std::abs(mean - 0.5 * (a + b)) < tol);
        assert(std::abs(var - (b - a) * (b - a) / 12.0) < tol);
        assert(std::abs(stddev - std::sqrt((b - a) * (b - a) / 12.0)) < tol);
    }

    // -------------------------
    // MGF / CGF tests
    // -------------------------
    {
        const double t = 0.5;
        const double mgf = fastdist::math::uniform_mgf_scalar(t, a, b);
        const double expected_mgf = (std::exp(b * t) - std::exp(a * t)) / (t * (b - a));
        assert(std::abs(mgf - expected_mgf) < tol);

        const double cgf = fastdist::math::uniform_cgf_scalar(t, a, b);
        const double expected_cgf = std::log(expected_mgf);
        assert(std::abs(cgf - expected_cgf) < tol);

        // t = 0 should return 1 for MGF
        assert(fastdist::math::uniform_mgf_scalar(0.0, a, b) == 1.0);
    }

    // -------------------------
    // RNG tests
    // -------------------------
    {
        constexpr int N = 250000;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = 0; i < N; ++i) {
            double x = fastdist::math::uniform_sample(a, b);
            sum += x;
            sumsq += x * x;

            // check bounds
            assert(x >= a && x <= b);
        }

        double mean = sum / N;
        double var = sumsq / N - mean * mean;

        // RNG mean / variance should be close
        double expected_mean = 0.5 * (a + b);
        double expected_var = (b - a) * (b - a) / 12.0;

        assert(std::abs(mean - expected_mean) < 5e-3);
        assert(std::abs(var - expected_var) < 5e-3);
    }

    std::cout << "Continuous uniform distribution tests passed!\n";
}
