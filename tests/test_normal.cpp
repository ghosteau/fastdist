// Unit tests for Normal distribution
#include <cassert>
#include <cmath>
#include <iostream>

#include <fastdist/math/normal.h>

void test_normal() {
    std::cout << "Running normal distribution tests...\n";

    constexpr double tol = 1e-12;
    constexpr double mu = 1.5;
    constexpr double sigma = 2.0;

    // -------------------------
    // PDF / logPDF / CDF tests
    // -------------------------
    {
        const double pdf0 = fastdist::math::normal_pdf_scalar(mu, mu, sigma);
        const double expected_pdf = 1.0 / (sigma * std::sqrt(2.0 * M_PI));
        assert(std::abs(pdf0 - expected_pdf) < tol);

        const double logpdf0 = fastdist::math::normal_logpdf_scalar(mu, mu, sigma);
        assert(std::abs(std::exp(logpdf0) - expected_pdf) < tol);

        const double cdf0 = fastdist::math::normal_cdf_scalar(mu, mu, sigma);
        assert(std::abs(cdf0 - 0.5) < tol);
    }

    // -------------------------
    // Mean / Variance / Stddev
    // -------------------------
    {
        assert(std::abs(fastdist::math::normal_mean(mu) - mu) < tol);
        assert(std::abs(fastdist::math::normal_variance(sigma) - sigma * sigma) < tol);
        assert(std::abs(fastdist::math::normal_stddev(sigma) - sigma) < tol);
    }

    // -------------------------
    // Z-score
    // -------------------------
    {
        const double z = fastdist::math::z_score(mu + sigma, mu, sigma);
        assert(std::abs(z - 1.0) < tol);
    }

    // -------------------------
    // MGF tests
    // M(t) = exp(mu t + 0.5 sigma^2 t^2)
    // -------------------------
    {
        const double mgf0 = fastdist::math::normal_mgf_scalar(0.0, mu, sigma);
        assert(std::abs(mgf0 - 1.0) < tol);

        constexpr double t = 0.25;
        const double expected = std::exp(mu * t + 0.5 * sigma * sigma * t * t);

        const double mgf = fastdist::math::normal_mgf_scalar(t, mu, sigma);
        assert(std::abs(mgf - expected) < tol);

        // M'(0) = mean
        constexpr double dt = 1e-6;
        const double dM =
                (fastdist::math::normal_mgf_scalar(dt, mu, sigma) - fastdist::math::normal_mgf_scalar(-dt, mu, sigma)) /
                (2.0 * dt);

        assert(std::abs(dM - mu) < 1e-6);
    }

    // -------------------------
    // CGF tests
    // K(t) = mu t + 0.5 sigma^2 t^2
    // -------------------------
    {
        const double cgf0 = fastdist::math::normal_cgf_scalar(0.0, mu, sigma);
        assert(std::abs(cgf0) < tol);

        const double t = 0.25;
        const double expected = mu * t + 0.5 * sigma * sigma * t * t;

        const double cgf = fastdist::math::normal_cgf_scalar(t, mu, sigma);
        assert(std::abs(cgf - expected) < tol);

        // K'(0) = mean
        constexpr double dt = 1e-6;
        const double dK =
                (fastdist::math::normal_cgf_scalar(dt, mu, sigma) - fastdist::math::normal_cgf_scalar(-dt, mu, sigma)) /
                (2.0 * dt);

        assert(std::abs(dK - mu) < 1e-6);

        // K''(0) = variance
        const double d2K = (fastdist::math::normal_cgf_scalar(dt, mu, sigma) -
                            2.0 * fastdist::math::normal_cgf_scalar(0.0, mu, sigma) +
                            fastdist::math::normal_cgf_scalar(-dt, mu, sigma)) /
                           (dt * dt);

        assert(std::abs(d2K - sigma * sigma) < 1e-6);
    }

    // -------------------------
    // Normal RNG tests
    // -------------------------
    {
        constexpr int N = 200000;
        double sum = 0.0;
        double sumsq = 0.0;

        for (int i = 0; i < N; ++i) {
            const double x = fastdist::math::normal_sample(mu, sigma);

            sum += x;
            sumsq += x * x;
        }

        const double mean = sum / N;
        const double var = sumsq / N - mean * mean;

        assert(std::abs(mean - mu) < 5e-2);
        assert(std::abs(var - sigma * sigma) < 5e-2);
    }

    // -------------------------
    // Log-normal RNG tests
    // -------------------------
    {
        constexpr int N = 250000;
        double sum = 0.0;

        for (int i = 0; i < N; ++i) {
            const double x = fastdist::math::normal_log_sample(mu, sigma);
            assert(x > 0.0);
            sum += std::log(x);
        }

        const double log_mean = sum / N;
        assert(std::abs(log_mean - mu) < 5e-2);
    }

    std::cout << "Normal distribution tests passed.\n";
}
