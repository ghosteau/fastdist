// Basic module tests for fastdist library.
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include <fastdist/math/bernoulli.h>
#include <fastdist/math/binomial.h>
#include <fastdist/math/discrete_uniform.h>
#include <fastdist/math/exponential.h>
#include <fastdist/math/geometric.h>
#include <fastdist/math/normal.h>
#include <fastdist/math/poisson.h>
#include <fastdist/math/uniform.h>

int main() {
    std::cout << "Running fastdist basic tests...\n";

    const double tol = 1e-12;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // -------------------------
    // Normal distribution tests
    // -------------------------
    {
        double pdf0 = fastdist::math::normal_pdf_scalar(0.0, 0.0, 1.0);
        assert(std::abs(pdf0 - 0.3989422804014327) < tol);

        double cdf0 = fastdist::math::normal_cdf_scalar(0.0, 0.0, 1.0);
        assert(std::abs(cdf0 - 0.5) < tol);

        assert(fastdist::math::normal_mean(2.0) == 2.0);
        assert(fastdist::math::normal_variance(2.0) == 4.0);

        assert(fastdist::math::z_score(1.0, 0.0, 2.0) == 0.5);
    }

    // ------------------------------
    // Exponential distribution tests
    // ------------------------------
    {
        double pdf0 = fastdist::math::exponential_pdf_scalar(0.0, 2.0);
        assert(std::abs(pdf0 - 2.0) < tol);

        double cdf0 = fastdist::math::exponential_cdf_scalar(0.0, 2.0);
        assert(std::abs(cdf0 - 0.0) < tol);

        assert(std::abs(fastdist::math::exponential_mean(2.0) - 0.5) < tol);
        assert(std::abs(fastdist::math::exponential_variance(2.0) - 0.25) < tol);
    }

    // -------------------------
    // Poisson distribution tests
    // -------------------------
    {
        double pmf0 = fastdist::math::poisson_pmf_scalar(0.0, 3.0);
        assert(std::abs(pmf0 - std::exp(-3.0)) < tol);

        double pmf3 = fastdist::math::poisson_pmf_scalar(3.0, 3.0);
        assert(std::abs(pmf3 - 0.22404180765538775) < tol);

        double cdf0 = fastdist::math::poisson_cdf_scalar(0.0, 3.0);
        assert(std::abs(cdf0 - std::exp(-3.0)) < tol);

        assert(fastdist::math::poisson_mean(3.0) == 3.0);
        assert(fastdist::math::poisson_variance(3.0) == 3.0);
    }

    // -------------------------
    // Bernoulli distribution tests
    // -------------------------
    {
        assert(std::abs(fastdist::math::bernoulli_pmf_scalar(1, 0.3) - 0.3) < tol);
        assert(std::abs(fastdist::math::bernoulli_pmf_scalar(0, 0.3) - 0.7) < tol);
        assert(fastdist::math::bernoulli_pmf_scalar(2, 0.3) == 0.0);

        assert(std::abs(fastdist::math::bernoulli_cdf_scalar(0, 0.3) - 0.7) < tol);
        assert(std::abs(fastdist::math::bernoulli_cdf_scalar(1, 0.3) - 1.0) < tol);

        assert(std::abs(fastdist::math::bernoulli_mean(0.3) - 0.3) < tol);
        assert(std::abs(fastdist::math::bernoulli_variance(0.3) - 0.21) < tol);
        assert(std::abs(fastdist::math::bernoulli_stddev(0.3) - std::sqrt(0.21)) < tol);
    }

    // -------------------------
    // Binomial distribution tests
    // -------------------------
    {
        double pmf2 = fastdist::math::binomial_pmf_scalar(2, 3, 0.5);
        assert(std::abs(pmf2 - 0.375) < tol);

        double cdf1 = fastdist::math::binomial_cdf_scalar(1, 3, 0.5);
        assert(std::abs(cdf1 - 0.5) < tol);

        assert(std::abs(fastdist::math::binomial_mean(3, 0.5) - 1.5) < tol);
        assert(std::abs(fastdist::math::binomial_variance(3, 0.5) - 0.75) < tol);
        assert(std::abs(fastdist::math::binomial_stddev(3, 0.5) - std::sqrt(0.75)) < tol);
    }

    // -------------------------
    // Discrete Uniform distribution tests
    // -------------------------
    {
        double pmf3 = fastdist::math::discrete_uniform_pmf_scalar(3, 1, 6);
        assert(std::abs(pmf3 - 1.0 / 6.0) < tol);

        double cdf4 = fastdist::math::discrete_uniform_cdf_scalar(4, 1, 6);
        assert(std::abs(cdf4 - 4.0 / 6.0) < tol);

        assert(std::abs(fastdist::math::discrete_uniform_mean(1, 6) - 3.5) < tol);
        assert(std::abs(fastdist::math::discrete_uniform_variance(1, 6) - 35.0 / 12.0) < tol);
        assert(std::abs(fastdist::math::discrete_uniform_stddev(1, 6) - std::sqrt(35.0 / 12.0)) < tol);
    }

    // -------------------------
    // Continuous Uniform distribution tests
    // -------------------------
    {
        double pdf0 = fastdist::math::uniform_pdf_scalar(0.5, 0.0, 1.0);
        assert(std::abs(pdf0 - 1.0) < tol);

        double pdf_out = fastdist::math::uniform_pdf_scalar(1.5, 0.0, 1.0);
        assert(std::abs(pdf_out - 0.0) < tol);

        double cdf0 = fastdist::math::uniform_cdf_scalar(0.5, 0.0, 1.0);
        assert(std::abs(cdf0 - 0.5) < tol);

        double cdf_out = fastdist::math::uniform_cdf_scalar(1.5, 0.0, 1.0);
        assert(std::abs(cdf_out - 1.0) < tol);

        assert(std::abs(fastdist::math::uniform_mean(0.0, 1.0) - 0.5) < tol);
        assert(std::abs(fastdist::math::uniform_variance(0.0, 1.0) - 1.0 / 12.0) < tol);
        assert(std::abs(fastdist::math::uniform_stddev(0.0, 1.0) - std::sqrt(1.0 / 12.0)) < tol);
    }

    // -------------------------
    // Geometric distribution tests
    // -------------------------
    {
        double pmf1 = fastdist::math::geometric_pmf_scalar(1, 0.25); // k=1
        assert(std::abs(pmf1 - 0.25) < tol);

        double pmf3 = fastdist::math::geometric_pmf_scalar(3, 0.25);
        assert(std::abs(pmf3 - (0.25 * std::pow(0.75, 2))) < tol);

        double cdf2 = fastdist::math::geometric_cdf_scalar(2, 0.25);
        assert(std::abs(cdf2 - (0.25 + 0.25 * 0.75)) < tol);

        double mean = fastdist::math::geometric_mean(0.25);
        assert(std::abs(mean - 4.0) < tol); // 1/p

        double var = fastdist::math::geometric_variance(0.25);
        assert(std::abs(var - 12.0) < tol); // (1-p)/p^2

        double stddev = fastdist::math::geometric_stddev(0.25);
        assert(std::abs(stddev - std::sqrt(12.0)) < tol);
    }

    std::cout << "All basic tests passed.\n";
    return 0;
}
