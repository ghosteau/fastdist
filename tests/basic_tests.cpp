// Basic module tests for fastdist library.
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include <fastdist/math/normal.h>
#include <fastdist/math/exponential.h>
#include <fastdist/math/poisson.h>

int main() {
    std::cout << "Running fastdist basic tests...\n";

    // -------------------------
    // Normal distribution tests
    // -------------------------
    {
        double pdf0 = fastdist::math::normal_pdf_scalar(0.0, 0.0, 1.0);
        assert(std::abs(pdf0 - 0.3989422804014327) < 1e-12);

        double cdf0 = fastdist::math::normal_cdf_scalar(0.0, 0.0, 1.0);
        assert(std::abs(cdf0 - 0.5) < 1e-12);

        assert(fastdist::math::normal_mean(2.0) == 2.0);
        assert(fastdist::math::normal_variance(2.0) == 4.0);

        // TODO: Add log-normal PDF testing
    }

    // ------------------------------
    // Exponential distribution tests
    // ------------------------------
    {
        double pdf0 = fastdist::math::exponential_pdf_scalar(0.0, 2.0);
        assert(std::abs(pdf0 - 2.0) < 1e-12);

        double cdf0 = fastdist::math::exponential_cdf_scalar(0.0, 2.0);
        assert(std::abs(cdf0 - 0.0) < 1e-12);

        assert(std::abs(fastdist::math::exponential_mean(2.0) - 0.5) < 1e-12);
        assert(std::abs(fastdist::math::exponential_variance(2.0) - 0.25) < 1e-12);
    }

    // -------------------------
    // Poisson distribution tests
    // -------------------------
    {
        double pmf0 = fastdist::math::poisson_pmf_scalar(0.0, 3.0);
        assert(std::abs(pmf0 - std::exp(-3.0)) < 1e-12);

        double pmf3 = fastdist::math::poisson_pmf_scalar(3.0, 3.0);
        assert(std::abs(pmf3 - 0.22404180765538775) < 1e-12);

        double cdf0 = fastdist::math::poisson_cdf_scalar(0.0, 3.0);
        assert(std::abs(cdf0 - std::exp(-3.0)) < 1e-12);

        assert(fastdist::math::poisson_mean(3.0) == 3.0);
        assert(fastdist::math::poisson_variance(3.0) == 3.0);
    }

    // --------------------------------
    // Invalid / boundary input tests
    // --------------------------------
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();

        // Normal: non-finite inputs or non-positive sigma -> NaN
        assert(std::isnan(fastdist::math::normal_pdf_scalar(nan, 0.0, 1.0)));
        assert(std::isnan(fastdist::math::normal_pdf_scalar(0.0, nan, 1.0)));
        assert(std::isnan(fastdist::math::normal_pdf_scalar(0.0, 0.0, nan)));
        assert(std::isnan(fastdist::math::normal_pdf_scalar(inf, 0.0, 1.0)));
        assert(std::isnan(fastdist::math::normal_cdf_scalar(0.0, 0.0, 0.0))); // sigma == 0
        assert(std::isnan(fastdist::math::normal_pdf_scalar(0.0, 0.0, -1.0))); // sigma < 0

        // Exponential: non-finite inputs or non-positive lambda -> NaN
        assert(std::isnan(fastdist::math::exponential_pdf_scalar(nan, 2.0)));
        assert(std::isnan(fastdist::math::exponential_pdf_scalar(0.0, nan)));
        assert(std::isnan(fastdist::math::exponential_cdf_scalar(0.0, 0.0))); // lambda == 0
        assert(std::isnan(fastdist::math::exponential_pdf_scalar(0.0, -1.0))); // lambda < 0

        // Exponential: x < 0 -> 0.0
        assert(std::abs(fastdist::math::exponential_pdf_scalar(-1.0, 2.0) - 0.0) < 1e-12);
        assert(std::abs(fastdist::math::exponential_cdf_scalar(-1.0, 2.0) - 0.0) < 1e-12);
    }

    // TODO: Add Bernoulli distribution tests
    // TODO: Add Binomial distribution tests

    std::cout << "All basic tests passed.\n";
    return 0;
}