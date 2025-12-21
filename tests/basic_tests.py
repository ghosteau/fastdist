import math

import fastdist


def main():
    print("Running fastdist basic tests...")

    tol = 1e-12
    nan = float("nan")

    # -------------------------
    # Normal distribution tests
    # -------------------------
    pdf0 = fastdist.normal_pdf_scalar(0.0, 0.0, 1.0)
    assert abs(pdf0 - 0.3989422804014327) < tol

    cdf0 = fastdist.normal_cdf_scalar(0.0, 0.0, 1.0)
    assert abs(cdf0 - 0.5) < tol

    assert fastdist.normal_mean(2.0) == 2.0
    assert fastdist.normal_variance(2.0) == 4.0
    assert fastdist.z_score(1.0, 0.0, 2.0) == 0.5

    # ------------------------------
    # Exponential distribution tests
    # ------------------------------
    pdf0 = fastdist.exponential_pdf_scalar(0.0, 2.0)
    assert abs(pdf0 - 2.0) < tol

    cdf0 = fastdist.exponential_cdf_scalar(0.0, 2.0)
    assert abs(cdf0 - 0.0) < tol

    assert abs(fastdist.exponential_mean(2.0) - 0.5) < tol
    assert abs(fastdist.exponential_variance(2.0) - 0.25) < tol

    # -------------------------
    # Poisson distribution tests
    # -------------------------
    pmf0 = fastdist.poisson_pmf_scalar(0.0, 3.0)
    assert abs(pmf0 - math.exp(-3.0)) < tol

    pmf3 = fastdist.poisson_pmf_scalar(3.0, 3.0)
    assert abs(pmf3 - 0.22404180765538775) < tol

    cdf0 = fastdist.poisson_cdf_scalar(0.0, 3.0)
    assert abs(cdf0 - math.exp(-3.0)) < tol

    assert fastdist.poisson_mean(3.0) == 3.0
    assert fastdist.poisson_variance(3.0) == 3.0

    # -------------------------
    # Bernoulli distribution tests
    # -------------------------
    assert abs(fastdist.bernoulli_pmf_scalar(1, 0.3) - 0.3) < tol
    assert abs(fastdist.bernoulli_pmf_scalar(0, 0.3) - 0.7) < tol
    assert fastdist.bernoulli_pmf_scalar(2, 0.3) == 0.0

    assert abs(fastdist.bernoulli_cdf_scalar(0, 0.3) - 0.7) < tol
    assert abs(fastdist.bernoulli_cdf_scalar(1, 0.3) - 1.0) < tol

    assert abs(fastdist.bernoulli_mean(0.3) - 0.3) < tol
    assert abs(fastdist.bernoulli_variance(0.3) - 0.21) < tol
    assert abs(
        fastdist.bernoulli_stddev(0.3) - math.sqrt(0.21)
    ) < tol

    # -------------------------
    # Binomial distribution tests
    # -------------------------
    pmf2 = fastdist.binomial_pmf_scalar(2, 3, 0.5)
    assert abs(pmf2 - 0.375) < tol

    cdf1 = fastdist.binomial_cdf_scalar(1, 3, 0.5)
    assert abs(cdf1 - 0.5) < tol

    assert abs(fastdist.binomial_mean(3, 0.5) - 1.5) < tol
    assert abs(fastdist.binomial_variance(3, 0.5) - 0.75) < tol
    assert abs(
        fastdist.binomial_stddev(3, 0.5) - math.sqrt(0.75)
    ) < tol

    # -------------------------
    # Discrete Uniform distribution tests
    # -------------------------
    pmf3 = fastdist.discrete_uniform_pmf_scalar(3, 1, 6)
    assert abs(pmf3 - 1.0 / 6.0) < tol

    cdf4 = fastdist.discrete_uniform_cdf_scalar(4, 1, 6)
    assert abs(cdf4 - 4.0 / 6.0) < tol

    assert abs(fastdist.discrete_uniform_mean(1, 6) - 3.5) < tol
    assert abs(
        fastdist.discrete_uniform_variance(1, 6)
        - 35.0 / 12.0
    ) < tol
    assert abs(
        fastdist.discrete_uniform_stddev(1, 6)
        - math.sqrt(35.0 / 12.0)
    ) < tol

    # -------------------------
    # Continuous Uniform distribution tests
    # -------------------------
    pdf0 = fastdist.uniform_pdf_scalar(0.5, 0.0, 1.0)
    assert abs(pdf0 - 1.0) < tol

    pdf_out = fastdist.uniform_pdf_scalar(1.5, 0.0, 1.0)
    assert abs(pdf_out - 0.0) < tol

    cdf0 = fastdist.uniform_cdf_scalar(0.5, 0.0, 1.0)
    assert abs(cdf0 - 0.5) < tol

    cdf_out = fastdist.uniform_cdf_scalar(1.5, 0.0, 1.0)
    assert abs(cdf_out - 1.0) < tol

    assert abs(fastdist.uniform_mean(0.0, 1.0) - 0.5) < tol
    assert abs(
        fastdist.uniform_variance(0.0, 1.0) - 1.0 / 12.0
    ) < tol
    assert abs(
        fastdist.uniform_stddev(0.0, 1.0)
        - math.sqrt(1.0 / 12.0)
    ) < tol

    # -------------------------
    # Geometric distribution tests
    # -------------------------
    pmf1 = fastdist.geometric_pmf_scalar(1, 0.25)
    assert abs(pmf1 - 0.25) < tol

    pmf3 = fastdist.geometric_pmf_scalar(3, 0.25)
    assert abs(pmf3 - (0.25 * math.pow(0.75, 2))) < tol

    cdf2 = fastdist.geometric_cdf_scalar(2, 0.25)
    assert abs(cdf2 - (0.25 + 0.25 * 0.75)) < tol

    mean = fastdist.geometric_mean(0.25)
    assert abs(mean - 4.0) < tol

    var = fastdist.geometric_variance(0.25)
    assert abs(var - 12.0) < tol

    stddev = fastdist.geometric_stddev(0.25)
    assert abs(stddev - math.sqrt(12.0)) < tol

    # -------------------------
    # Chebyshev inequality test
    # -------------------------
    bound = fastdist.chebyshev_bound(4.0, 2.0)
    assert abs(bound - 1.0) < tol

    bad = fastdist.chebyshev_bound(4.0, 0.0)
    assert math.isnan(bad)

    print("All basic tests passed.")


if __name__ == "__main__":
    main()
