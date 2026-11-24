/**
 * @file discrete_uniform.h
 * @brief Discrete uniform distribution on the integers {a, ..., b}.
 *
 *     P(X = x) = 1 / (b - a + 1)
 *
 * See uniform.h for the continuous case.
 *
 * @par Contract
 * NaN unless a <= b; outside the support the PMF is 0.
 *
 * @warning discrete_uniform_sample() signals invalid input with INT_MIN, which
 * is also a legal draw when @p a is INT_MIN.
 */
#ifndef DISCRETE_UNIFORM_H
#define DISCRETE_UNIFORM_H

namespace probforge::math {
    /** @brief P(X = x) = 1 / n, where n = b - a + 1, widened so the full int range cannot wrap. */
    double discrete_uniform_pmf_scalar(int x, int a, int b);

    /** @brief P(X <= x) = (x - a + 1) / n. */
    double discrete_uniform_cdf_scalar(int x, int a, int b);

    /** @brief E[X] = (a + b) / 2. */
    double discrete_uniform_mean(int a, int b);

    /** @brief Var[X] = (n^2 - 1) / 12. */
    double discrete_uniform_variance(int a, int b);

    /** @brief SD[X] = sqrt((n^2 - 1) / 12). */
    double discrete_uniform_stddev(int a, int b);

    /** @brief MGF, M(t) = e^(ta) (e^(tn) - 1) / (n (e^t - 1)); M(0) = 1. */
    double discrete_uniform_mgf_scalar(double t, int a, int b);

    /** @brief CGF, K(t) = log M(t). */
    double discrete_uniform_cgf_scalar(double t, int a, int b);

    /** @brief One draw, uniform over {a, ..., b}. See the warning above on its error signal. */
    int discrete_uniform_sample(int a, int b);
} // namespace probforge::math

#endif // DISCRETE_UNIFORM_H
