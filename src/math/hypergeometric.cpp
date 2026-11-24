// Hypergeometric distribution. See include/probforge/math/hypergeometric.h.
#include <algorithm>
#include <cmath>
#include <limits>
#include <probforge/math/hypergeometric.h>
#include <probforge/math/rng.h>
#include <random>

namespace probforge::math {

    namespace {
        // Everything below assumes these hold.
        inline bool params_valid(const int population, const int successes, const int draws) {
            return population >= 0 && successes >= 0 && draws >= 0 && successes <= population && draws <= population;
        }

        // Smallest k with non-zero probability; see the header on why it can exceed 0.
        inline int support_min(const int population, const int successes, const int draws) {
            return std::max(0, draws - (population - successes));
        }

        // Largest k with non-zero probability.
        inline int support_max(const int successes, const int draws) { return std::min(draws, successes); }

        // log C(n, r) through lgamma: the factorials overflow past 170, the ratio does not.
        inline double log_choose(const double n, const double r) {
            return std::lgamma(n + 1.0) - std::lgamma(r + 1.0) - std::lgamma(n - r + 1.0);
        }

        // Unvalidated; callers must have checked the parameters and that k is in support.
        inline double log_pmf_unchecked(const int k, const int population, const int successes, const int draws) {
            const double N = population;
            const double K = successes;
            const double n = draws;
            const double x = k;
            return log_choose(K, x) + log_choose(N - K, n - x) - log_choose(N, n);
        }
    } // namespace

    double hypergeometric_logpmf_scalar(const int k, const int population, const int successes, const int draws) {
        if (!params_valid(population, successes, draws)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (k < support_min(population, successes, draws) || k > support_max(successes, draws)) {
            return -std::numeric_limits<double>::infinity();
        }

        return log_pmf_unchecked(k, population, successes, draws);
    }

    double hypergeometric_pmf_scalar(const int k, const int population, const int successes, const int draws) {
        return std::exp(hypergeometric_logpmf_scalar(k, population, successes, draws));
    }

    double hypergeometric_cdf_scalar(const int k, const int population, const int successes, const int draws) {
        if (!params_valid(population, successes, draws)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const int lo = support_min(population, successes, draws);
        const int hi = support_max(successes, draws);

        if (k < lo) return 0.0;
        if (k >= hi) return 1.0;

        // P(j) = P(j-1) * ((K-j+1) / j) * ((n-j+1) / (N-K-n+j)), so the sum costs
        // one exp. It must start at the bottom of the support, not at 0: below it
        // every term is zero and the recurrence would divide by zero.
        double term = std::exp(log_pmf_unchecked(lo, population, successes, draws));
        double sum = term;

        // Terms only shrink past the mode, so one that no longer moves the total
        // there means every later term is negligible too.
        const double mode = (static_cast<double>(draws) + 1.0) * (static_cast<double>(successes) + 1.0) /
                            (static_cast<double>(population) + 2.0);

        for (int j = lo + 1; j <= k; ++j) {
            const double numerator = static_cast<double>(successes - j + 1) * static_cast<double>(draws - j + 1);
            const double denominator = static_cast<double>(j) * static_cast<double>(population - successes - draws + j);
            term *= numerator / denominator;

            const double next = sum + term;
            if (static_cast<double>(j) > mode && next == sum) break;
            sum = next;
        }

        // Accumulated rounding can leave the total a few ULP outside [0, 1].
        return std::min(std::max(sum, 0.0), 1.0);
    }

    double hypergeometric_mean(const int population, const int successes, const int draws) {
        if (!params_valid(population, successes, draws) || population == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return static_cast<double>(draws) * static_cast<double>(successes) / static_cast<double>(population);
    }

    double hypergeometric_variance(const int population, const int successes, const int draws) {
        if (!params_valid(population, successes, draws) || population == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Deterministic here, and the correction factor would evaluate 0/0.
        if (population == 1) return 0.0;

        const double N = population;
        const double K = successes;
        const double n = draws;

        return n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1.0));
    }

    double hypergeometric_stddev(const int population, const int successes, const int draws) {
        return std::sqrt(hypergeometric_variance(population, successes, draws));
    }

    // log-sum-exp over the finite support: factoring out the largest exponent
    // stops a large t overflowing a term the total would have survived.
    namespace {
        double log_mgf(const double t, const int population, const int successes, const int draws) {
            const int lo = support_min(population, successes, draws);
            const int hi = support_max(successes, draws);

            double max_exponent = -std::numeric_limits<double>::infinity();
            for (int k = lo; k <= hi; ++k) {
                const double e = t * static_cast<double>(k) + log_pmf_unchecked(k, population, successes, draws);
                if (e > max_exponent) max_exponent = e;
            }

            if (!std::isfinite(max_exponent)) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            double sum = 0.0;
            for (int k = lo; k <= hi; ++k) {
                const double e = t * static_cast<double>(k) + log_pmf_unchecked(k, population, successes, draws);
                sum += std::exp(e - max_exponent);
            }

            return max_exponent + std::log(sum);
        }
    } // namespace

    double hypergeometric_mgf_scalar(const double t, const int population, const int successes, const int draws) {
        if (!std::isfinite(t) || !params_valid(population, successes, draws) || population == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::exp(log_mgf(t, population, successes, draws));
    }

    double hypergeometric_cgf_scalar(const double t, const int population, const int successes, const int draws) {
        if (!std::isfinite(t) || !params_valid(population, successes, draws) || population == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return log_mgf(t, population, successes, draws);
    }

    int hypergeometric_sample(const int population, const int successes, const int draws) {
        if (!params_valid(population, successes, draws)) {
            return -1; // signal invalid input
        }

        int remaining_population = population;
        int remaining_successes = successes;
        int observed = 0;

        std::uniform_real_distribution dist(0.0, 1.0);

        for (int i = 0; i < draws; ++i) {
            // Once either pool runs out the rest of the draw is determined.
            if (remaining_successes == 0) break;
            if (remaining_successes == remaining_population) {
                observed += draws - i;
                break;
            }

            const double p_success =
                    static_cast<double>(remaining_successes) / static_cast<double>(remaining_population);
            if (dist(rng()) < p_success) {
                ++observed;
                --remaining_successes;
            }
            --remaining_population;
        }

        return observed;
    }

} // namespace probforge::math
