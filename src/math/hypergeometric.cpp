// Function definitions for hypergeometric distribution functions
#include <algorithm>
#include <cmath>
#include <fastdist/math/hypergeometric.h>
#include <fastdist/math/rng.h>
#include <limits>
#include <random>

namespace fastdist::math {

    namespace {
        // A population cannot hold more successes than items, and no more can be
        // drawn than exist. Everything below assumes these hold.
        inline bool params_valid(const int population, const int successes, const int draws) {
            return population >= 0 && successes >= 0 && draws >= 0 && successes <= population && draws <= population;
        }

        // Smallest k with non-zero probability: if the draw is larger than the
        // number of failures available, the surplus must be successes.
        inline int support_min(const int population, const int successes, const int draws) {
            return std::max(0, draws - (population - successes));
        }

        // Largest k with non-zero probability: cannot exceed either the successes
        // in the population or the size of the draw.
        inline int support_max(const int successes, const int draws) { return std::min(draws, successes); }

        // log C(n, r), through lgamma so the coefficient cannot overflow. The
        // factorials themselves exceed a double past 170 while their ratio stays
        // an ordinary number.
        inline double log_choose(const double n, const double r) {
            return std::lgamma(n + 1.0) - std::lgamma(r + 1.0) - std::lgamma(n - r + 1.0);
        }

        // Unvalidated log PMF, for callers that have already checked their
        // arguments and know k is inside the support.
        inline double log_pmf_unchecked(const int k, const int population, const int successes, const int draws) {
            const double N = population;
            const double K = successes;
            const double n = draws;
            const double x = k;
            return log_choose(K, x) + log_choose(N - K, n - x) - log_choose(N, n);
        }
    } // namespace

    // log P(X = k)
    double hypergeometric_logpmf_scalar(const int k, const int population, const int successes, const int draws) {
        if (!params_valid(population, successes, draws)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (k < support_min(population, successes, draws) || k > support_max(successes, draws)) {
            return -std::numeric_limits<double>::infinity();
        }

        return log_pmf_unchecked(k, population, successes, draws);
    }

    // P(X = k) = C(K, k) C(N-K, n-k) / C(N, n)
    double hypergeometric_pmf_scalar(const int k, const int population, const int successes, const int draws) {
        return std::exp(hypergeometric_logpmf_scalar(k, population, successes, draws));
    }

    // P(X <= k)
    double hypergeometric_cdf_scalar(const int k, const int population, const int successes, const int draws) {
        if (!params_valid(population, successes, draws)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const int lo = support_min(population, successes, draws);
        const int hi = support_max(successes, draws);

        if (k < lo) return 0.0;
        if (k >= hi) return 1.0;

        // Consecutive PMF terms satisfy
        //     P(j) = P(j-1) * ((K - j + 1) / j) * ((n - j + 1) / (N - K - n + j))
        // so the whole sum costs a single exp rather than one per term.
        //
        // Summing from the lower end of the support rather than from 0 matters:
        // below support_min every term is zero, and starting there would divide
        // by a zero denominator in the recurrence.
        double term = std::exp(log_pmf_unchecked(lo, population, successes, draws));
        double sum = term;

        // Terms rise to the mode and fall after it, so once one past the mode no
        // longer moves the total, none of the rest can either. The support can
        // hold billions of points, and the sum converges long before the end.
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

        // Summing PMF terms accumulates rounding error, so the total can land a
        // few ULP outside [0, 1].
        return std::min(std::max(sum, 0.0), 1.0);
    }

    // E[X] = n K / N
    double hypergeometric_mean(const int population, const int successes, const int draws) {
        if (!params_valid(population, successes, draws) || population == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return static_cast<double>(draws) * static_cast<double>(successes) / static_cast<double>(population);
    }

    // Var[X] = n (K/N) ((N-K)/N) ((N-n)/(N-1))
    //
    // The last factor is the finite population correction, and it is what
    // separates this from a binomial variance: drawing without replacement
    // reduces the spread, down to zero when the whole population is drawn.
    double hypergeometric_variance(const int population, const int successes, const int draws) {
        if (!params_valid(population, successes, draws) || population == 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // A population of one is deterministic -- the draw is either empty or is
        // the whole population -- so the variance is zero. The correction factor
        // would evaluate 0/0 here. SciPy reports NaN for this case.
        if (population == 1) return 0.0;

        const double N = population;
        const double K = successes;
        const double n = draws;

        return n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1.0));
    }

    double hypergeometric_stddev(const int population, const int successes, const int draws) {
        return std::sqrt(hypergeometric_variance(population, successes, draws));
    }

    // M_X(t) = sum_k e^{tk} P(X = k)
    //
    // The hypergeometric MGF has no elementary closed form -- it is a Gaussian
    // hypergeometric function 2F1 -- but the support is finite, so the defining
    // sum is exact and finite too. It is accumulated by the log-sum-exp trick:
    // the largest exponent is factored out before exponentiating, so a large t
    // cannot overflow an intermediate term that the total would survive.
    namespace {
        // Returns log M_X(t), the CGF, which the MGF then exponentiates.
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

    // X ~ Hypergeometric(N, K, n)
    //
    // Drawn by simulating the sampling directly: at each step the chance that
    // the next item is a success is the proportion of successes among what is
    // left. There is no std::hypergeometric_distribution, and this is exact for
    // any parameters rather than an approximation that degrades at the edges.
    int hypergeometric_sample(const int population, const int successes, const int draws) {
        if (!params_valid(population, successes, draws)) {
            return -1; // signal invalid input
        }

        int remaining_population = population;
        int remaining_successes = successes;
        int observed = 0;

        std::uniform_real_distribution dist(0.0, 1.0);

        for (int i = 0; i < draws; ++i) {
            // Once the successes or the failures run out the rest of the draw is
            // determined, so stop rolling for it.
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

} // namespace fastdist::math
