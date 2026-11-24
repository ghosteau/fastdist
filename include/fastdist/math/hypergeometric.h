// Header file for hypergeometric distribution functions
#ifndef HYPERGEOMETRIC_H
#define HYPERGEOMETRIC_H

// The hypergeometric distribution counts successes in a sample drawn WITHOUT
// replacement, which is what separates it from the binomial: each draw changes
// the composition of what is left. It is the exact null distribution behind
// Fisher's exact test and behind enrichment tests generally.
//
//     population  N  size of the whole population
//     successes   K  items in the population that count as a success, K <= N
//     draws       n  items drawn, without replacement, n <= N
//     k              successes observed among the draws
//
//     P(X = k) = C(K, k) C(N - K, n - k) / C(N, n)
//
// The support is max(0, n - (N - K)) <= k <= min(n, K): a draw cannot contain
// more successes than exist or than were drawn, and if the draw is larger than
// the number of failures available it must contain some successes.
//
// Parameters are named rather than given as the (M, n, N) triple SciPy uses,
// because that ordering is easy to transpose silently. The mapping is
// SciPy's M = population, n = successes, N = draws.
namespace fastdist::math {
    // Computes the log probability mass function (log PMF) of the hypergeometric distribution
    double hypergeometric_logpmf_scalar(int k, int population, int successes, int draws);
    // Computes the probability mass function (PMF) of the hypergeometric distribution
    double hypergeometric_pmf_scalar(int k, int population, int successes, int draws);
    // Computes the cumulative distribution function (CDF) of the hypergeometric distribution
    double hypergeometric_cdf_scalar(int k, int population, int successes, int draws);
    // Computes the mean of the hypergeometric distribution
    double hypergeometric_mean(int population, int successes, int draws);
    // Computes the variance of the hypergeometric distribution
    double hypergeometric_variance(int population, int successes, int draws);
    // Computes the standard deviation of the hypergeometric distribution
    double hypergeometric_stddev(int population, int successes, int draws);
    // Computes the moment generating function (MGF) at point t
    double hypergeometric_mgf_scalar(double t, int population, int successes, int draws);
    // Computes the cumulant generating function (CGF) at point t
    double hypergeometric_cgf_scalar(double t, int population, int successes, int draws);
    // Draws a random sample from the hypergeometric distribution
    int hypergeometric_sample(int population, int successes, int draws);
} // namespace fastdist::math

#endif // HYPERGEOMETRIC_H
