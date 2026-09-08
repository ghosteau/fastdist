// Configuration header for FastDist library

#ifndef CONFIG_H
#define CONFIG_H

// Iteration ceiling for the Beta and Gamma series and continued fractions.
//
// Every loop using this exits as soon as its term falls below EPS, so the bound
// only matters for parameters that converge slowly, and raising it costs
// ordinary calls nothing.
//
// The gamma series is the binding constraint: near x = alpha it needs roughly
// sqrt(2 * alpha * ln(1/EPS)) terms, about 227 at alpha = 1000 and 683 at
// alpha = 10000. At the previous ceiling of 100 it simply stopped early and
// returned the truncated sum, so Gamma(1000, 0.5).cdf(500) was wrong by 9e-4
// and Gamma(10000, ...) by 0.16, with no indication anything had gone wrong.
//
// 1000 covers alpha up to roughly 20000. Beyond that the result degrades
// silently again; a shape parameter that large needs a different algorithm
// (a normal approximation, or Temme's uniform asymptotic expansion) rather
// than a larger ceiling.
constexpr unsigned int MAX_ITER = 1000;
constexpr double EPS = 1e-12;
constexpr double FPMIN = 1e-30;

#endif // CONFIG_H
