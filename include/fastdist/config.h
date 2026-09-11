// Configuration header for FastDist library

#ifndef CONFIG_H
#define CONFIG_H

// Iteration ceiling for the Beta and Gamma series and continued fractions.
// Every loop stops as soon as its term falls below EPS, so the ceiling only
// costs anything for slowly converging parameters. The gamma series is the
// binding case: near x = alpha it needs about sqrt(2 alpha ln(1/EPS)) terms,
// so 1000 covers alpha up to roughly 20000. Beyond that the truncated sum is
// returned without warning; shapes that large need an asymptotic method such
// as Temme's expansion rather than a higher ceiling.
constexpr unsigned int MAX_ITER = 1000;
// Relative convergence tolerance for those loops.
constexpr double EPS = 1e-12;
// Floor that keeps Lentz's method from dividing by an exact zero.
constexpr double FPMIN = 1e-30;

#endif // CONFIG_H
