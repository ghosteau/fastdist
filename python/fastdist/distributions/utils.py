# python/distributions/utils.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Utils:
    @classmethod
    def _chebyshev_bound(cls, variance, k):
        return _core.chebyshev_bound(variance, k)

    @classmethod
    def _bayes_rule(cls, p_B_given_A, p_A, p_B):
        return _core.bayes_rule(p_B_given_A, p_A, p_B)

    @classmethod
    def _law_of_total_probability(cls, p_A, p_B_given_A):
        return _core.law_of_total_probability(p_A, p_B_given_A)

    @classmethod
    def _sigmoid(cls, x):
        return _core.sigmoid(x)

    @classmethod
    def _logit(cls, p):
        return _core.logit(p)

    @classmethod
    def _euclidean_distance(cls, x, y):
        return _core.euclidean_distance(x, y)

    @classmethod
    def _manhattan_distance(cls, x, y):
        return _core.manhattan_distance(x, y)

    @classmethod
    def _coefficient_of_variation(cls, mean, stddev):
        return _core.coefficient_of_variation(mean, stddev)

    @classmethod
    def _covariance(cls, mean_x, mean_y, E_xy):
        return _core.covariance(mean_x, mean_y, E_xy)

    @classmethod
    def _choose(cls, n, k):
        return _core.choose(n, k)

    @classmethod
    def _permutation(cls, n, k):
        return _core.permutation(n, k)

    @classmethod
    def _factorial(cls, n):
        return _core.factorial(n)

    @classmethod
    def _gamma(cls, x):
        return _core.gamma(x)

    @classmethod
    def _log_gamma(cls, x):
        return _core.log_gamma(x)

    @classmethod
    def _binomial(cls, n, a, b):
        return _core.binomial(n, a, b)
