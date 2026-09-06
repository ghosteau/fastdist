# Type stubs for the compiled _fastdist extension.
#
# GENERATED FILE -- do not edit by hand. Regenerate after changing any binding:
#
#     pip install pybind11-stubgen
#     pybind11-stubgen fastdist._fastdist -o stubs
#     cp stubs/fastdist/_fastdist.pyi python/fastdist/_fastdist.pyi
#
# Generated from a CPU-only build (FASTDIST_ENABLE_CUDA=0), which is what the
# published wheels are. A CUDA build additionally exposes *_cuda functions; the
# bindings for those live inside #ifdef FASTDIST_ENABLE_CUDA and so are absent
# here.
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['bayes_rule', 'bernoulli_cdf_cpu', 'bernoulli_cdf_scalar', 'bernoulli_cgf_cpu', 'bernoulli_cgf_scalar', 'bernoulli_mean', 'bernoulli_mgf_cpu', 'bernoulli_mgf_scalar', 'bernoulli_pmf_cpu', 'bernoulli_pmf_scalar', 'bernoulli_sample', 'bernoulli_stddev', 'bernoulli_variance', 'beta_cdf_scalar', 'beta_mean', 'beta_pdf_scalar', 'beta_sample', 'beta_stddev', 'beta_variance', 'binomial', 'binomial_cdf_scalar', 'binomial_cgf_scalar', 'binomial_logpmf_scalar', 'binomial_mean', 'binomial_mgf_scalar', 'binomial_pmf_scalar', 'binomial_sample', 'binomial_stddev', 'binomial_variance', 'chebyshev_bound', 'chi_square_cdf_scalar', 'chi_square_cgf_scalar', 'chi_square_mean', 'chi_square_mgf_scalar', 'chi_square_pdf_scalar', 'chi_square_sample', 'chi_square_stddev', 'chi_square_variance', 'choose', 'coefficient_of_variation', 'cosine_similarity', 'covariance', 'discrete_uniform_cdf_scalar', 'discrete_uniform_cgf_scalar', 'discrete_uniform_mean', 'discrete_uniform_mgf_scalar', 'discrete_uniform_pmf_scalar', 'discrete_uniform_sample', 'discrete_uniform_stddev', 'discrete_uniform_variance', 'euclidean_distance', 'exponential_cdf_cpu', 'exponential_cdf_scalar', 'exponential_cgf_cpu', 'exponential_cgf_scalar', 'exponential_mean', 'exponential_mgf_cpu', 'exponential_mgf_scalar', 'exponential_pdf_cpu', 'exponential_pdf_scalar', 'exponential_sample', 'exponential_stddev', 'exponential_variance', 'factorial', 'gamma', 'gamma_cdf_scalar', 'gamma_cgf_scalar', 'gamma_mean', 'gamma_mgf_scalar', 'gamma_pdf_scalar', 'gamma_sample', 'gamma_stddev', 'gamma_variance', 'geometric_cdf_scalar', 'geometric_cgf_scalar', 'geometric_mean', 'geometric_mgf_scalar', 'geometric_pmf_scalar', 'geometric_sample', 'geometric_stddev', 'geometric_variance', 'law_of_total_probability', 'log_gamma', 'logit', 'logit_cpu', 'manhattan_distance', 'negative_binomial_cdf_scalar', 'negative_binomial_cgf_scalar', 'negative_binomial_mean', 'negative_binomial_mgf_scalar', 'negative_binomial_pmf_scalar', 'negative_binomial_sample', 'negative_binomial_stddev', 'negative_binomial_variance', 'normal_cdf_cpu', 'normal_cdf_scalar', 'normal_cgf_cpu', 'normal_cgf_scalar', 'normal_log_sample', 'normal_logpdf_cpu', 'normal_logpdf_scalar', 'normal_mean', 'normal_mgf_cpu', 'normal_mgf_scalar', 'normal_pdf_cpu', 'normal_pdf_scalar', 'normal_sample', 'normal_stddev', 'normal_variance', 'permutation', 'poisson_cdf_cpu', 'poisson_cdf_scalar', 'poisson_cgf_cpu', 'poisson_cgf_scalar', 'poisson_mean', 'poisson_mgf_cpu', 'poisson_mgf_scalar', 'poisson_pmf_cpu', 'poisson_pmf_scalar', 'poisson_sample', 'poisson_stddev', 'poisson_variance', 'sigmoid', 'sigmoid_cpu', 'uniform_cdf_cpu', 'uniform_cdf_scalar', 'uniform_cgf_cpu', 'uniform_cgf_scalar', 'uniform_mean', 'uniform_mgf_cpu', 'uniform_mgf_scalar', 'uniform_pdf_cpu', 'uniform_pdf_scalar', 'uniform_sample', 'uniform_stddev', 'uniform_variance', 'z_score']
def bayes_rule(p_B_given_A: typing.SupportsFloat, p_A: typing.SupportsFloat, p_B: typing.SupportsFloat) -> float:
    """
    Apply Bayes' rule to compute posterior probability
    """
def bernoulli_cdf_cpu(k: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], p: typing.SupportsFloat, step_size: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute Bernoulli CDF on CPU
    """
def bernoulli_cdf_scalar(k: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute CDF of Bernoulli distribution
    """
def bernoulli_cgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], p: typing.SupportsFloat, step_size: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute Bernoulli CGF on CPU
    """
def bernoulli_cgf_scalar(t: typing.SupportsFloat, p: typing.SupportsFloat) -> float:
    """
    Compute CGF of Bernoulli distribution
    """
def bernoulli_mean(p: typing.SupportsFloat) -> float:
    """
    Compute mean of Bernoulli distribution
    """
def bernoulli_mgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], p: typing.SupportsFloat, step_size: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute Bernoulli MGF on CPU
    """
def bernoulli_mgf_scalar(t: typing.SupportsFloat, p: typing.SupportsFloat) -> float:
    """
    Compute MGF of Bernoulli distribution
    """
def bernoulli_pmf_cpu(k: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], p: typing.SupportsFloat, step_size: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute Bernoulli PDF on CPU
    """
def bernoulli_pmf_scalar(k: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute PMF of Bernoulli distribution
    """
def bernoulli_sample(p: typing.SupportsFloat) -> int:
    """
    Draw random sample from Bernoulli distribution
    """
def bernoulli_stddev(p: typing.SupportsFloat) -> float:
    """
    Compute standard deviation of Bernoulli distribution
    """
def bernoulli_variance(p: typing.SupportsFloat) -> float:
    """
    Compute variance of Bernoulli distribution
    """
def beta_cdf_scalar(x: typing.SupportsFloat, alpha: typing.SupportsFloat, beta: typing.SupportsFloat) -> float:
    """
    Compute CDF of Beta distribution
    """
def beta_mean(alpha: typing.SupportsFloat, beta: typing.SupportsFloat) -> float:
    """
    Compute mean of Beta distribution
    """
def beta_pdf_scalar(x: typing.SupportsFloat, alpha: typing.SupportsFloat, beta: typing.SupportsFloat) -> float:
    """
    Compute PDF of Beta distribution
    """
def beta_sample(alpha: typing.SupportsFloat, beta: typing.SupportsFloat) -> float:
    """
    Draw random sample from Beta distribution
    """
def beta_stddev(alpha: typing.SupportsFloat, beta: typing.SupportsFloat) -> float:
    """
    Compute standard deviation of Beta distribution
    """
def beta_variance(alpha: typing.SupportsFloat, beta: typing.SupportsFloat) -> float:
    """
    Compute variance of Beta distribution
    """
def binomial(n: typing.SupportsInt, a: typing.SupportsFloat, b: typing.SupportsFloat) -> float:
    """
    Compute binomial probability term
    """
def binomial_cdf_scalar(x: typing.SupportsInt, n: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute CDF of Binomial distribution
    """
def binomial_cgf_scalar(t: typing.SupportsFloat, n: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute CGF of Binomial distribution
    """
def binomial_logpmf_scalar(x: typing.SupportsInt, n: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute log PMF of Binomial distribution
    """
def binomial_mean(n: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute mean of Binomial distribution
    """
def binomial_mgf_scalar(t: typing.SupportsFloat, n: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute MGF of Binomial distribution
    """
def binomial_pmf_scalar(x: typing.SupportsInt, n: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute PMF of Binomial distribution
    """
def binomial_sample(n: typing.SupportsInt, p: typing.SupportsFloat) -> int:
    """
    Draw random sample from Binomial distribution
    """
def binomial_stddev(n: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute standard deviation of Binomial distribution
    """
def binomial_variance(n: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute variance of Binomial distribution
    """
def chebyshev_bound(variance: typing.SupportsFloat, k: typing.SupportsFloat) -> float:
    """
    Compute Chebyshev bound on tail probability
    """
def chi_square_cdf_scalar(x: typing.SupportsFloat, k: typing.SupportsFloat) -> float:
    """
    Compute CDF of Chi-square distribution
    """
def chi_square_cgf_scalar(t: typing.SupportsFloat, k: typing.SupportsFloat) -> float:
    """
    Compute CGF of Chi-square distribution
    """
def chi_square_mean(k: typing.SupportsFloat) -> float:
    """
    Compute mean of Chi-square distribution
    """
def chi_square_mgf_scalar(t: typing.SupportsFloat, k: typing.SupportsFloat) -> float:
    """
    Compute MGF of Chi-square distribution
    """
def chi_square_pdf_scalar(x: typing.SupportsFloat, k: typing.SupportsFloat) -> float:
    """
    Compute PDF of Chi-square distribution
    """
def chi_square_sample(k: typing.SupportsFloat) -> float:
    """
    Draw random sample from Chi-square distribution
    """
def chi_square_stddev(k: typing.SupportsFloat) -> float:
    """
    Compute standard deviation of Chi-square distribution
    """
def chi_square_variance(k: typing.SupportsFloat) -> float:
    """
    Compute variance of Chi-square distribution
    """
def choose(n: typing.SupportsInt, k: typing.SupportsInt) -> float:
    """
    Compute binomial coefficient n choose k
    """
def coefficient_of_variation(mean: typing.SupportsFloat, stddev: typing.SupportsFloat) -> float:
    """
    Compute coefficient of variation
    """
def cosine_similarity(x: collections.abc.Sequence[typing.SupportsFloat], y: collections.abc.Sequence[typing.SupportsFloat]) -> float:
    """
    Compute cosine similarity between two vectors
    """
def covariance(mean_x: typing.SupportsFloat, mean_y: typing.SupportsFloat, E_xy: typing.SupportsFloat) -> float:
    """
    Compute covariance given means and expectation of product
    """
def discrete_uniform_cdf_scalar(x: typing.SupportsInt, a: typing.SupportsInt, b: typing.SupportsInt) -> float:
    """
    Compute CDF of discrete uniform distribution
    """
def discrete_uniform_cgf_scalar(t: typing.SupportsFloat, a: typing.SupportsInt, b: typing.SupportsInt) -> float:
    """
    Compute CGF of discrete uniform distribution
    """
def discrete_uniform_mean(a: typing.SupportsInt, b: typing.SupportsInt) -> float:
    """
    Compute mean of discrete uniform distribution
    """
def discrete_uniform_mgf_scalar(t: typing.SupportsFloat, a: typing.SupportsInt, b: typing.SupportsInt) -> float:
    """
    Compute MGF of discrete uniform distribution
    """
def discrete_uniform_pmf_scalar(x: typing.SupportsInt, a: typing.SupportsInt, b: typing.SupportsInt) -> float:
    """
    Compute PMF of discrete uniform distribution
    """
def discrete_uniform_sample(a: typing.SupportsInt, b: typing.SupportsInt) -> int:
    """
    Draw random sample from discrete uniform distribution
    """
def discrete_uniform_stddev(a: typing.SupportsInt, b: typing.SupportsInt) -> float:
    """
    Compute standard deviation of discrete uniform distribution
    """
def discrete_uniform_variance(a: typing.SupportsInt, b: typing.SupportsInt) -> float:
    """
    Compute variance of discrete uniform distribution
    """
def euclidean_distance(x: collections.abc.Sequence[typing.SupportsFloat], y: collections.abc.Sequence[typing.SupportsFloat]) -> float:
    """
    Compute Euclidean distance between two vectors
    """
def exponential_cdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat, step_size: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute exponential CDF on CPU
    """
def exponential_cdf_scalar(x: typing.SupportsFloat, lambda_: typing.SupportsFloat) -> float:
    """
    Compute CDF of exponential distribution
    """
def exponential_cgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat, step_size: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute exponential CGF on CPU
    """
def exponential_cgf_scalar(t: typing.SupportsFloat, lambda_: typing.SupportsFloat) -> float:
    """
    Compute CGF of exponential distribution
    """
def exponential_mean(lambda_: typing.SupportsFloat) -> float:
    """
    Compute mean of exponential distribution
    """
def exponential_mgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat, step_size: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute exponential MGF on CPU
    """
def exponential_mgf_scalar(t: typing.SupportsFloat, lambda_: typing.SupportsFloat) -> float:
    """
    Compute MGF of exponential distribution
    """
def exponential_pdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat, step_size: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute exponential PDF on CPU
    """
def exponential_pdf_scalar(x: typing.SupportsFloat, lambda_: typing.SupportsFloat) -> float:
    """
    Compute PDF of exponential distribution
    """
def exponential_sample(lambda_: typing.SupportsFloat) -> float:
    """
    Draw random sample from exponential distribution
    """
def exponential_stddev(lambda_: typing.SupportsFloat) -> float:
    """
    Compute standard deviation of exponential distribution
    """
def exponential_variance(lambda_: typing.SupportsFloat) -> float:
    """
    Compute variance of exponential distribution
    """
def factorial(n: typing.SupportsInt) -> float:
    """
    Compute factorial of integer n
    """
def gamma(x: typing.SupportsFloat) -> float:
    """
    Compute Gamma function
    """
def gamma_cdf_scalar(x: typing.SupportsFloat, alpha: typing.SupportsFloat, theta: typing.SupportsFloat) -> float:
    """
    Compute CDF of Gamma distribution
    """
def gamma_cgf_scalar(t: typing.SupportsFloat, alpha: typing.SupportsFloat, theta: typing.SupportsFloat) -> float:
    """
    Compute CGF of Gamma distribution
    """
def gamma_mean(alpha: typing.SupportsFloat, theta: typing.SupportsFloat) -> float:
    """
    Compute mean of Gamma distribution
    """
def gamma_mgf_scalar(t: typing.SupportsFloat, alpha: typing.SupportsFloat, theta: typing.SupportsFloat) -> float:
    """
    Compute MGF of Gamma distribution
    """
def gamma_pdf_scalar(x: typing.SupportsFloat, alpha: typing.SupportsFloat, theta: typing.SupportsFloat) -> float:
    """
    Compute PDF of Gamma distribution
    """
def gamma_sample(alpha: typing.SupportsFloat, theta: typing.SupportsFloat) -> float:
    """
    Draw random sample from Gamma distribution
    """
def gamma_stddev(alpha: typing.SupportsFloat, theta: typing.SupportsFloat) -> float:
    """
    Compute standard deviation of Gamma distribution
    """
def gamma_variance(alpha: typing.SupportsFloat, theta: typing.SupportsFloat) -> float:
    """
    Compute variance of Gamma distribution
    """
def geometric_cdf_scalar(k: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute CDF of geometric distribution
    """
def geometric_cgf_scalar(t: typing.SupportsFloat, p: typing.SupportsFloat) -> float:
    """
    Compute CGF of geometric distribution
    """
def geometric_mean(p: typing.SupportsFloat) -> float:
    """
    Compute mean of geometric distribution
    """
def geometric_mgf_scalar(t: typing.SupportsFloat, p: typing.SupportsFloat) -> float:
    """
    Compute MGF of geometric distribution
    """
def geometric_pmf_scalar(k: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute PMF of geometric distribution
    """
def geometric_sample(p: typing.SupportsFloat) -> int:
    """
    Draw random sample from geometric distribution
    """
def geometric_stddev(p: typing.SupportsFloat) -> float:
    """
    Compute standard deviation of geometric distribution
    """
def geometric_variance(p: typing.SupportsFloat) -> float:
    """
    Compute variance of geometric distribution
    """
def law_of_total_probability(probs_B_given_A: collections.abc.Sequence[typing.SupportsFloat], probs_A: collections.abc.Sequence[typing.SupportsFloat]) -> float:
    """
    Compute probability using law of total probability
    """
def log_gamma(x: typing.SupportsFloat) -> float:
    """
    Compute logarithm of Gamma function
    """
def logit(p: typing.SupportsFloat) -> float:
    """
    Compute logit (inverse sigmoid) function
    """
def logit_cpu(p: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute logit on
            CPU
    """
def manhattan_distance(x: collections.abc.Sequence[typing.SupportsFloat], y: collections.abc.Sequence[typing.SupportsFloat]) -> float:
    """
    Compute Manhattan (L1) distance between two vectors
    """
def negative_binomial_cdf_scalar(k: typing.SupportsInt, r: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute CDF of negative binomial distribution
    """
def negative_binomial_cgf_scalar(t: typing.SupportsFloat, r: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute CGF of negative binomial distribution
    """
def negative_binomial_mean(r: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute mean of negative binomial distribution
    """
def negative_binomial_mgf_scalar(t: typing.SupportsFloat, r: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute MGF of negative binomial distribution
    """
def negative_binomial_pmf_scalar(k: typing.SupportsInt, r: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute PMF of negative binomial distribution
    """
def negative_binomial_sample(r: typing.SupportsInt, p: typing.SupportsFloat) -> int:
    """
    Draw random sample from negative binomial distribution
    """
def negative_binomial_stddev(r: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute standard deviation of negative binomial distribution
    """
def negative_binomial_variance(r: typing.SupportsInt, p: typing.SupportsFloat) -> float:
    """
    Compute variance of negative binomial distribution
    """
def normal_cdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat, sigma: typing.SupportsFloat, step_size: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal CDF on CPU
    """
def normal_cdf_scalar(x: typing.SupportsFloat, mu: typing.SupportsFloat, sigma: typing.SupportsFloat) -> float:
    """
    Compute CDF of normal distribution
    """
def normal_cgf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat, sigma: typing.SupportsFloat, step_size: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal CGF on CPU
    """
def normal_cgf_scalar(t: typing.SupportsFloat, mu: typing.SupportsFloat, sigma: typing.SupportsFloat) -> float:
    """
    Compute CGF of normal distribution
    """
def normal_log_sample(mu: typing.SupportsFloat, sigma: typing.SupportsFloat) -> float:
    """
    Draw log-domain random sample from normal distribution
    """
def normal_logpdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat, sigma: typing.SupportsFloat, step_size: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal Log PDF on CPU
    """
def normal_logpdf_scalar(x: typing.SupportsFloat, mu: typing.SupportsFloat, sigma: typing.SupportsFloat) -> float:
    """
    Compute log-PDF of normal distribution
    """
def normal_mean(mu: typing.SupportsFloat) -> float:
    """
    Compute mean of normal distribution
    """
def normal_mgf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat, sigma: typing.SupportsFloat, step_size: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal MGF on CPU
    """
def normal_mgf_scalar(t: typing.SupportsFloat, mu: typing.SupportsFloat, sigma: typing.SupportsFloat) -> float:
    """
    Compute MGF of normal distribution
    """
def normal_pdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat, sigma: typing.SupportsFloat, step_size: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal PDF on CPU
    """
def normal_pdf_scalar(x: typing.SupportsFloat, mu: typing.SupportsFloat, sigma: typing.SupportsFloat) -> float:
    """
    Compute PDF of normal distribution
    """
def normal_sample(mu: typing.SupportsFloat, sigma: typing.SupportsFloat) -> float:
    """
    Draw random sample from normal distribution
    """
def normal_stddev(sigma: typing.SupportsFloat) -> float:
    """
    Compute standard deviation of normal distribution
    """
def normal_variance(sigma: typing.SupportsFloat) -> float:
    """
    Compute variance of normal distribution
    """
def permutation(n: typing.SupportsInt, k: typing.SupportsInt) -> float:
    """
    Compute number of permutations of k items from n
    """
def poisson_cdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat, step_size: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute poisson CDF on CPU
    """
def poisson_cdf_scalar(k: typing.SupportsFloat, lambda_: typing.SupportsFloat) -> float:
    """
    Compute CDF of Poisson distribution
    """
def poisson_cgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat, step_size: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute poisson CGF on CPU
    """
def poisson_cgf_scalar(t: typing.SupportsFloat, lambda_: typing.SupportsFloat) -> float:
    """
    Compute CGF of Poisson distribution
    """
def poisson_mean(lambda_: typing.SupportsFloat) -> float:
    """
    Compute mean of Poisson distribution
    """
def poisson_mgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat, step_size: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute poisson MGF on CPU
    """
def poisson_mgf_scalar(t: typing.SupportsFloat, lambda_: typing.SupportsFloat) -> float:
    """
    Compute MGF of Poisson distribution
    """
def poisson_pmf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat, step_size: typing.SupportsInt) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute poisson PMF on CPU
    """
def poisson_pmf_scalar(k: typing.SupportsFloat, lambda_: typing.SupportsFloat) -> float:
    """
    Compute PMF of Poisson distribution
    """
def poisson_sample(lambda_: typing.SupportsFloat) -> int:
    """
    Draw random sample from Poisson distribution
    """
def poisson_stddev(lambda_: typing.SupportsFloat) -> float:
    """
    Compute standard deviation of Poisson distribution
    """
def poisson_variance(lambda_: typing.SupportsFloat) -> float:
    """
    Compute variance of Poisson distribution
    """
def sigmoid(x: typing.SupportsFloat) -> float:
    """
    Compute sigmoid function
    """
def sigmoid_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute sigmoid on CPU
    """
def uniform_cdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], a: typing.SupportsFloat, b: typing.SupportsFloat, step_size: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute uniform CDF on CPU
    """
def uniform_cdf_scalar(x: typing.SupportsFloat, a: typing.SupportsFloat, b: typing.SupportsFloat) -> float:
    """
    Compute CDF of continuous uniform distribution
    """
def uniform_cgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], a: typing.SupportsFloat, b: typing.SupportsFloat, step_size: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute uniform CGF on CPU
    """
def uniform_cgf_scalar(t: typing.SupportsFloat, a: typing.SupportsFloat, b: typing.SupportsFloat) -> float:
    """
    Compute CGF of continuous uniform distribution
    """
def uniform_mean(a: typing.SupportsFloat, b: typing.SupportsFloat) -> float:
    """
    Compute mean of continuous uniform distribution
    """
def uniform_mgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], a: typing.SupportsFloat, b: typing.SupportsFloat, step_size: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute uniform MGF on CPU
    """
def uniform_mgf_scalar(t: typing.SupportsFloat, a: typing.SupportsFloat, b: typing.SupportsFloat) -> float:
    """
    Compute MGF of continuous uniform distribution
    """
def uniform_pdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], a: typing.SupportsFloat, b: typing.SupportsFloat, step_size: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute uniform PMF on CPU
    """
def uniform_pdf_scalar(x: typing.SupportsFloat, a: typing.SupportsFloat, b: typing.SupportsFloat) -> float:
    """
    Compute PDF of continuous uniform distribution
    """
def uniform_sample(a: typing.SupportsFloat, b: typing.SupportsFloat) -> float:
    """
    Draw random sample from continuous uniform distribution
    """
def uniform_stddev(a: typing.SupportsFloat, b: typing.SupportsFloat) -> float:
    """
    Compute standard deviation of continuous uniform distribution
    """
def uniform_variance(a: typing.SupportsFloat, b: typing.SupportsFloat) -> float:
    """
    Compute variance of continuous uniform distribution
    """
def z_score(x: typing.SupportsFloat, mu: typing.SupportsFloat, sigma: typing.SupportsFloat) -> float:
    """
    Compute z-score for normal distribution
    """
__version__: str = '0.1.0'
