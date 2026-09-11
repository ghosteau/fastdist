from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['bayes_rule', 'bernoulli_cdf_cpu', 'bernoulli_cdf_cuda', 'bernoulli_cdf_scalar', 'bernoulli_cgf_cpu', 'bernoulli_cgf_cuda', 'bernoulli_cgf_scalar', 'bernoulli_mean', 'bernoulli_mgf_cpu', 'bernoulli_mgf_cuda', 'bernoulli_mgf_scalar', 'bernoulli_pmf_cpu', 'bernoulli_pmf_cuda', 'bernoulli_pmf_scalar', 'bernoulli_sample', 'bernoulli_stddev', 'bernoulli_variance', 'beta_cdf_scalar', 'beta_mean', 'beta_pdf_scalar', 'beta_sample', 'beta_stddev', 'beta_variance', 'binomial', 'binomial_cdf_scalar', 'binomial_cgf_scalar', 'binomial_logpmf_scalar', 'binomial_mean', 'binomial_mgf_scalar', 'binomial_pmf_scalar', 'binomial_sample', 'binomial_stddev', 'binomial_variance', 'chebyshev_bound', 'chi_square_cdf_scalar', 'chi_square_cgf_scalar', 'chi_square_mean', 'chi_square_mgf_scalar', 'chi_square_pdf_scalar', 'chi_square_sample', 'chi_square_stddev', 'chi_square_variance', 'choose', 'coefficient_of_variation', 'cosine_similarity', 'cosine_similarity_cuda', 'covariance', 'discrete_uniform_cdf_scalar', 'discrete_uniform_cgf_scalar', 'discrete_uniform_mean', 'discrete_uniform_mgf_scalar', 'discrete_uniform_pmf_scalar', 'discrete_uniform_sample', 'discrete_uniform_stddev', 'discrete_uniform_variance', 'euclidean_distance', 'euclidean_distance_cuda', 'exponential_cdf_cpu', 'exponential_cdf_cuda', 'exponential_cdf_scalar', 'exponential_cgf_cpu', 'exponential_cgf_cuda', 'exponential_cgf_scalar', 'exponential_mean', 'exponential_mgf_cpu', 'exponential_mgf_cuda', 'exponential_mgf_scalar', 'exponential_pdf_cpu', 'exponential_pdf_cuda', 'exponential_pdf_scalar', 'exponential_sample', 'exponential_stddev', 'exponential_variance', 'factorial', 'gamma', 'gamma_cdf_scalar', 'gamma_cgf_scalar', 'gamma_mean', 'gamma_mgf_scalar', 'gamma_pdf_scalar', 'gamma_sample', 'gamma_stddev', 'gamma_variance', 'geometric_cdf_scalar', 'geometric_cgf_scalar', 'geometric_mean', 'geometric_mgf_scalar', 'geometric_pmf_scalar', 'geometric_sample', 'geometric_stddev', 'geometric_variance', 'law_of_total_probability', 'log_gamma', 'logit', 'logit_cpu', 'logit_cuda', 'manhattan_distance', 'manhattan_distance_cuda', 'negative_binomial_cdf_scalar', 'negative_binomial_cgf_scalar', 'negative_binomial_mean', 'negative_binomial_mgf_scalar', 'negative_binomial_pmf_scalar', 'negative_binomial_sample', 'negative_binomial_stddev', 'negative_binomial_variance', 'normal_cdf_cpu', 'normal_cdf_cuda', 'normal_cdf_scalar', 'normal_cgf_cpu', 'normal_cgf_cuda', 'normal_cgf_scalar', 'normal_log_sample', 'normal_logpdf_cpu', 'normal_logpdf_cuda', 'normal_logpdf_scalar', 'normal_mean', 'normal_mgf_cpu', 'normal_mgf_cuda', 'normal_mgf_scalar', 'normal_pdf_cpu', 'normal_pdf_cuda', 'normal_pdf_scalar', 'normal_sample', 'normal_stddev', 'normal_variance', 'permutation', 'poisson_cdf_cpu', 'poisson_cdf_cuda', 'poisson_cdf_scalar', 'poisson_cgf_cpu', 'poisson_cgf_cuda', 'poisson_cgf_scalar', 'poisson_mean', 'poisson_mgf_cpu', 'poisson_mgf_cuda', 'poisson_mgf_scalar', 'poisson_pmf_cpu', 'poisson_pmf_cuda', 'poisson_pmf_scalar', 'poisson_sample', 'poisson_stddev', 'poisson_variance', 'seed', 'seed_from_entropy', 'sigmoid', 'sigmoid_cpu', 'sigmoid_cuda', 'uniform_cdf_cpu', 'uniform_cdf_cuda', 'uniform_cdf_scalar', 'uniform_cgf_cpu', 'uniform_cgf_cuda', 'uniform_cgf_scalar', 'uniform_mean', 'uniform_mgf_cpu', 'uniform_mgf_cuda', 'uniform_mgf_scalar', 'uniform_pdf_cpu', 'uniform_pdf_cuda', 'uniform_pdf_scalar', 'uniform_sample', 'uniform_stddev', 'uniform_variance', 'z_score']
def bayes_rule(p_B_given_A: typing.SupportsFloat | typing.SupportsIndex, p_A: typing.SupportsFloat | typing.SupportsIndex, p_B: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Apply Bayes' rule to compute posterior probability
    """
def bernoulli_cdf_cpu(k: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], p: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute Bernoulli CDF on CPU
    """
def bernoulli_cdf_cuda(k: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], p: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute bernoulli PDF using CUDA (GPU)
    """
def bernoulli_cdf_scalar(k: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CDF of Bernoulli distribution
    """
def bernoulli_cgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], p: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute Bernoulli CGF on CPU
    """
def bernoulli_cgf_cuda(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], p: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute bernoulli PDF using CUDA (GPU)
    """
def bernoulli_cgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CGF of Bernoulli distribution
    """
def bernoulli_mean(p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute mean of Bernoulli distribution
    """
def bernoulli_mgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], p: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute Bernoulli MGF on CPU
    """
def bernoulli_mgf_cuda(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], p: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute bernoulli PDF using CUDA (GPU)
    """
def bernoulli_mgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute MGF of Bernoulli distribution
    """
def bernoulli_pmf_cpu(k: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], p: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute Bernoulli PDF on CPU
    """
def bernoulli_pmf_cuda(k: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], p: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute bernoulli PDF using CUDA (GPU)
    """
def bernoulli_pmf_scalar(k: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute PMF of Bernoulli distribution
    """
def bernoulli_sample(p: typing.SupportsFloat | typing.SupportsIndex) -> int:
    """
    Draw random sample from Bernoulli distribution
    """
def bernoulli_stddev(p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute standard deviation of Bernoulli distribution
    """
def bernoulli_variance(p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute variance of Bernoulli distribution
    """
def beta_cdf_scalar(x: typing.SupportsFloat | typing.SupportsIndex, alpha: typing.SupportsFloat | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CDF of Beta distribution
    """
def beta_mean(alpha: typing.SupportsFloat | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute mean of Beta distribution
    """
def beta_pdf_scalar(x: typing.SupportsFloat | typing.SupportsIndex, alpha: typing.SupportsFloat | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute PDF of Beta distribution
    """
def beta_sample(alpha: typing.SupportsFloat | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Draw random sample from Beta distribution
    """
def beta_stddev(alpha: typing.SupportsFloat | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute standard deviation of Beta distribution
    """
def beta_variance(alpha: typing.SupportsFloat | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute variance of Beta distribution
    """
def binomial(n: typing.SupportsInt | typing.SupportsIndex, a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute binomial probability term
    """
def binomial_cdf_scalar(x: typing.SupportsInt | typing.SupportsIndex, n: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CDF of Binomial distribution
    """
def binomial_cgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, n: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CGF of Binomial distribution
    """
def binomial_logpmf_scalar(x: typing.SupportsInt | typing.SupportsIndex, n: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute log PMF of Binomial distribution
    """
def binomial_mean(n: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute mean of Binomial distribution
    """
def binomial_mgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, n: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute MGF of Binomial distribution
    """
def binomial_pmf_scalar(x: typing.SupportsInt | typing.SupportsIndex, n: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute PMF of Binomial distribution
    """
def binomial_sample(n: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> int:
    """
    Draw random sample from Binomial distribution
    """
def binomial_stddev(n: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute standard deviation of Binomial distribution
    """
def binomial_variance(n: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute variance of Binomial distribution
    """
def chebyshev_bound(variance: typing.SupportsFloat | typing.SupportsIndex, k: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute Chebyshev bound on tail probability
    """
def chi_square_cdf_scalar(x: typing.SupportsFloat | typing.SupportsIndex, k: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CDF of Chi-square distribution
    """
def chi_square_cgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, k: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CGF of Chi-square distribution
    """
def chi_square_mean(k: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute mean of Chi-square distribution
    """
def chi_square_mgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, k: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute MGF of Chi-square distribution
    """
def chi_square_pdf_scalar(x: typing.SupportsFloat | typing.SupportsIndex, k: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute PDF of Chi-square distribution
    """
def chi_square_sample(k: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Draw random sample from Chi-square distribution
    """
def chi_square_stddev(k: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute standard deviation of Chi-square distribution
    """
def chi_square_variance(k: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute variance of Chi-square distribution
    """
def choose(n: typing.SupportsInt | typing.SupportsIndex, k: typing.SupportsInt | typing.SupportsIndex) -> float:
    """
    Compute binomial coefficient n choose k
    """
def coefficient_of_variation(mean: typing.SupportsFloat | typing.SupportsIndex, stddev: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute coefficient of variation
    """
def cosine_similarity(x: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], y: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> float:
    """
    Compute cosine similarity between two vectors
    """
def cosine_similarity_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], y: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute manhattan distance using CUDA (GPU)
    """
def covariance(mean_x: typing.SupportsFloat | typing.SupportsIndex, mean_y: typing.SupportsFloat | typing.SupportsIndex, E_xy: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute covariance given means and expectation of product
    """
def discrete_uniform_cdf_scalar(x: typing.SupportsInt | typing.SupportsIndex, a: typing.SupportsInt | typing.SupportsIndex, b: typing.SupportsInt | typing.SupportsIndex) -> float:
    """
    Compute CDF of discrete uniform distribution
    """
def discrete_uniform_cgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, a: typing.SupportsInt | typing.SupportsIndex, b: typing.SupportsInt | typing.SupportsIndex) -> float:
    """
    Compute CGF of discrete uniform distribution
    """
def discrete_uniform_mean(a: typing.SupportsInt | typing.SupportsIndex, b: typing.SupportsInt | typing.SupportsIndex) -> float:
    """
    Compute mean of discrete uniform distribution
    """
def discrete_uniform_mgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, a: typing.SupportsInt | typing.SupportsIndex, b: typing.SupportsInt | typing.SupportsIndex) -> float:
    """
    Compute MGF of discrete uniform distribution
    """
def discrete_uniform_pmf_scalar(x: typing.SupportsInt | typing.SupportsIndex, a: typing.SupportsInt | typing.SupportsIndex, b: typing.SupportsInt | typing.SupportsIndex) -> float:
    """
    Compute PMF of discrete uniform distribution
    """
def discrete_uniform_sample(a: typing.SupportsInt | typing.SupportsIndex, b: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    Draw random sample from discrete uniform distribution
    """
def discrete_uniform_stddev(a: typing.SupportsInt | typing.SupportsIndex, b: typing.SupportsInt | typing.SupportsIndex) -> float:
    """
    Compute standard deviation of discrete uniform distribution
    """
def discrete_uniform_variance(a: typing.SupportsInt | typing.SupportsIndex, b: typing.SupportsInt | typing.SupportsIndex) -> float:
    """
    Compute variance of discrete uniform distribution
    """
def euclidean_distance(x: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], y: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> float:
    """
    Compute Euclidean distance between two vectors
    """
def euclidean_distance_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], y: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute euclidean distance using CUDA (GPU)
    """
def exponential_cdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute exponential CDF on CPU
    """
def exponential_cdf_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute exponential PDF using CUDA (GPU)
    """
def exponential_cdf_scalar(x: typing.SupportsFloat | typing.SupportsIndex, lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CDF of exponential distribution
    """
def exponential_cgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute exponential CGF on CPU
    """
def exponential_cgf_cuda(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute exponential PDF using CUDA (GPU)
    """
def exponential_cgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CGF of exponential distribution
    """
def exponential_mean(lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute mean of exponential distribution
    """
def exponential_mgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute exponential MGF on CPU
    """
def exponential_mgf_cuda(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute exponential PDF using CUDA (GPU)
    """
def exponential_mgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute MGF of exponential distribution
    """
def exponential_pdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute exponential PDF on CPU
    """
def exponential_pdf_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute exponential PDF using CUDA (GPU)
    """
def exponential_pdf_scalar(x: typing.SupportsFloat | typing.SupportsIndex, lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute PDF of exponential distribution
    """
def exponential_sample(lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Draw random sample from exponential distribution
    """
def exponential_stddev(lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute standard deviation of exponential distribution
    """
def exponential_variance(lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute variance of exponential distribution
    """
def factorial(n: typing.SupportsInt | typing.SupportsIndex) -> float:
    """
    Compute factorial of integer n
    """
def gamma(x: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute Gamma function
    """
def gamma_cdf_scalar(x: typing.SupportsFloat | typing.SupportsIndex, alpha: typing.SupportsFloat | typing.SupportsIndex, theta: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CDF of Gamma distribution
    """
def gamma_cgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, alpha: typing.SupportsFloat | typing.SupportsIndex, theta: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CGF of Gamma distribution
    """
def gamma_mean(alpha: typing.SupportsFloat | typing.SupportsIndex, theta: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute mean of Gamma distribution
    """
def gamma_mgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, alpha: typing.SupportsFloat | typing.SupportsIndex, theta: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute MGF of Gamma distribution
    """
def gamma_pdf_scalar(x: typing.SupportsFloat | typing.SupportsIndex, alpha: typing.SupportsFloat | typing.SupportsIndex, theta: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute PDF of Gamma distribution
    """
def gamma_sample(alpha: typing.SupportsFloat | typing.SupportsIndex, theta: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Draw random sample from Gamma distribution
    """
def gamma_stddev(alpha: typing.SupportsFloat | typing.SupportsIndex, theta: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute standard deviation of Gamma distribution
    """
def gamma_variance(alpha: typing.SupportsFloat | typing.SupportsIndex, theta: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute variance of Gamma distribution
    """
def geometric_cdf_scalar(k: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CDF of geometric distribution
    """
def geometric_cgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CGF of geometric distribution
    """
def geometric_mean(p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute mean of geometric distribution
    """
def geometric_mgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute MGF of geometric distribution
    """
def geometric_pmf_scalar(k: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute PMF of geometric distribution
    """
def geometric_sample(p: typing.SupportsFloat | typing.SupportsIndex) -> int:
    """
    Draw random sample from geometric distribution
    """
def geometric_stddev(p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute standard deviation of geometric distribution
    """
def geometric_variance(p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute variance of geometric distribution
    """
def law_of_total_probability(probs_B_given_A: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], probs_A: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> float:
    """
    Compute probability using law of total probability
    """
def log_gamma(x: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute logarithm of Gamma function
    """
def logit(p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute logit (inverse sigmoid) function
    """
def logit_cpu(p: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute logit on
            CPU
    """
def logit_cuda(p: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute logit using CUDA (GPU)
    """
def manhattan_distance(x: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], y: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> float:
    """
    Compute Manhattan (L1) distance between two vectors
    """
def manhattan_distance_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], y: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute manhattan distance using CUDA (GPU)
    """
def negative_binomial_cdf_scalar(k: typing.SupportsInt | typing.SupportsIndex, r: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CDF of negative binomial distribution
    """
def negative_binomial_cgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CGF of negative binomial distribution
    """
def negative_binomial_mean(r: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute mean of negative binomial distribution
    """
def negative_binomial_mgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, r: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute MGF of negative binomial distribution
    """
def negative_binomial_pmf_scalar(k: typing.SupportsInt | typing.SupportsIndex, r: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute PMF of negative binomial distribution
    """
def negative_binomial_sample(r: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> int:
    """
    Draw random sample from negative binomial distribution
    """
def negative_binomial_stddev(r: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute standard deviation of negative binomial distribution
    """
def negative_binomial_variance(r: typing.SupportsInt | typing.SupportsIndex, p: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute variance of negative binomial distribution
    """
def normal_cdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal CDF on CPU
    """
def normal_cdf_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal PDF using CUDA (GPU)
    """
def normal_cdf_scalar(x: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CDF of normal distribution
    """
def normal_cgf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal CGF on CPU
    """
def normal_cgf_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal PDF using CUDA (GPU)
    """
def normal_cgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CGF of normal distribution
    """
def normal_log_sample(mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Draw log-domain random sample from normal distribution
    """
def normal_logpdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal Log PDF on CPU
    """
def normal_logpdf_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal PDF using CUDA (GPU)
    """
def normal_logpdf_scalar(x: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute log-PDF of normal distribution
    """
def normal_mean(mu: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute mean of normal distribution
    """
def normal_mgf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal MGF on CPU
    """
def normal_mgf_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal PDF using CUDA (GPU)
    """
def normal_mgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute MGF of normal distribution
    """
def normal_pdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal PDF on CPU
    """
def normal_pdf_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute normal PDF using CUDA (GPU)
    """
def normal_pdf_scalar(x: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute PDF of normal distribution
    """
def normal_sample(mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Draw random sample from normal distribution
    """
def normal_stddev(sigma: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute standard deviation of normal distribution
    """
def normal_variance(sigma: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute variance of normal distribution
    """
def permutation(n: typing.SupportsInt | typing.SupportsIndex, k: typing.SupportsInt | typing.SupportsIndex) -> float:
    """
    Compute number of permutations of k items from n
    """
def poisson_cdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute poisson CDF on CPU
    """
def poisson_cdf_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute poisson CDF using CUDA (GPU)
    """
def poisson_cdf_scalar(k: typing.SupportsFloat | typing.SupportsIndex, lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CDF of Poisson distribution
    """
def poisson_cgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute poisson CGF on CPU
    """
def poisson_cgf_cuda(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute poisson CGF using CUDA (GPU)
    """
def poisson_cgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CGF of Poisson distribution
    """
def poisson_mean(lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute mean of Poisson distribution
    """
def poisson_mgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute poisson MGF on CPU
    """
def poisson_mgf_cuda(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute poisson MGF using CUDA (GPU)
    """
def poisson_mgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute MGF of Poisson distribution
    """
def poisson_pmf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute poisson PMF on CPU
    """
def poisson_pmf_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], lambda_: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsInt | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute poisson PMF using CUDA (GPU)
    """
def poisson_pmf_scalar(k: typing.SupportsFloat | typing.SupportsIndex, lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute PMF of Poisson distribution
    """
def poisson_sample(lambda_: typing.SupportsFloat | typing.SupportsIndex) -> int:
    """
    Draw random sample from Poisson distribution
    """
def poisson_stddev(lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute standard deviation of Poisson distribution
    """
def poisson_variance(lambda_: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute variance of Poisson distribution
    """
def seed(value: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Seed the sampling engine so draws are reproducible.
    
    Pins the calling thread's random stream to a fixed sequence: the same seed
    replays the same draws on every run.
    
    Two caveats. The seed applies to the calling thread only, so a worker thread
    that has not been seeded keeps its own entropy-initialised stream. And while
    the underlying Mersenne Twister engine is specified bit-for-bit by the C++
    standard, the distribution adaptors built on it are not -- the same seed
    therefore yields different samples on Linux, macOS and Windows. A seed makes a
    run reproducible on one platform and toolchain, not across all of them.
    """
def seed_from_entropy() -> None:
    """
    Return the sampling engine to non-deterministic behaviour.
    
    Draws a fresh seed from the OS entropy source. This is the state every thread
    starts in, so it is only needed to undo a previous seed() call.
    """
def sigmoid(x: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute sigmoid function
    """
def sigmoid_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute sigmoid on CPU
    """
def sigmoid_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute sigmoid using CUDA (GPU)
    """
def uniform_cdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute uniform CDF on CPU
    """
def uniform_cdf_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute uniform CDF using CUDA (GPU)
    """
def uniform_cdf_scalar(x: typing.SupportsFloat | typing.SupportsIndex, a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CDF of continuous uniform distribution
    """
def uniform_cgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute uniform CGF on CPU
    """
def uniform_cgf_cuda(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute uniform CGF using CUDA (GPU)
    """
def uniform_cgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute CGF of continuous uniform distribution
    """
def uniform_mean(a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute mean of continuous uniform distribution
    """
def uniform_mgf_cpu(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute uniform MGF on CPU
    """
def uniform_mgf_cuda(t: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute uniform MGF using CUDA (GPU)
    """
def uniform_mgf_scalar(t: typing.SupportsFloat | typing.SupportsIndex, a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute MGF of continuous uniform distribution
    """
def uniform_pdf_cpu(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute uniform PMF on CPU
    """
def uniform_pdf_cuda(x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex, step_size: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]:
    """
    Batch compute uniform PMF using CUDA (GPU)
    """
def uniform_pdf_scalar(x: typing.SupportsFloat | typing.SupportsIndex, a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute PDF of continuous uniform distribution
    """
def uniform_sample(a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Draw random sample from continuous uniform distribution
    """
def uniform_stddev(a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute standard deviation of continuous uniform distribution
    """
def uniform_variance(a: typing.SupportsFloat | typing.SupportsIndex, b: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute variance of continuous uniform distribution
    """
def z_score(x: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex) -> float:
    """
    Compute z-score for normal distribution
    """
__version__: str = '0.1.0'
