// pybind11 bindings for /src/math/utils.cpp

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <limits>

#ifdef FASTDIST_ENABLE_CUDA
#include "cuda/utils.cuh"
#endif

#include "fastdist/math/utils.h"

#include "fastdist/wrappers/utils_wrapper.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

void bind_utils(py::module_ &m) {
    m.def("chebyshev_bound", &fastdist::math::chebyshev_bound, py::arg("variance"), py::arg("k"),
          R"pbdoc(Compute Chebyshev bound on tail probability)pbdoc");

    m.def("bayes_rule", &fastdist::math::bayes_rule, py::arg("p_B_given_A"), py::arg("p_A"), py::arg("p_B"),
          R"pbdoc(Apply Bayes' rule to compute posterior probability)pbdoc");

    m.def("law_of_total_probability",
          py::overload_cast<const std::vector<double> &, const std::vector<double> &>(
                  &fastdist::math::law_of_total_probability),
          py::arg("probs_B_given_A"), py::arg("probs_A"),
          R"pbdoc(Compute probability using law of total probability)pbdoc");

    m.def("sigmoid", &fastdist::math::sigmoid, py::arg("x"), R"pbdoc(Compute sigmoid function)pbdoc");

    m.def("logit", &fastdist::math::logit, py::arg("p"), R"pbdoc(Compute logit (inverse sigmoid) function)pbdoc");

    m.def("euclidean_distance", &fastdist::math::euclidean_distance, py::arg("x"), py::arg("y"),
          R"pbdoc(Compute Euclidean distance between two vectors)pbdoc");

    m.def("manhattan_distance", &fastdist::math::manhattan_distance, py::arg("x"), py::arg("y"),
          R"pbdoc(Compute Manhattan (L1) distance between two vectors)pbdoc");

    m.def("cosine_similarity", &fastdist::math::cosine_similarity, py::arg("x"), py::arg("y"),
          R"pbdoc(Compute cosine similarity between two vectors)pbdoc");

    m.def("coefficient_of_variation", &fastdist::math::coefficient_of_variation, py::arg("mean"), py::arg("stddev"),
          R"pbdoc(Compute coefficient of variation)pbdoc");

    m.def("covariance", &fastdist::math::covariance, py::arg("mean_x"), py::arg("mean_y"), py::arg("E_xy"),
          R"pbdoc(Compute covariance given means and expectation of product)pbdoc");

    m.def("choose", &fastdist::math::choose, py::arg("n"), py::arg("k"),
          R"pbdoc(Compute binomial coefficient n choose k)pbdoc");

    m.def("permutation", &fastdist::math::permutation, py::arg("n"), py::arg("k"),
          R"pbdoc(Compute number of permutations of k items from n)pbdoc");

    m.def("factorial", &fastdist::math::factorial, py::arg("n"), R"pbdoc(Compute factorial of integer n)pbdoc");

    m.def("gamma", &fastdist::math::gamma, py::arg("x"), R"pbdoc(Compute Gamma function)pbdoc");

    m.def("log_gamma", &fastdist::math::log_gamma, py::arg("x"), R"pbdoc(Compute logarithm of Gamma function)pbdoc");

    m.def("binomial", &fastdist::math::binomial, py::arg("n"), py::arg("a"), py::arg("b"),
          R"pbdoc(Compute binomial probability term)pbdoc");

    // Batch Functions
    m.def("sigmoid_cpu", &fastdist::math::sigmoid_cpu_wrapper, py::arg("x"),
          R"pbdoc(Batch compute sigmoid on CPU)pbdoc");

    m.def("logit_cpu", &fastdist::math::logit_cpu_wrapper, py::arg("p"), R"pbdoc(Batch compute logit on
        CPU)pbdoc");

#ifdef FASTDIST_ENABLE_CUDA
    // CUDA Functions
    m.def("sigmoid_cuda", &fastdist::math::sigmoid_cuda_wrapper, py::arg("x"),
          R"pbdoc(Batch compute sigmoid using CUDA (GPU))pbdoc");
    m.def("logit_cuda", &fastdist::math::logit_cuda_wrapper, py::arg("p"),
          R"pbdoc(Batch compute logit using CUDA (GPU))pbdoc");
    m.def("euclidean_distance_cuda", &fastdist::math::euclidean_distance_cuda_wrapper, py::arg("x"), py::arg("y"),
          R"pbdoc(Batch compute euclidean distance using CUDA (GPU))pbdoc");
    m.def("manhattan_distance_cuda", &fastdist::math::manhattan_distance_cuda_wrapper, py::arg("x"), py::arg("y"),
          R"pbdoc(Batch compute manhattan distance using CUDA (GPU))pbdoc");
    m.def("cosine_similarity_cuda", &fastdist::math::cosine_similarity_cuda_wrapper, py::arg("x"), py::arg("y"),
          R"pbdoc(Batch compute manhattan distance using CUDA (GPU))pbdoc");
#endif
}
