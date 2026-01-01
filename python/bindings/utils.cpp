// pybind11 bindings for /src/math/utils.cpp
#include "fastdist/math/utils.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

void bind_utils(py::module_ &m) {
    m.def("chebyshev_bound", &fastdist::math::chebyshev_bound, py::arg("variance"), py::arg("k"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("bayes_rule", &fastdist::math::bayes_rule, py::arg("p_B_given_A"), py::arg("p_A"), py::arg("p_B"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("law_of_total_probability",
          py::overload_cast<const std::vector<double> &, const std::vector<double> &>(
                  &fastdist::math::law_of_total_probability),
          py::arg("probs_B_given_A"), py::arg("probs_A"),
          R"pbdoc(Manny!
        )pbdoc");


    m.def("sigmoid", &fastdist::math::sigmoid, py::arg("x"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("logit", &fastdist::math::logit, py::arg("p"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("euclidean_distance", &fastdist::math::euclidean_distance, py::arg("x"), py::arg("y"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("manhattan_distance", &fastdist::math::manhattan_distance, py::arg("x"), py::arg("y"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("coefficient_of_variation", &fastdist::math::coefficient_of_variation, py::arg("mean"), py::arg("stddev"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("covariance", &fastdist::math::covariance, py::arg("mean_x"), py::arg("mean_y"), py::arg("E_xy"),
          R"pbdoc(Manny!.
        )pbdoc");

    m.def("choose", &fastdist::math::choose, py::arg("n"), py::arg("k"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("permutation", &fastdist::math::permutation, py::arg("n"), py::arg("k"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("factorial", &fastdist::math::factorial, py::arg("n"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("gamma", &fastdist::math::gamma, py::arg("x"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("log_gamma", &fastdist::math::log_gamma, py::arg("x"),
          R"pbdoc(Manny!
        )pbdoc");

    m.def("binomial", &fastdist::math::binomial, py::arg("n"), py::arg("a"), py::arg("b"),
          R"pbdoc(Manny!
        )pbdoc");
}
