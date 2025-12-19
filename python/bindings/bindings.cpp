// CPP file to link all other bindings
#include <pybind11/pybind11.h>


namespace py = pybind11;

void bind_bernoulli(py::module_ &m);
void bind_normal(py::module_ &m);
void bind_poisson(py::module_ &m);
void bind_exponential(py::module_ &m);

PYBIND11_MODULE(fastdist, m) {
    bind_bernoulli(m);
    bind_normal(m);
    bind_poisson(m);
    bind_exponential(m);
}
