// math/normal_wrapper.cpp

#include "probforge/wrappers/normal_wrapper.h"
#include "probforge/math/normal.h"
#include "wrapper_utility.h"

#ifdef PROBFORGE_ENABLE_CUDA
#include "probforge/cuda/normal.cuh"
#endif

namespace probforge::math {
    py::array_t<double> normal_pdf_cpu_wrapper(const py::array_t<double>& x, const double mu, const double sigma,
                                               const double stepSize) {
        return probforge::wrapper::run_cpu_wrapper<double, double>(normal_pdf_batch, x, mu, sigma, stepSize);
    }

    py::array_t<double> normal_logpdf_cpu_wrapper(const py::array_t<double>& x, const double mu, const double sigma,
                                                  const double stepSize) {
        return probforge::wrapper::run_cpu_wrapper<double, double>(normal_logpdf_batch, x, mu, sigma, stepSize);
    }

    py::array_t<double> normal_cdf_cpu_wrapper(const py::array_t<double>& x, const double mu, const double sigma,
                                               const double stepSize) {
        return probforge::wrapper::run_cpu_wrapper<double, double>(normal_cdf_batch, x, mu, sigma, stepSize);
    }

    py::array_t<double> normal_mgf_cpu_wrapper(const py::array_t<double>& t, const double mu, const double sigma,
                                               const double stepSize) {
        return probforge::wrapper::run_cpu_wrapper<double, double>(normal_mgf_batch, t, mu, sigma, stepSize);
    }

    py::array_t<double> normal_cgf_cpu_wrapper(const py::array_t<double>& t, const double mu, const double sigma,
                                               const double stepSize) {
        return probforge::wrapper::run_cpu_wrapper<double, double>(normal_cgf_batch, t, mu, sigma, stepSize);
    }

#ifdef PROBFORGE_ENABLE_CUDA
    py::array_t<double> normal_pdf_cuda_wrapper(const py::array_t<double>& x, const double mu, const double sigma,
                                                const double stepSize) {
        return probforge::wrapper::run_cuda_wrapper<double, double>(probforge::cuda::normal::normal_pdf_dispatcher, x,
                                                                    mu, sigma, stepSize);
    }

    py::array_t<double> normal_logpdf_cuda_wrapper(const py::array_t<double>& x, const double mu, const double sigma,
                                                   const double stepSize) {
        return probforge::wrapper::run_cuda_wrapper<double, double>(probforge::cuda::normal::normal_logpdf_dispatcher,
                                                                    x, mu, sigma, stepSize);
    }

    py::array_t<double> normal_cdf_cuda_wrapper(const py::array_t<double>& x, const double mu, const double sigma,
                                                const double stepSize) {
        return probforge::wrapper::run_cuda_wrapper<double, double>(probforge::cuda::normal::normal_cdf_dispatcher, x,
                                                                    mu, sigma, stepSize);
    }

    py::array_t<double> normal_mgf_cuda_wrapper(const py::array_t<double>& t, const double mu, const double sigma,
                                                const double stepSize) {
        return probforge::wrapper::run_cuda_wrapper<double, double>(probforge::cuda::normal::normal_mgf_dispatcher, t,
                                                                    mu, sigma, stepSize);
    }

    py::array_t<double> normal_cgf_cuda_wrapper(const py::array_t<double>& t, const double mu, const double sigma,
                                                const double stepSize) {
        return probforge::wrapper::run_cuda_wrapper<double, double>(probforge::cuda::normal::normal_cgf_dispatcher, t,
                                                                    mu, sigma, stepSize);
    }
#endif
} // namespace probforge::math
