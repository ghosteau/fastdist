// math/normal_wrapper.cpp

#include "../../include/fastdist/wrappers/normal_wrapper.h"
#include "fastdist/math/normal.h"

#ifdef FASTDIST_ENABLE_CUDA
#include "fastdist/cuda/normal.cuh"
#endif

namespace fastdist::math {
    py::array_t<double> normal_pdf_cpu_wrapper(const py::array_t<double>& x, const double mu, const double sigma) {
        const pybind11::buffer_info buf = x.request();
        auto result = py::array_t<double>(buf.size);
        const pybind11::buffer_info result_buf = result.request();

        // Get raw pointers
        const auto* x_ptr = static_cast<double*>(buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        // Call the pure C++ function
        normal_pdf_batch(x_ptr, buf.size, out_ptr, mu, sigma);

        return result;
    }

#ifdef FASTDIST_ENABLE_CUDA
    py::array_t<double> normal_pdf_cuda_wrapper(const py::array_t<double>& x, const double mu, const double sigma) {
        const auto buf = x.request();
        auto result = py::array_t<double>(buf.size);
        const auto result_buf = result.request();

        const auto* x_ptr = static_cast<const double*>(buf.ptr);
        auto* out_ptr = static_cast<double*>(result_buf.ptr);

        fastdist::cuda::normal::normal_pdf_dispatcher(x_ptr, out_ptr, buf.size, mu, sigma);

        return result;
    }
#endif
} // namespace fastdist::math
