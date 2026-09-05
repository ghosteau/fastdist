// Function definitions for the shared pseudo-random number generator
#include <cstdint>
#include <fastdist/math/rng.h>
#include <random>

namespace fastdist::math {

    std::mt19937& rng() {
        // Function-local rather than namespace-scope so the engine is created on
        // first use. A namespace-scope thread_local would be constructed on every
        // thread that touches the library, including threads that never sample,
        // and would pay a std::random_device read to do it.
        thread_local std::mt19937 engine{std::random_device{}()};
        return engine;
    }

    void seed_rng(const std::uint32_t value) { rng().seed(value); }

    void seed_rng_from_entropy() { rng().seed(std::random_device{}()); }

} // namespace fastdist::math
