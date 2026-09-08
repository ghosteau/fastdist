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

    void seed_rng(const std::uint64_t value) {
        // mt19937::seed(uint32_t) derives all 624 state words from one word by a
        // simple recurrence. seed_seq exists to do this mixing properly, and it
        // also lets the full 64 bits contribute rather than just the low 32.
        std::seed_seq sequence{static_cast<std::uint32_t>(value & 0xFFFFFFFFu),
                               static_cast<std::uint32_t>(value >> 32)};
        rng().seed(sequence);
    }

    void seed_rng_from_entropy() { rng().seed(std::random_device{}()); }

} // namespace fastdist::math
