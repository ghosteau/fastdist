// The shared pseudo-random engine. See include/probforge/math/rng.h.
#include <cstdint>
#include <probforge/math/rng.h>
#include <random>

namespace probforge::math {

    std::mt19937& rng() {
        // Function-local so it is created on first use. At namespace scope every
        // thread touching the library would construct one, paying a
        // std::random_device read even if it never samples.
        thread_local std::mt19937 engine{std::random_device{}()};
        return engine;
    }

    void seed_rng(const std::uint64_t value) {
        // mt19937::seed(uint32_t) derives all 624 state words from one by a simple
        // recurrence; seed_seq mixes properly and uses all 64 bits.
        std::seed_seq sequence{static_cast<std::uint32_t>(value & 0xFFFFFFFFu),
                               static_cast<std::uint32_t>(value >> 32)};
        rng().seed(sequence);
    }

    void seed_rng_from_entropy() { rng().seed(std::random_device{}()); }

} // namespace probforge::math
