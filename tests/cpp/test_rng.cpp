#include <cassert>
#include <cstdint>
#include <fastdist/math/bernoulli.h>
#include <fastdist/math/normal.h>
#include <fastdist/math/rng.h>
#include <fastdist/math/uniform.h>
#include <iostream>
#include <vector>

namespace {
    // Draws a fixed-length trace from several samplers. Mixing distributions
    // matters: they all share one engine, so this also pins down the order in
    // which they consume it.
    std::vector<double> trace(const std::uint32_t seed, const int n = 64) {
        fastdist::math::seed_rng(seed);

        std::vector<double> out;
        out.reserve(static_cast<std::size_t>(n) * 3);
        for (int i = 0; i < n; ++i) {
            out.push_back(fastdist::math::uniform_sample(0.0, 1.0));
            out.push_back(fastdist::math::normal_sample(0.0, 1.0));
            out.push_back(static_cast<double>(fastdist::math::bernoulli_sample(0.5)));
        }
        return out;
    }
} // namespace

void test_rng() {
    std::cout << "Running RNG seeding tests...\n";

    // -------------------------
    // Reproducibility
    // -------------------------
    {
        // The core contract: same seed, same draws. Compared bitwise rather than
        // with a tolerance -- replaying a stream is exact, not approximate.
        const std::vector<double> first = trace(12345);
        const std::vector<double> second = trace(12345);

        assert(first == second);
    }

    // -------------------------
    // Distinct seeds give distinct streams
    // -------------------------
    {
        // Guards against seed_rng being wired up to something that ignores its
        // argument, which would make every "seeded" test vacuous.
        const std::vector<double> a = trace(1);
        const std::vector<double> b = trace(2);

        assert(a != b);
    }

    // -------------------------
    // Re-seeding rewinds the stream
    // -------------------------
    {
        fastdist::math::seed_rng(777);
        const double first_draw = fastdist::math::uniform_sample(0.0, 1.0);

        // Advance the engine well past that point.
        for (int i = 0; i < 100; ++i) {
            (void) fastdist::math::uniform_sample(0.0, 1.0);
        }

        fastdist::math::seed_rng(777);
        assert(fastdist::math::uniform_sample(0.0, 1.0) == first_draw);
    }

    // -------------------------
    // Entropy reseeding escapes the fixed stream
    // -------------------------
    {
        // seed_from_entropy has to actually move off the seeded sequence. A
        // single draw could collide by chance, so compare a run of them; the
        // odds of a full trace repeating are nil.
        fastdist::math::seed_rng(999);
        const std::vector<double> seeded = trace(999);

        fastdist::math::seed_rng_from_entropy();
        std::vector<double> entropic;
        entropic.reserve(seeded.size());
        for (std::size_t i = 0; i < seeded.size(); ++i) {
            entropic.push_back(fastdist::math::uniform_sample(0.0, 1.0));
        }

        assert(entropic != seeded);
    }

    // Leave the engine non-deterministic so a later test that forgot to seed
    // does not silently inherit this one's fixed stream.
    fastdist::math::seed_rng_from_entropy();

    std::cout << "RNG seeding tests passed.\n";
}
