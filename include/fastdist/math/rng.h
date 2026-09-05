// Header file for the shared pseudo-random number generator
#ifndef RNG_H
#define RNG_H

#include <cstdint>
#include <random>

namespace fastdist::math {
    // The engine every *_sample() function draws from.
    //
    // The engine is thread_local: each thread owns an independent stream, so
    // concurrent sampling needs no locking and threads never interleave draws.
    // The flip side is that seeding is also per-thread -- see seed_rng below.
    std::mt19937& rng();

    // Pins the calling thread's stream to a fixed sequence. The same seed
    // reproduces the same draws on every run, which makes sampling-based tests
    // deterministic and lets callers reproduce a result exactly.
    //
    // Two caveats worth knowing:
    //
    //  1. This seeds the calling thread only. A thread that has not been seeded
    //     keeps its own entropy-initialised stream.
    //
    //  2. std::mt19937 is specified bit-for-bit by the standard, but the
    //     distribution adaptors layered on top of it (std::normal_distribution
    //     and friends) are not. Identical engine state therefore yields
    //     different samples across libstdc++, libc++ and MSVC. A seed makes a
    //     run reproducible on one platform and toolchain, not across all of
    //     them.
    void seed_rng(std::uint32_t value);

    // Returns the calling thread's stream to non-deterministic behaviour by
    // drawing a fresh seed from the OS entropy source. This is the state every
    // thread starts in, so it is only needed to undo a previous seed_rng call.
    void seed_rng_from_entropy();
} // namespace fastdist::math

#endif // RNG_H
