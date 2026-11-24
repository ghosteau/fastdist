/**
 * @file rng.h
 * @brief The shared pseudo-random engine every *_sample() function draws from.
 */
#ifndef RNG_H
#define RNG_H

#include <cstdint>
#include <random>

namespace probforge::math {
    /**
     * @brief The calling thread's engine.
     *
     * thread_local, so each thread owns an independent stream: concurrent
     * sampling needs no locking and threads never interleave draws. Seeding is
     * per-thread for the same reason -- see seed_rng().
     */
    std::mt19937& rng();

    /**
     * @brief Pins the calling thread's stream to a fixed sequence.
     *
     * Spread across mt19937's full state through std::seed_seq, so seeds one
     * bit apart give unrelated streams from the first draw.
     *
     * @param value Any 64-bit seed.
     *
     * @warning Seeds the calling thread only, and reproduces a run on one
     * toolchain only: the distribution adaptors over mt19937 are unspecified.
     */
    void seed_rng(std::uint64_t value);

    /**
     * @brief Returns the calling thread's stream to non-deterministic behavior.
     *
     * Draws a fresh seed from the OS entropy source. Every thread starts in
     * this state, so this is only needed to undo a previous seed_rng() call.
     */
    void seed_rng_from_entropy();
} // namespace probforge::math

#endif // RNG_H
