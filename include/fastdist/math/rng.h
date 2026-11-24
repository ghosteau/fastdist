/**
 * @file rng.h
 * @brief The shared pseudo-random engine every *_sample() function draws from.
 */
#ifndef RNG_H
#define RNG_H

#include <cstdint>
#include <random>

namespace fastdist::math {
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
     * The same seed reproduces the same draws on every run. The value is spread
     * across mt19937's full 19937-bit state through std::seed_seq rather than
     * handed over as a single word, so seeds differing by one bit give
     * unrelated streams from the first draw.
     *
     * @param value Any 64-bit seed.
     *
     * @warning Seeds the calling thread only; an unseeded thread keeps its own
     * entropy-initialized stream.
     * @warning std::mt19937 is specified bit-for-bit by the standard, but the
     * distribution adaptors layered on it (std::normal_distribution and
     * friends) are not. The same seed therefore gives different samples across
     * libstdc++, libc++ and MSVC: reproducible on one toolchain, not across all.
     */
    void seed_rng(std::uint64_t value);

    /**
     * @brief Returns the calling thread's stream to non-deterministic behavior.
     *
     * Draws a fresh seed from the OS entropy source. Every thread starts in
     * this state, so this is only needed to undo a previous seed_rng() call.
     */
    void seed_rng_from_entropy();
} // namespace fastdist::math

#endif // RNG_H
