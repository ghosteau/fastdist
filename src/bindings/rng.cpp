// pybind11 bindings for /src/math/rng.cpp
#include "fastdist/math/rng.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

void bind_rng(py::module_ &m) {
    m.def("seed", &fastdist::math::seed_rng, py::arg("value"),
          R"pbdoc(Seed the sampling engine so draws are reproducible.

Pins the calling thread's random stream to a fixed sequence: the same seed
replays the same draws on every run.

Two caveats. The seed applies to the calling thread only, so a worker thread
that has not been seeded keeps its own entropy-initialised stream. And while
the underlying Mersenne Twister engine is specified bit-for-bit by the C++
standard, the distribution adaptors built on it are not -- the same seed
therefore yields different samples on Linux, macOS and Windows. A seed makes a
run reproducible on one platform and toolchain, not across all of them.)pbdoc");

    m.def("seed_from_entropy", &fastdist::math::seed_rng_from_entropy,
          R"pbdoc(Return the sampling engine to non-deterministic behaviour.

Draws a fresh seed from the OS entropy source. This is the state every thread
starts in, so it is only needed to undo a previous seed() call.)pbdoc");
}
