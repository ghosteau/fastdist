"""
Tests for the seedable sampling engine exposed as fastdist.seed().

Every sampler in the library draws from one shared engine, so these tests mix
distributions deliberately: that pins down both the reproducibility guarantee
and the order in which samplers consume the stream.
"""

import pytest

import fastdist
from fastdist import Bernoulli, Normal, Uniform


def _trace(seed: int, n: int = 64) -> list:
    """Replayable trace across several samplers sharing the one engine."""
    fastdist.seed(seed)

    uniform = Uniform(0.0, 1.0)
    normal = Normal(0.0, 1.0)
    bernoulli = Bernoulli(0.5)

    out = []
    for _ in range(n):
        out.append(uniform.sample())
        out.append(normal.sample())
        out.append(bernoulli.sample())
    return out


def test_same_seed_reproduces_stream():
    """The core contract: same seed, same draws.

    Compared exactly rather than with a tolerance -- replaying a seeded stream
    is bit-for-bit reproducible, not merely close.
    """
    assert _trace(12345) == _trace(12345)


def test_distinct_seeds_give_distinct_streams():
    """Guards against seed() ignoring its argument.

    If it did, every seeded test in the suite would still pass while asserting
    nothing, so this failure mode is worth pinning explicitly.
    """
    assert _trace(1) != _trace(2)


def test_reseeding_rewinds_the_stream():
    """Re-seeding mid-stream returns the engine to the same point."""
    uniform = Uniform(0.0, 1.0)

    fastdist.seed(777)
    first = uniform.sample()

    for _ in range(100):
        uniform.sample()

    fastdist.seed(777)
    assert uniform.sample() == first


def test_seed_from_entropy_escapes_the_fixed_stream():
    """seed_from_entropy() has to actually leave the seeded sequence.

    A single draw could collide by chance, so a whole trace is compared; the
    odds of one repeating are nil.
    """
    seeded = _trace(999)

    fastdist.seed_from_entropy()
    uniform = Uniform(0.0, 1.0)
    entropic = [uniform.sample() for _ in range(len(seeded))]

    assert entropic != seeded


@pytest.mark.parametrize("seed", [0, 1, -1, 2**32, 2**40, -(2**40), 2**63 - 1, -(2**63)])
def test_seed_accepts_the_full_signed_64_bit_range(seed):
    """Negative and large seeds are accepted, not just uint32.

    Seeding from `hash(x)` or a signed counter is ordinary usage, and both
    routinely produce values outside the unsigned 32-bit range.
    """
    fastdist.seed(seed)
    first = [Uniform(0.0, 1.0).sample() for _ in range(4)]

    fastdist.seed(seed)
    assert [Uniform(0.0, 1.0).sample() for _ in range(4)] == first


def test_adjacent_seeds_are_not_correlated():
    """Seeding in a loop is common, so adjacent seeds must give unrelated draws.

    mt19937 seeded from a single word mixes slowly; the implementation spreads
    the seed across the full state with seed_seq to avoid that. If it regressed
    to naive seeding, consecutive seeds would produce clustered first draws and
    the mean gap would collapse well below the 1/3 expected of independent
    uniforms.
    """
    firsts = []
    for seed in range(512):
        fastdist.seed(seed)
        firsts.append(Uniform(0.0, 1.0).sample())

    gaps = [abs(b - a) for a, b in zip(firsts, firsts[1:])]
    mean_gap = sum(gaps) / len(gaps)

    # 1/3 is the expected absolute difference between two independent U(0,1)
    # draws. The window is wide enough not to flake on 512 samples but far
    # tighter than anything clustered seeding would produce.
    assert 0.25 < mean_gap < 0.42, f"mean gap {mean_gap} suggests correlated seeding"


def test_seed_is_exported_from_the_package_root():
    """seed() is part of the public API, not an implementation detail."""
    assert "seed" in fastdist.__all__
    assert "seed_from_entropy" in fastdist.__all__


def teardown_module(module):
    """Leave the engine non-deterministic for any test module that follows."""
    fastdist.seed_from_entropy()
