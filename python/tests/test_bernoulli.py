import numpy as np
import pytest

from fastdist.distributions.bernoulli import Bernoulli


class TestBernoulli:
    # ------------------
    # Initialization
    # ------------------
    @pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 1.0])
    def test_init_valid_p(self, p):
        b = Bernoulli(p)
        assert b.p == float(p)

    @pytest.mark.parametrize("p", [-0.1, 1.1])
    def test_init_invalid_p_value(self, p):
        with pytest.raises(ValueError, match=r"p must be in the interval \[0, 1\]"):
            Bernoulli(p)

    @pytest.mark.parametrize("p", ["0.5", None, object()])
    def test_init_invalid_p_type(self, p):
        with pytest.raises(TypeError):
            Bernoulli(p)

    # ------------------
    # Property setter
    # ------------------
    def test_p_setter_valid(self):
        b = Bernoulli(0.2)
        b.p = 0.7
        assert b.p == 0.7

    def test_p_setter_invalid(self):
        b = Bernoulli(0.2)
        with pytest.raises(ValueError):
            b.p = -0.1

    # ------------------
    # Representation
    # ------------------
    def test_repr(self):
        b = Bernoulli(0.3)
        assert repr(b) == "Bernoulli(p=0.3)"

    # ------------------
    # Scalar classmethods
    # ------------------
    @pytest.mark.parametrize("k, p", [(0, 0.3), (1, 0.7)])
    def test_pmf_scalar_valid(self, k, p):
        val = Bernoulli.pmf_scalar(k, p)
        assert isinstance(val, float)

    @pytest.mark.parametrize("method_name, args", [
        ("pmf_scalar", (2, -0.1)),
        ("cdf_scalar", (0, 1.5)),
        ("mgf_scalar", (1.0, -0.1)),
        ("cgf_scalar", (1.0, 2.0)),
    ])
    def test_classmethods_reject_invalid_p(self, method_name, args):
        method = getattr(Bernoulli, method_name)
        with pytest.raises(ValueError, match=r"p must be in the interval \[0, 1\]"):
            method(*args)

    # ------------------
    # Instance scalar methods
    # ------------------
    def test_mean_variance_stddev(self):
        b = Bernoulli(0.25)

        assert b.mean() == pytest.approx(0.25)
        assert b.variance() == pytest.approx(0.25 * 0.75)
        assert b.stddev() == pytest.approx(np.sqrt(0.25 * 0.75))

    def test_mean_variance_stddev_override_p(self):
        b = Bernoulli(0.1)

        assert b.mean(0.6) == pytest.approx(0.6)
        assert b.variance(0.6) == pytest.approx(0.6 * 0.4)
        assert b.stddev(0.6) == pytest.approx(np.sqrt(0.6 * 0.4))

    def test_override_invalid_p(self):
        b = Bernoulli(0.5)
        with pytest.raises(ValueError):
            b.mean(-0.2)

    # ------------------
    # PMF / CDF instance methods
    # ------------------
    def test_pmf_scalar_instance(self):
        b = Bernoulli(0.4)
        val = b.pmf(1)
        assert isinstance(val, float)

    def test_cdf_scalar_instance(self):
        b = Bernoulli(0.4)
        val = b.cdf(0)
        assert isinstance(val, float)

    def test_pmf_array_instance(self):
        b = Bernoulli(0.4)
        k = [0, 1, 1, 0]
        vals = b.pmf(k)
        assert isinstance(vals, np.ndarray)
        assert vals.shape == (4,)

    # ------------------
    # MGF / CGF
    # ------------------
    def test_mgf_scalar(self):
        b = Bernoulli(0.3)
        val = b.mgf(1.0)
        assert isinstance(val, float)

    def test_cgf_scalar(self):
        b = Bernoulli(0.3)
        val = b.cgf(1.0)
        assert isinstance(val, float)

    def test_mgf_array(self):
        b = Bernoulli(0.3)
        t = [0.0, 1.0, 2.0]
        vals = b.mgf(t)
        assert isinstance(vals, np.ndarray)
        assert vals.shape == (3,)

    # ------------------
    # Sampling
    # ------------------
    def test_sample(self):
        b = Bernoulli(0.8)
        s = b.sample()
        assert s in (0, 1)

    def test_sample_override_p(self):
        b = Bernoulli(0.8)
        s = b.sample(0.1)
        assert s in (0, 1)

    # ------------------
    # CUDA availability
    # ------------------
    def test_is_cuda_available(self):
        assert isinstance(Bernoulli.is_cuda_available(), bool)

    def test_cuda_methods_or_raise(self):
        b = Bernoulli(0.5)
        k = [0, 1]

        if Bernoulli.is_cuda_available():
            out = b.pmf_cuda(k)
            assert isinstance(out, np.ndarray)
        else:
            with pytest.raises(RuntimeError):
                b.pmf_cuda(k)
