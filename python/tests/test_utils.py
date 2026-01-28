from unittest.mock import patch

import pytest

# Imported from utils module
import fastdist.distributions.utils as utils_module
from fastdist.distributions.utils import Utils


@pytest.fixture
def mock_core():
    """
    Patch the internal C++ core for the duration of each test.
    """
    with patch.object(utils_module, "_core", autospec=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Core Utilities Delegation Tests
# ---------------------------------------------------------------------------

def test_chebyshev_bound_delegates_to_core(mock_core):
    mock_core.chebyshev_bound.return_value = 0.75
    result = Utils.chebyshev_bound(variance=1.0, k=2.0)
    mock_core.chebyshev_bound.assert_called_once_with(1.0, 2.0)
    assert result == 0.75


def test_bayes_rule_delegates_to_core(mock_core):
    mock_core.bayes_rule.return_value = 0.3
    result = Utils.bayes_rule(p_B_given_A=0.6, p_A=0.5, p_B=1.0)
    mock_core.bayes_rule.assert_called_once_with(0.6, 0.5, 1.0)
    assert result == 0.3


def test_sigmoid_delegates_to_core(mock_core):
    mock_core.sigmoid.return_value = 0.5
    result = Utils.sigmoid(x=0.0)
    mock_core.sigmoid.assert_called_once_with(0.0)
    assert result == 0.5


def test_logit_delegates_to_core(mock_core):
    mock_core.logit.return_value = 0.0
    result = Utils.logit(p=0.5)
    mock_core.logit.assert_called_once_with(0.5)
    assert result == 0.0


def test_euclidean_distance_delegates_to_core(mock_core):
    mock_core.euclidean_distance.return_value = 5.0

    x = [3.0, 0.0]
    y = [0.0, 4.0]

    result = Utils.euclidean_distance(x, y)

    called_args, _ = mock_core.euclidean_distance.call_args
    import numpy as np
    assert np.array_equal(called_args[0], np.array(x))
    assert np.array_equal(called_args[1], np.array(y))

    assert result == 5.0


def test_manhattan_distance_delegates_to_core(mock_core):
    mock_core.manhattan_distance.return_value = 7.0

    x = [3.0, 0.0]
    y = [0.0, 4.0]

    result = Utils.manhattan_distance(x, y)

    called_args, _ = mock_core.manhattan_distance.call_args
    import numpy as np
    assert np.array_equal(called_args[0], np.array(x))
    assert np.array_equal(called_args[1], np.array(y))

    assert result == 7.0


def test_coefficient_of_variation_delegates_to_core(mock_core):
    mock_core.coefficient_of_variation.return_value = 0.2
    result = Utils.coefficient_of_variation(mean=10.0, stddev=2.0)
    mock_core.coefficient_of_variation.assert_called_once_with(10.0, 2.0)
    assert result == 0.2


def test_covariance_delegates_to_core(mock_core):
    mock_core.covariance.return_value = 2.5
    result = Utils.covariance(mean_x=1.0, mean_y=2.0, E_xy=4.5)
    mock_core.covariance.assert_called_once_with(1.0, 2.0, 4.5)
    assert result == 2.5


# ---------------------------------------------------------------------------
# Combinatorics and Special Functions Delegation Tests
# ---------------------------------------------------------------------------

def test_choose_delegates_to_core(mock_core):
    mock_core.choose.return_value = 10
    result = Utils.choose(n=5, k=2)
    mock_core.choose.assert_called_once_with(5, 2)
    assert result == 10


def test_permutation_delegates_to_core(mock_core):
    mock_core.permutation.return_value = 60
    result = Utils.permutation(n=5, k=3)
    mock_core.permutation.assert_called_once_with(5, 3)
    assert result == 60


def test_factorial_delegates_to_core(mock_core):
    mock_core.factorial.return_value = 120
    result = Utils.factorial(n=5)
    mock_core.factorial.assert_called_once_with(5)
    assert result == 120


def test_gamma_delegates_to_core(mock_core):
    # Gamma(4) = 3! = 6
    mock_core.gamma.return_value = 6.0
    result = Utils.gamma(x=4.0)
    mock_core.gamma.assert_called_once_with(4.0)
    assert result == 6.0


def test_log_gamma_delegates_to_core(mock_core):
    # LogGamma(4) = log(6) ≈ 1.79176
    mock_core.log_gamma.return_value = 1.79176
    result = Utils.log_gamma(x=4.0)
    mock_core.log_gamma.assert_called_once_with(4.0)
    assert result == 1.79176


# ---------------------------------------------------------------------------
# Slots behavior (Inherited from Normal/Uniform test structure)
# ---------------------------------------------------------------------------

def test_utils_class_has_no_slots():
    # Since Utils uses only @classmethod, it doesn't need __slots__.
    # A positive test ensures it does NOT have __slots__ defined (or if it did, it would pass).
    # Since no __slots__ are defined, dynamic attributes should be allowed if instantiated,
    # but the class itself is generally not meant for instantiation.
    # The structure suggests testing the base case: no instance attributes to slot.
    assert not hasattr(Utils, '__slots__')
