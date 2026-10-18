import numpy as np

from franka_dynamics_priors.env import reference, scenario
from franka_dynamics_priors.priors import _poly_features


def test_frozen_scenario_is_deterministic_and_in_range():
    first = scenario(10000, "train")
    assert first == scenario(10000, "train")
    assert first["task"] in {"point_to_point", "single_joint", "multi_joint"}


def test_reference_has_seven_joint_outputs():
    item = scenario(10000, "train")
    q, dq, ddq = reference(item, 0.25, np.zeros(7))
    assert q.shape == dq.shape == ddq.shape == (7,)
    assert np.isfinite(np.r_[q, dq, ddq]).all()


def test_polynomial_feature_layout_matches_expected_monomials():
    x = np.array([[2.0, 3.0]])
    powers = np.array([[1, 0], [0, 1], [1, 1], [2, 0]])
    np.testing.assert_allclose(_poly_features(x, powers), [[2.0, 3.0, 6.0, 4.0]])
