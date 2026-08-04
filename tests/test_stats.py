"""Statistical routines checked against known closed-form values."""

import numpy as np
import pytest
from scipy import stats

from src.evaluation.stats_tests import (ancova, cronbach_alpha, independent_t_test,
                                        power_two_sample_t, required_n_per_group)


def test_t_test_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 2, 30), rng.normal(8, 2, 30)
    ours = independent_t_test(a, b)
    ref = stats.ttest_ind(a, b, equal_var=True)
    assert ours.t == pytest.approx(float(ref.statistic), rel=1e-9)
    assert ours.p_value == pytest.approx(float(ref.pvalue), rel=1e-9)


def test_cohens_d_sign_and_scale():
    a = np.array([12.0] * 10)
    b = np.array([10.0] * 10)
    result = independent_t_test(a + np.linspace(-1, 1, 10), b + np.linspace(-1, 1, 10))
    assert result.cohens_d > 0
    assert result.mean_diff == pytest.approx(2.0)


def test_ancova_recovers_a_known_treatment_effect():
    rng = np.random.default_rng(1)
    n = 200
    pre = rng.normal(50, 10, n)
    group = np.array([1] * (n // 2) + [0] * (n // 2))
    post = 0.6 * pre + 10 * group + rng.normal(0, 3, n) + 20
    result = ancova(post, group, pre)
    assert result["adjusted_difference"] == pytest.approx(10, abs=1.0)
    assert result["covariate_slope"] == pytest.approx(0.6, abs=0.1)
    assert result["p_value"] < 1e-6
    assert result["slopes_homogeneous"]


def test_power_increases_with_effect_and_sample_size():
    assert power_two_sample_t(1.0, 26) > power_two_sample_t(0.5, 26)
    assert power_two_sample_t(0.5, 200) > power_two_sample_t(0.5, 26)
    assert 0 < power_two_sample_t(1.17, 26) < 1


def test_required_n_matches_power_curve():
    n = required_n_per_group(0.8, power=0.80)
    assert power_two_sample_t(0.8, n) >= 0.80
    assert power_two_sample_t(0.8, n - 1) < 0.80


def test_cronbach_alpha_range():
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(50, 1))
    consistent = latent + rng.normal(0, 0.2, size=(50, 10))
    noise = rng.normal(size=(50, 10))
    assert cronbach_alpha(consistent) > 0.9
    assert cronbach_alpha(noise) < 0.5
