"""Statistics for the classroom study (Sections 5.5, 6.2, 6.3).

Implemented from first principles on numpy/scipy so every number in
``results/tables/`` can be traced to a formula rather than to a library default:

* independent-samples t-test with Cohen's d and a 95% CI on the mean difference
* one-way ANCOVA (post-test ~ group + pre-test) with adjusted means, the group
  effect, its CI, and the homogeneity-of-regression-slopes check
* Shapiro-Wilk normality and Levene homogeneity-of-variance checks
* a priori power for a two-sample t-test via the non-central t distribution

A note on power: the paper reports "statistical power (alpha = 0.05, d = 1.17,
n = 52) = 0.97" *after* observing d = 1.17.  Power computed from the observed
effect is a monotone function of the p-value and adds no information; this
module therefore reports both the *a priori* power for the pre-registered
d = 1.00 and the observed-d value, clearly labelled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
@dataclass
class TTestResult:
    n1: int
    n2: int
    mean1: float
    mean2: float
    mean_diff: float
    pooled_sd: float
    t: float
    df: int
    p_value: float
    cohens_d: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> Dict[str, float]:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def independent_t_test(group1: Sequence[float], group2: Sequence[float],
                       alpha: float = 0.05) -> TTestResult:
    """Student's t-test (equal variances), Cohen's d and the CI on the difference."""
    a, b = np.asarray(group1, float), np.asarray(group2, float)
    n1, n2 = len(a), len(b)
    df = n1 + n2 - 2
    var1, var2 = a.var(ddof=1), b.var(ddof=1)

    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
    pooled_sd = float(np.sqrt(pooled_var))
    se = pooled_sd * np.sqrt(1 / n1 + 1 / n2)

    diff = float(a.mean() - b.mean())
    t = diff / se
    p = float(2 * stats.t.sf(abs(t), df))
    crit = stats.t.ppf(1 - alpha / 2, df)

    return TTestResult(n1=n1, n2=n2, mean1=float(a.mean()), mean2=float(b.mean()),
                       mean_diff=diff, pooled_sd=pooled_sd, t=float(t), df=int(df),
                       p_value=p, cohens_d=float(diff / pooled_sd),
                       ci_low=float(diff - crit * se), ci_high=float(diff + crit * se))


# --------------------------------------------------------------------------- #
def _ols(X: np.ndarray, y: np.ndarray):
    """Least squares with the pieces needed for t/F inference."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    n, k = X.shape
    df_resid = n - k
    sigma2 = float(residuals @ residuals / df_resid)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    ss_resid = float(residuals @ residuals)
    return beta, se, df_resid, ss_resid, sigma2


def ancova(post: Sequence[float], group: Sequence[int], pre: Sequence[float],
           alpha: float = 0.05) -> Dict[str, float]:
    """One-way ANCOVA: post ~ intercept + group + pre.

    ``group`` is 1 for the experimental condition and 0 for the control, so the
    group coefficient *is* the adjusted between-group difference in post-test
    scores at the common (mean) pre-test level.
    """
    post = np.asarray(post, float)
    grp = np.asarray(group, float)
    pre = np.asarray(pre, float)
    n = len(post)

    X_full = np.column_stack([np.ones(n), grp, pre])
    beta, se, df_resid, ss_full, _ = _ols(X_full, post)

    t_group = beta[1] / se[1]
    p_group = float(2 * stats.t.sf(abs(t_group), df_resid))
    crit = stats.t.ppf(1 - alpha / 2, df_resid)

    # F-test for the group factor: compare against the covariate-only model.
    X_reduced = np.column_stack([np.ones(n), pre])
    _, _, df_reduced, ss_reduced, _ = _ols(X_reduced, post)
    f_group = ((ss_reduced - ss_full) / 1) / (ss_full / df_resid)
    p_f = float(stats.f.sf(f_group, 1, df_resid))

    # Homogeneity of regression slopes: is the group x pre-test interaction zero?
    X_inter = np.column_stack([np.ones(n), grp, pre, grp * pre])
    _, _, df_inter, ss_inter, _ = _ols(X_inter, post)
    f_inter = ((ss_full - ss_inter) / 1) / (ss_inter / df_inter)
    p_inter = float(stats.f.sf(f_inter, 1, df_inter))

    pre_mean = float(pre.mean())
    adj_exp = float(beta[0] + beta[1] + beta[2] * pre_mean)
    adj_ctrl = float(beta[0] + beta[2] * pre_mean)

    # Partial eta squared for the group factor.
    partial_eta2 = float((ss_reduced - ss_full) / ((ss_reduced - ss_full) + ss_full))

    return {
        "adjusted_mean_experimental": round(adj_exp, 3),
        "adjusted_mean_control": round(adj_ctrl, 3),
        "adjusted_difference": round(float(beta[1]), 3),
        "se": round(float(se[1]), 3),
        "ci_low": round(float(beta[1] - crit * se[1]), 3),
        "ci_high": round(float(beta[1] + crit * se[1]), 3),
        "t": round(float(t_group), 3),
        "p_value": p_group,
        "F_group": round(float(f_group), 3),
        "p_F": p_f,
        "df_resid": int(df_resid),
        "covariate_slope": round(float(beta[2]), 3),
        "partial_eta_squared": round(partial_eta2, 3),
        "slopes_interaction_F": round(float(f_inter), 3),
        "slopes_interaction_p": round(p_inter, 3),
        "slopes_homogeneous": bool(p_inter > alpha),
    }


# --------------------------------------------------------------------------- #
def assumption_checks(exp_pre, ctrl_pre, exp_gain, ctrl_gain,
                      alpha: float = 0.05) -> list:
    """Shapiro-Wilk and Levene checks, formatted like Table 5 of the paper."""
    rows = []
    for label, sample in [("Shapiro-Wilk (exp. pre-test)", exp_pre),
                          ("Shapiro-Wilk (ctrl. pre-test)", ctrl_pre),
                          ("Shapiro-Wilk (exp. gain)", exp_gain),
                          ("Shapiro-Wilk (ctrl. gain)", ctrl_gain)]:
        w, p = stats.shapiro(np.asarray(sample, float))
        rows.append({"test": label, "statistic": f"W = {w:.3f}", "p_value": round(float(p), 4),
                     "interpretation": ("Normal distribution not rejected" if p > alpha
                                        else "Normality REJECTED")})
    f, p = stats.levene(np.asarray(exp_gain, float), np.asarray(ctrl_gain, float),
                        center="mean")
    n = len(exp_gain) + len(ctrl_gain)
    rows.append({"test": "Levene's test (gain scores)",
                 "statistic": f"F(1,{n - 2}) = {f:.2f}", "p_value": round(float(p), 4),
                 "interpretation": ("Homogeneity of variance satisfied" if p > alpha
                                    else "Homogeneity VIOLATED")})
    return rows


def power_two_sample_t(d: float, n_per_group: int, alpha: float = 0.05) -> float:
    """Power of a two-sided two-sample t-test via the non-central t distribution."""
    df = 2 * n_per_group - 2
    ncp = d * np.sqrt(n_per_group / 2.0)
    crit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))


def required_n_per_group(d: float, power: float = 0.80, alpha: float = 0.05,
                         max_n: int = 2000) -> int:
    """Smallest per-group n reaching the requested power (grid search)."""
    for n in range(4, max_n):
        if power_two_sample_t(d, n, alpha) >= power:
            return n
    return max_n


def correlation(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    r, p = stats.pearsonr(np.asarray(x, float), np.asarray(y, float))
    return {"pearson_r": round(float(r), 4), "p_value": round(float(p), 4),
            "n": len(x)}


def cronbach_alpha(item_matrix: np.ndarray) -> float:
    """Internal consistency of a questionnaire (respondents x items)."""
    item_matrix = np.asarray(item_matrix, float)
    k = item_matrix.shape[1]
    item_vars = item_matrix.var(axis=0, ddof=1).sum()
    total_var = item_matrix.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return 0.0
    return float((k / (k - 1)) * (1 - item_vars / total_var))
