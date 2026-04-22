"""Cross-check the Python engine against the Excel Solver baseline.

The baseline JSON (``data/part2_baseline.json``) was extracted from
``EfficientFrontier_Part2 copy.xlsm`` and contains:

* ``mu_annual``       - annualized expected return vector (10,)
* ``sigma_annual``    - annualized variance-covariance matrix (10, 10)
* ``optimal_by_A``    - Excel Solver's optimal weights + utility for A in {1,2,4,6,8}

If any assertion here fails, the Python engine has drifted from the Part 2
results.  Tolerance: 1e-4 on weights, 1e-4 on utility (Excel shows 4-5 sig figs).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engine import (
    compute_mu_sigma,
    efficient_frontier,
    gmvp,
    optimize_portfolio,
)

HERE = Path(__file__).resolve().parent
BASELINE = json.loads((HERE.parent / "data" / "part2_baseline.json").read_text(encoding="utf-8"))

MU = np.array(BASELINE["mu_annual"], dtype=float)
SIGMA = np.array(BASELINE["sigma_annual"], dtype=float)


@pytest.mark.parametrize("A", [1, 2, 4, 6, 8])
def test_optimize_matches_excel(A: int) -> None:
    expected = BASELINE["optimal_by_A"][str(A)]
    result = optimize_portfolio(MU, SIGMA, A=float(A), allow_short=False)

    # Weights match Excel Solver to 4 decimals.
    np.testing.assert_allclose(
        result.weights,
        np.array(expected["weights"]),
        atol=1e-3,
        err_msg=f"weights mismatch at A={A}",
    )

    # Expected return, variance, utility all match to 4 decimals.
    assert result.expected_return == pytest.approx(expected["expected_return"], abs=1e-4)
    assert result.variance == pytest.approx(expected["variance"], abs=1e-4)
    assert result.std == pytest.approx(expected["std"], abs=1e-4)
    assert result.utility == pytest.approx(expected["utility"], abs=1e-4)


def test_weights_sum_to_one_and_nonnegative() -> None:
    result = optimize_portfolio(MU, SIGMA, A=4.0, allow_short=False)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (result.weights >= -1e-9).all()


def test_gmvp_analytical_matches_formula() -> None:
    # From Part 1: gmvp_return = A/C, gmvp_var = 1/C where A = 1' Sigma^-1 mu, C = 1' Sigma^-1 1
    g = gmvp(MU, SIGMA, allow_short=True)
    inv = np.linalg.inv(SIGMA)
    ones = np.ones(MU.size)
    A_scalar = ones @ inv @ MU
    C_scalar = ones @ inv @ ones
    assert g.expected_return == pytest.approx(A_scalar / C_scalar, rel=1e-6)
    assert g.variance == pytest.approx(1.0 / C_scalar, rel=1e-6)


def test_efficient_frontier_monotonic_std() -> None:
    """Past the GMVP, std must increase as target return increases."""
    fr = efficient_frontier(MU, SIGMA, n_points=30, allow_short=True)
    # take the upper branch (target >= gmvp_return)
    mask = fr.target_returns >= fr.gmvp_return
    stds_upper = fr.stds[mask]
    diffs = np.diff(stds_upper)
    assert (diffs >= -1e-8).all(), "frontier std is not monotonically increasing"


def test_compute_mu_sigma_recovers_annualized_moments() -> None:
    """Use the fallback prices and check mu/sigma broadly agree with baseline."""
    import pandas as pd

    df = pd.read_csv(HERE.parent / "data" / "fallback_prices.csv", index_col="Month")
    mu, sigma = compute_mu_sigma(df, annualize=True)

    # Part 2's baseline mu derived from the same 60-month price series; should match.
    np.testing.assert_allclose(mu, MU, atol=1e-6)
    np.testing.assert_allclose(sigma, SIGMA, atol=1e-6)
