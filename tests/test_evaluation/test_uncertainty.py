"""Tests for strategy and backtest uncertainty helpers."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.evaluation.uncertainty import (
    BacktestUncertaintyResult,
    PairedUncertaintyResult,
    RealityCheckResult,
    SelectionAdjustmentResult,
    compute_backtest_uncertainty,
    compute_paired_uncertainty,
    compute_reality_check,
    compute_selection_adjustment,
    pick_block_length,
)


def test_pick_block_length_prefers_explicit_then_rebalance_and_horizon_floor():
    returns = np.random.default_rng(0).normal(0.001, 0.02, 80)

    assert pick_block_length(returns, explicit=7, rebalance_step=3, horizon=20) == 7
    assert pick_block_length(returns, rebalance_step=3, horizon=20) == 20
    assert pick_block_length(returns, rebalance_step=21, horizon=5) == 21


def test_compute_backtest_uncertainty_returns_typed_result_with_intervals():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.001, 0.015, 120)

    result = compute_backtest_uncertainty(
        returns,
        periods_per_year=252,
        horizon=21,
        n_boot=80,
        seed=7,
    )

    assert isinstance(result, BacktestUncertaintyResult)
    assert result.bootstrap_block_length >= 21
    assert result.bootstrap_samples == 80
    assert result.n_observations == 120
    assert result.sharpe_ci_lower <= result.sharpe_ci_upper
    assert np.isfinite(result.annualized_return_hac_se)
    assert "sharpe" in result.to_dict()


def test_compute_backtest_uncertainty_accepts_polars_dataframe():
    returns = pl.DataFrame({"daily_return": [0.01, -0.005, 0.003, 0.002, 0.004]})

    result = compute_backtest_uncertainty(returns, n_boot=20, seed=2)

    assert isinstance(result, BacktestUncertaintyResult)
    assert result.n_observations == 5


def test_compute_paired_uncertainty_detects_dominant_challenger():
    rng = np.random.default_rng(2)
    baseline = rng.normal(0.0002, 0.01, 140)
    challenger = baseline + rng.normal(0.001, 0.002, 140)

    result = compute_paired_uncertainty(
        challenger,
        baseline,
        periods_per_year=252,
        horizon=5,
        n_boot=80,
        seed=4,
    )

    assert isinstance(result, PairedUncertaintyResult)
    assert result.sharpe_diff > 0
    assert result.annualized_return_diff > 0
    assert result.probability_challenger_wins > 0.8


def test_compute_paired_uncertainty_requires_aligned_lengths():
    with pytest.raises(ValueError, match="same aligned length"):
        compute_paired_uncertainty(np.ones(10), np.ones(9), n_boot=10)


def test_compute_selection_adjustment_reports_dsr_and_pbo_when_folds_provided():
    rng = np.random.default_rng(3)
    returns_by_variant = {
        "a": rng.normal(0.0010, 0.01, 120),
        "b": rng.normal(0.0005, 0.01, 120),
        "c": rng.normal(0.0000, 0.01, 120),
    }
    fold_perf = {
        "a": np.array([1.0, 0.9, 0.8, 0.7]),
        "b": np.array([0.4, 0.5, 0.6, 0.5]),
        "c": np.array([0.1, 0.2, 0.1, 0.0]),
    }

    result = compute_selection_adjustment(
        returns_by_variant,
        periods_per_year=252,
        pbo_returns_by_fold=fold_perf,
    )

    assert isinstance(result, SelectionAdjustmentResult)
    assert result.n_variants == 3
    assert result.leader in returns_by_variant
    assert result.pbo is not None
    assert result.pbo_n_combinations is not None


def test_compute_reality_check_identifies_best_challenger_name():
    rng = np.random.default_rng(4)
    benchmark = rng.normal(0.0, 0.01, 100)
    challengers = {
        "weak": benchmark + rng.normal(0.0001, 0.002, 100),
        "strong": benchmark + rng.normal(0.001, 0.002, 100),
    }

    result = compute_reality_check(challengers, benchmark, n_bootstrap=40, seed=5)

    assert isinstance(result, RealityCheckResult)
    assert result.best_strategy in challengers
    assert result.n_strategies == 2
