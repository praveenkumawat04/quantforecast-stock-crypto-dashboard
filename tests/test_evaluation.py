"""Unit tests for forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from utils.evaluation import (
    EvaluationError,
    directional_accuracy,
    evaluate_forecast,
    format_metrics,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


class TestMetricValues:
    def test_known_mae(self) -> None:
        actual = [100.0, 110.0, 120.0]
        predicted = [102.0, 108.0, 126.0]
        assert mean_absolute_error(actual, predicted) == pytest.approx(10 / 3)

    def test_known_rmse(self) -> None:
        actual = [100.0, 110.0, 120.0]
        predicted = [102.0, 108.0, 126.0]
        assert root_mean_squared_error(actual, predicted) == pytest.approx(np.sqrt(44 / 3))

    def test_known_mape(self) -> None:
        actual = [100.0, 110.0, 120.0]
        predicted = [102.0, 108.0, 126.0]
        expected = ((2 / 100) + (2 / 110) + (6 / 120)) / 3 * 100
        assert mean_absolute_percentage_error(actual, predicted) == pytest.approx(expected)

    def test_perfect_forecast(self) -> None:
        values = [10.0, 20.0, 30.0]
        metrics = evaluate_forecast(values, values, model_name="Perfect")
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mape"] == 0.0
        assert metrics["model"] == "Perfect"
        assert metrics["n_samples"] == 3

    def test_directional_accuracy(self) -> None:
        actual = [100.0, 101.0, 102.0, 101.0]
        predicted = [100.0, 102.0, 103.0, 100.0]
        # Directions: actual +,+,- ; predicted +,+,- -> 100%
        assert directional_accuracy(actual, predicted) == pytest.approx(100.0)

        bad = [100.0, 99.0, 98.0, 99.0]
        assert directional_accuracy(actual, bad) == pytest.approx(0.0)


class TestEdgeCases:
    def test_mape_excludes_zero_actuals(self) -> None:
        assert mean_absolute_percentage_error([0.0, 100.0], [10.0, 90.0]) == pytest.approx(10.0)

    def test_mape_all_zero_returns_nan(self) -> None:
        assert np.isnan(mean_absolute_percentage_error([0.0, 0.0], [1.0, 2.0]))

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(EvaluationError, match="Length mismatch"):
            mean_absolute_error([1, 2, 3], [1, 2])

    def test_empty_raises(self) -> None:
        with pytest.raises(EvaluationError):
            mean_absolute_error([], [])

    def test_nan_values_are_filtered(self) -> None:
        actual = [100.0, np.nan, 120.0]
        predicted = [102.0, 105.0, np.nan]
        metrics = evaluate_forecast(actual, predicted)
        assert metrics["n_samples"] == 1
        assert metrics["mae"] == pytest.approx(2.0)

    def test_all_non_finite_raises(self) -> None:
        with pytest.raises(EvaluationError, match="finite"):
            evaluate_forecast([np.nan], [np.nan])

    def test_directional_accuracy_needs_two_points(self) -> None:
        assert np.isnan(directional_accuracy([100.0], [101.0]))

    def test_format_metrics(self) -> None:
        formatted = format_metrics({"mae": 1.234, "rmse": 2.5, "mape": 3.456})
        assert formatted["MAE"] == "1.23"
        assert formatted["RMSE"] == "2.50"
        assert formatted["MAPE"] == "3.46%"

    def test_format_metrics_handles_nan(self) -> None:
        formatted = format_metrics({"mae": float("nan"), "rmse": 1.0, "mape": float("nan")})
        assert formatted["MAE"] == "N/A"
        assert formatted["MAPE"] == "N/A"
