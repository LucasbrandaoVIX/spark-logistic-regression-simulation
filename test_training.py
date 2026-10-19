"""Regressões numéricas, dados e comunicação real; nenhuma depende do arquivo TLC."""

import contextlib
import io
import multiprocessing as mp
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from train import (
    _partition_metrics,
    _receive,
    _worker,
    loss_and_gradient,
    main,
    prepare_data,
    run_benchmark,
    sigmoid,
    split_partitions,
    tip_labels,
    train_distributed,
    train_sequential,
)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [[-1.0, 0.2], [0.4, 1.1], [1.2, -0.5], [0.7, 0.9], [-0.3, -1.0]]
        )
        self.y = np.array([0.0, 1.0, 1.0, 1.0, 0.0])

    def test_gradient_against_independent_finite_differences(self):
        weights = np.array([0.1, -0.2, 0.3])
        _, actual = loss_and_gradient(self.X, self.y, weights)
        expected = []
        design = np.column_stack((np.ones(len(self.y)), self.X))
        for index in range(len(weights)):
            step = np.zeros_like(weights)
            step[index] = 1e-6
            losses = []
            for candidate in [weights + step, weights - step]:
                logits = design @ candidate
                losses.append(np.mean(np.logaddexp(0, logits) - self.y * logits))
            expected.append((losses[0] - losses[1]) / 2e-6)
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_extreme_logits_have_finite_correct_loss(self):
        with np.errstate(over="raise", invalid="raise"):
            np.testing.assert_allclose(
                sigmoid(np.array([-1000.0, 0.0, 1000.0])), [0, 0.5, 1]
            )
            loss, gradient = loss_and_gradient(
                np.zeros((1, 1)), np.zeros(1), np.array([1000.0, 0.0])
            )
        self.assertEqual(loss, 1000.0)
        np.testing.assert_allclose(gradient, [1, 0])

    def test_weighted_unequal_partitions_match_loss_and_gradient(self):
        weights = np.array([0.1, -0.2, 0.3])
        loss, expected = loss_and_gradient(self.X, self.y, weights)
        local = [
            _partition_metrics(x, y, weights)
            for x, y in split_partitions(self.X, self.y, 3)
        ]
        n = sum(part[0] for part in local)
        np.testing.assert_allclose(
            sum(part[1] for part in local) / n, expected, atol=1e-12
        )
        self.assertAlmostEqual(sum(part[2] for part in local) / n, loss)

    def test_history_includes_initial_and_returned_model(self):
        weights, history, _ = train_sequential(self.X, self.y, 4, 0.1)
        self.assertEqual(len(history), 5)
        self.assertAlmostEqual(history[0], np.log(2))
        self.assertAlmostEqual(
            history[-1], loss_and_gradient(self.X, self.y, weights)[0]
        )

    def test_spawn_workers_are_distinct_and_keep_fixed_partitions(self):
        expected, expected_loss, _ = train_sequential(self.X, self.y, 4, 0.1)
        timings = {}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            weights, history, total = train_distributed(
                self.X, self.y, 3, 4, 0.1, demo=True, timings=timings
            )
        np.testing.assert_allclose(weights, expected, atol=1e-12)
        np.testing.assert_allclose(history, expected_loss, atol=1e-12)
        infos = timings["workers"]
        self.assertEqual(len({info["pid"] for info in infos}), 3)
        self.assertNotIn(os.getpid(), [info["pid"] for info in infos])
        self.assertEqual([info["partition"] for info in infos], [0, 1, 2])
        self.assertEqual([info["rows"] for info in infos], [2, 2, 1])
        self.assertIn("REDUCE", output.getvalue())
        self.assertEqual(output.getvalue().count("MAP partição="), 3)
        self.assertAlmostEqual(
            total,
            sum(
                timings[key]
                for key in [
                    "partition_seconds",
                    "startup_seconds",
                    "iteration_seconds",
                    "shutdown_seconds",
                ]
            ),
        )

    @unittest.skipUnless(
        "forkserver" in mp.get_all_start_methods(), "forkserver indisponível"
    )
    def test_forkserver_also_matches_sequential(self):
        expected, loss, _ = train_sequential(self.X, self.y, 3, 0.1)
        actual, actual_loss, _ = train_distributed(
            self.X, self.y, 2, 3, 0.1, start_method="forkserver"
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(actual_loss, loss, atol=1e-12)

    def test_partitions_cover_original_rows_and_share_driver_memory(self):
        parts = split_partitions(self.X, self.y, 3)
        np.testing.assert_array_equal(np.concatenate([x for x, _ in parts]), self.X)
        np.testing.assert_array_equal(np.concatenate([y for _, y in parts]), self.y)
        self.assertTrue(all(np.shares_memory(x, self.X) for x, _ in parts))
        with self.assertRaises(ValueError):
            split_partitions(self.X, self.y, 8)

    def test_invalid_training_inputs_fail_before_process_creation(self):
        for X, y, epochs, rate in [
            (self.X[:0], self.y[:0], 3, 0.1),
            (self.X, self.y[:2], 3, 0.1),
            (self.X, self.y, 0, 0.1),
            (self.X, self.y, 3, np.nan),
            (self.X * np.inf, self.y, 3, 0.1),
        ]:
            with self.subTest(epochs=epochs, rate=rate), self.assertRaises(ValueError):
                train_distributed(X, y, 2, epochs, rate)

    def test_worker_failure_cleans_up_started_processes(self):
        before = {p.pid for p in mp.active_children()}
        with patch("train._receive", side_effect=RuntimeError("falha simulada")):
            with self.assertRaisesRegex(RuntimeError, "falha simulada"):
                train_distributed(self.X, self.y, 2, 1, 0.1)
        self.assertEqual({p.pid for p in mp.active_children()}, before)

    def test_worker_timeout_has_clear_error(self):
        parent, child = mp.get_context("spawn").Pipe()
        try:
            with self.assertRaisesRegex(RuntimeError, "não respondeu"):
                _receive(parent, 0.01)
        finally:
            parent.close()
            child.close()

    def test_worker_reports_calculation_error_to_driver(self):
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_worker, args=(child, 0, self.X, self.y))
        process.start()
        child.close()
        try:
            self.assertEqual(_receive(parent, 10)["rows"], len(self.y))
            parent.send(np.zeros(1))  # Formato incompatível: falha dentro do worker.
            with self.assertRaisesRegex(RuntimeError, "Falha no worker"):
                _receive(parent, 10)
            process.join(10)
            self.assertFalse(process.is_alive())
        finally:
            if process.is_alive():
                process.terminate()
            process.join()
            parent.close()

    def test_benchmark_includes_one_process_control_and_train_baseline(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_benchmark(
                self.X,
                self.X,
                self.y,
                self.y,
                workers=[1, 2],
                epochs=2,
                learning_rate=0.1,
                repeats=2,
                seed=42,
                start_method="spawn",
            )
        self.assertEqual(
            [(r["mode"], r["workers"]) for r in result["results"]],
            [("sequencial", 1), ("processos", 1), ("processos", 2)],
        )
        self.assertTrue(result["equivalence_checked_every_run"])
        for row in result["results"]:
            self.assertEqual(len(row["runs"]), 2)
            self.assertEqual(len(row["loss"]), 3)
        self.assertEqual(result["majority_baseline"]["predicted_class"], 1)
        self.assertEqual(result["majority_baseline"]["accuracy"], 0.6)
        self.assertEqual(
            result["majority_baseline"]["confusion_matrix"], [[0, 2], [0, 3]]
        )
        self.assertNotEqual(
            result["execution_orders"][0], result["execution_orders"][1]
        )


class DataTests(unittest.TestCase):
    def frame(self):
        pickup = pd.date_range("2025-01-10", periods=20, freq="h")
        return pd.DataFrame(
            {
                "tpep_pickup_datetime": pickup,
                "tpep_dropoff_datetime": pickup + pd.Timedelta(minutes=20),
                "trip_distance": np.arange(20) + 1.0,
                "fare_amount": 17.0,
                "passenger_count": 1.0,
                "tip_amount": [3.39, 3.4] * 10,
                "payment_type": 1,
            }
        )

    def prepare(self, frame, **kwargs):
        with patch("pandas.read_parquet", return_value=frame.copy()):
            return prepare_data(
                Path("fixture.parquet"),
                kwargs.pop("sample_size", 0),
                42,
                kwargs.pop("test_size", 0.2),
                **kwargs
            )

    def test_exact_twenty_percent_and_adjacent_cents(self):
        np.testing.assert_array_equal(
            tip_labels(
                np.array([3.39, 3.4, 3.41, 2.98, 4.8]), np.array([17, 17, 17, 14.9, 24])
            ),
            [0, 1, 1, 1, 1],
        )
        self.assertLess(3.4 / 17, 0.2)

    def test_invalid_money_is_rejected(self):
        for tips, fares in [
            ([np.inf], [17]),
            ([1], [0]),
            ([-1], [17]),
            ([1], [np.nan]),
            ([1e308], [17]),
            ([1], [0.001]),
        ]:
            with self.subTest(tips=tips, fares=fares), self.assertRaises(ValueError):
                tip_labels(np.array(tips), np.array(fares))

    def test_filters_are_counted_once_and_dropoff_can_cross_month(self):
        valid = self.frame()
        valid.loc[0, "tpep_pickup_datetime"] = pd.Timestamp("2025-01-31 23:55")
        valid.loc[0, "tpep_dropoff_datetime"] = pd.Timestamp("2025-02-01 00:15")
        bad = self.frame().iloc[:6].copy()
        bad.loc[0, "payment_type"] = 2
        bad.loc[1, "tpep_pickup_datetime"] = pd.Timestamp("2024-12-31")
        bad.loc[2, "tpep_dropoff_datetime"] = pd.NaT
        bad.loc[3, "tip_amount"] = np.inf
        bad.loc[4, "passenger_count"] = np.nan
        bad.loc[5, "trip_distance"] = 0
        info = {}
        self.prepare(pd.concat([valid, bad], ignore_index=True), metadata=info)
        self.assertEqual(info["valid_rows"], 20)
        self.assertEqual(
            info["raw_rows"], info["valid_rows"] + sum(info["discarded"].values())
        )
        self.assertEqual(info["discarded"]["pickup_outside_month"], 1)
        self.assertEqual(info["discarded"]["nonfinite_values"], 2)
        self.assertEqual(info["discarded"]["timestamps"], 1)
        self.assertEqual(info["train_rows"] + info["test_rows"], 20)

    def test_scaler_is_fit_on_training_only_and_split_is_reproducible(self):
        info = {}
        a = self.prepare(self.frame(), metadata=info)
        b = self.prepare(self.frame())
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_allclose(a[0].mean(axis=0), 0, atol=1e-14)
        self.assertNotAlmostEqual(
            info["scaler_mean"][0], self.frame().trip_distance.mean()
        )
        np.testing.assert_array_equal(info["train_class_counts"], [8, 8])

    def test_invalid_splits_and_period_have_clear_errors(self):
        for kwargs in [
            {"sample_size": -1},
            {"sample_size": 2},
            {"test_size": np.nan},
            {"test_size": 0},
            {"test_size": 0.01},
            {"month": "2025-13"},
            {"month": "2025"},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.prepare(self.frame(), **kwargs)
        single_class = self.frame().assign(tip_amount=0)
        with self.assertRaisesRegex(ValueError, "cada classe"):
            self.prepare(single_class)
        with self.assertRaisesRegex(ValueError, "corridas válidas"):
            self.prepare(self.frame().assign(payment_type=2))

    def test_cli_rejects_nonfinite_rate_and_duplicate_workers(self):
        for flags in [
            ["--learning-rate", "nan"],
            ["--workers", "2", "2"],
            ["--seed", "-1"],
        ]:
            with patch("sys.argv", ["train.py"] + flags), contextlib.redirect_stderr(
                io.StringIO()
            ):
                with self.assertRaises(SystemExit) as exc:
                    main()
                self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
