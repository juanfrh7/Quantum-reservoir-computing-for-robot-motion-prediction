"""Run with python -m unittest tests.test_numpy; NumPy is the only extra dependency."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reservoirs_numpy import ESNReservoir, QuantumReservoir, load_model
from utils_numpy import evaluate, make_synthetic_dataset, predict, reservoir_features


class NumpyReservoirTests(unittest.TestCase):
    def models(self):
        """Create seeded ESN and quantum fixtures with different input/output widths."""
        # Unequal widths detect assumptions that decision vectors must match inputs.
        return [
            ESNReservoir(dict(input_dim=2, output_dim=3, reservoir_size=5,
                              time_step=2, leak_rate=0.3)),
            QuantumReservoir(2, 3, 3, interaction="XY", damping=0.2,
                             connectivity="nearest_neighbour", time_step=2),
        ]

    def test_prediction_roundtrip_and_batch_independence(self):
        """Check checkpoint restoration and independence from prediction batch size."""
        X = np.random.default_rng(7).normal(size=(7, 4, 2))
        with tempfile.TemporaryDirectory() as directory:
            for model in self.models():
                with self.subTest(model=model.model_type):
                    # Save actual parameters, including edits that cannot be
                    # reproduced by reconstructing from the random seed.
                    model.w_in.weight *= 0.7
                    model.w_out.bias += 0.2
                    if model.model_type == "quantum":
                        model.J *= 0.6
                        model._build_operators()
                    else:
                        model.w_res *= 0.6
                    expected = predict(model, X, batch_size=3)
                    np.testing.assert_allclose(predict(model, X, 1), expected, atol=1e-12)
                    path = Path(directory) / "model.npz"
                    model.save(path)
                    restored = load_model(path)
                    np.testing.assert_allclose(predict(restored, X, 4), expected, atol=1e-12)
                    self.assertIsNone(restored.reservoir_state)

    def test_esn_update_and_time_step_feature_order(self):
        """Compare ESN substep features with a manually computed leaky recurrence."""
        # Keep each intermediate state to verify the concatenated temporal order.
        model = self.models()[0]
        inputs = np.array([[0.1, -0.3], [0.2, 0.4]])
        state = np.zeros((2, model.reservoir_dim))
        steps = []
        for _ in range(model.time_step):
            state = 0.7 * state + 0.3 * np.tanh(
                state @ model.w_res.T + inputs @ model.w_in.weight.T + model.w_in.bias
            )
            steps.append(state)
        np.testing.assert_allclose(model.features(inputs), np.concatenate(steps, axis=-1))

    def test_quantum_analytic_encoding_and_damping(self):
        """Check one-qubit observable expectations against analytic values."""
        # Disable fixed evolution to isolate RY encoding followed by amplitude damping.
        model = QuantumReservoir(1, 1, 1, hz=0, Jz=0, damping=0.3,
                                 single_observables=("Z", "X", "Y"),
                                 pair_observables=())
        model.w_in.weight[:] = 1
        model.w_in.bias[:] = 0
        angle = 0.7
        features = model.reservoir_step([[angle]])
        np.testing.assert_allclose(features, [[
            0.3 + 0.7 * np.cos(angle), np.sqrt(0.7) * np.sin(angle), 0
        ]], atol=1e-12)

    def test_quantum_state_validity_and_reset(self):
        """Check density-matrix validity, batch guards, and deterministic state resets."""
        # A physical density matrix is Hermitian, positive semidefinite, and has unit trace.
        model = self.models()[1]
        X = np.random.default_rng(4).normal(size=(2, 2))
        first = model(X)
        model(X)
        state = model.get_reservoir_state()
        np.testing.assert_allclose(state, state.conj().swapaxes(-1, -2), atol=1e-12)
        np.testing.assert_allclose(np.trace(state, axis1=1, axis2=2), 1, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(state).min(), -1e-12)
        with self.assertRaises(ValueError):
            model(np.ones((3, 2)))
        model.initialize(2)
        np.testing.assert_allclose(model(X), first, atol=1e-12)

    def test_readout_is_affine_matrix_multiplication(self):
        """Check NumPy readouts use the loaded weights and bias directly."""
        # Arbitrary edited weights detect accidental reinitialization during inference.
        rng = np.random.default_rng(3)
        for model in self.models():
            model.w_out.weight[:] = rng.normal(size=model.w_out.weight.shape)
            model.w_out.bias[:] = rng.normal(size=model.w_out.bias.shape)
            features = rng.normal(size=(4, 7, model.w_out.weight.shape[1]))
            np.testing.assert_allclose(model.readout(features),
                                       features @ model.w_out.weight.T + model.w_out.bias)

    def test_sequence_features_and_metrics(self):
        """Check full-video feature order, prediction shape, and target metrics."""
        # Compare batched extraction with explicit chronological frame updates.
        X = np.random.default_rng(8).normal(size=(3, 4, 2))
        y = np.random.default_rng(18).normal(size=(3, 4, 3))
        for model in self.models():
            features = reservoir_features(model, X)
            model.initialize(len(X))
            for t in range(X.shape[1]):
                np.testing.assert_allclose(features[:, t], model.features(X[:, t]))
            predictions = predict(model, X)
            self.assertEqual(predictions.shape, (3, 4, 3))
            np.testing.assert_allclose(model(X), predictions, atol=1e-12)
            error = predictions - y
            metrics = evaluate(model, X, y)
            self.assertAlmostEqual(metrics["loss"], float(np.mean(error ** 2)))
            self.assertAlmostEqual(metrics["mae"], float(np.mean(np.abs(error))))

    def test_decision_target_alignment(self):
        """Check metrics use supplied decisions rather than deriving targets from inputs."""
        # Zero readouts make the expected error depend only on the external targets.
        X = np.full((2, 3, 2), 999.0)
        y = np.arange(1, 19, dtype=float).reshape(2, 3, 3)
        for model in self.models():
            model.w_out.weight[:] = 0
            model.w_out.bias[:] = 0
            self.assertEqual(evaluate(model, X, y, batch_size=1), {
                "loss": float(np.mean(y ** 2)), "mae": float(np.mean(y))
            })

    def test_same_frame_target_is_not_shifted(self):
        """Check predictions are scored against targets at the same frame index."""
        # Self-generated targets yield zero error; shifting them must change the score.
        X = np.random.default_rng(23).normal(size=(2, 4, 2))
        for model in self.models():
            y = predict(model, X)
            self.assertEqual(evaluate(model, X, y), {"loss": 0.0, "mae": 0.0})
            self.assertGreater(evaluate(model, X, np.roll(y, 1, axis=1))["loss"], 0)

    def test_future_frames_do_not_leak_into_predictions(self):
        """Check future input changes cannot affect earlier decision predictions."""
        X = np.random.default_rng(9).normal(size=(3, 5, 2))
        changed = X.copy()
        changed[:, 2:] += 5
        for model in self.models():
            expected = predict(model, X)
            actual = predict(model, changed)
            np.testing.assert_allclose(actual[:, :2], expected[:, :2], atol=1e-12)
            # The final frame contributes only to its own decision prediction.
            changed_last = X.copy()
            changed_last[:, -1] += 100
            actual_last = predict(model, changed_last)
            np.testing.assert_allclose(actual_last[:, :-1], expected[:, :-1], atol=1e-12)
            self.assertGreater(np.max(np.abs(actual_last[:, -1] - expected[:, -1])), 1e-6)

    def test_video_features_preserve_prediction_order(self):
        """Check dense cached features retain the video and frame axes."""
        # Applying the same readout to cached features must reproduce full inference.
        X = make_synthetic_dataset(10, 5, 2)
        self.assertEqual(X.shape, (10, 5, 2))
        for model in self.models():
            features = reservoir_features(model, X)
            self.assertEqual(features.shape[:2], X.shape[:2])
            np.testing.assert_allclose(model.readout(features), predict(model, X), atol=1e-12)

    def test_invalid_inputs(self):
        """Check malformed frame batches and videos are rejected."""
        # Cover wrong ranks, empty axes, nonfinite values, and mismatched feature widths.
        for model in self.models():
            for invalid in (0, [], [[1]], [[np.nan, 0]]):
                with self.subTest(model=model.model_type, invalid=invalid):
                    with self.assertRaises(ValueError):
                        model(invalid)
            for invalid in (np.empty((0, 2, 2)), np.zeros((3, 0, 2)),
                            np.zeros((3, 2)), np.zeros((3, 4, 1))):
                with self.assertRaises(ValueError):
                    predict(model, invalid)

    def test_invalid_targets_and_single_frame_videos(self):
        """Check single-frame videos work and incompatible targets fail."""
        # Target validation must reject shape broadcasting and nonfinite coordinates.
        X = np.zeros((2, 1, 2))
        for model in self.models():
            self.assertEqual(predict(model, X).shape, (2, 1, 3))
            for invalid in (np.zeros((2, 1, 2)), np.zeros((2, 3)),
                            np.zeros((1, 1, 3)), np.full((2, 1, 3), np.nan)):
                with self.assertRaises(ValueError):
                    evaluate(model, X, invalid)

    def test_main_loads_pickle_decision_targets(self):
        """Check file-loaded and directly supplied decision targets give identical results."""
        # Temporary model and target files keep the test independent of project datasets.
        import main_numpy
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        import pickle

        X = np.random.default_rng(31).normal(size=(2, 3, 2))
        y = np.random.default_rng(32).normal(size=(2, 3, 3))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            esn_path, quantum_path = directory / "esn.npz", directory / "quantum.npz"
            esn, quantum = self.models()
            esn.save(esn_path)
            quantum.save(quantum_path)
            dataset_path = directory / "final_dataset.pkl"
            with dataset_path.open("wb") as output:
                pickle.dump({"inputs": [[list(range(len(v))), v.tolist()] for v in X],
                             "targets": [[list(range(len(v))), v.tolist()] for v in y]}, output)
            with patch.multiple(main_numpy, ESN_PARAMS=esn_path, QUANTUM_PARAMS=quantum_path,
                                DATASET_PATH=dataset_path), redirect_stdout(StringIO()):
                predictions, metrics = main_numpy.main(X)
                direct_predictions, direct_metrics = main_numpy.main(X, y)
            self.assertEqual(metrics, direct_metrics)
            for name, model in (("ESN", esn), ("Quantum", quantum)):
                np.testing.assert_allclose(predictions[name], direct_predictions[name])
                expected = evaluate(model, X, y)
                self.assertEqual(metrics[name], {"mse": expected["loss"], "mae": expected["mae"]})


if __name__ == "__main__":
    unittest.main()
