"""Check variable-length video handling with NumPy as the only dependency."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import main_numpy
from reservoirs_numpy import ESNReservoir, QuantumReservoir
from utils_numpy import evaluate, predict, reservoir_features
from video_data import frame_metrics, load_dataset, load_videos, save_videos, validate_pairs


class VariableVideoTests(unittest.TestCase):
    def models(self):
        """Build small reservoirs with different input and decision widths."""
        # Multiple substeps also exercise feature ordering on short videos.
        return [ESNReservoir(dict(input_dim=2, output_dim=3, reservoir_size=4, time_step=2)),
                QuantumReservoir(2, 3, 2, damping=0.2, time_step=2)]

    def videos(self):
        """Create videos whose unequal lengths are deliberately out of order."""
        # Repeated lengths require batching to restore the original video order.
        rng = np.random.default_rng(71)
        return [rng.normal(size=(length, 2)) for length in (4, 1, 4, 2, 1)]

    def test_prediction_lengths_order_and_state_isolation(self):
        """Compare grouped predictions with independent calls for every video."""
        # Batching must neither reorder videos nor carry another video's state.
        X = self.videos()
        for model in self.models():
            actual = predict(model, X, batch_size=2)
            self.assertIsInstance(actual, list)
            self.assertEqual([p.shape for p in actual], [(len(v), 3) for v in X])
            for i, video in enumerate(X):
                np.testing.assert_allclose(actual[i], model(video[None])[0], atol=1e-12)
                np.testing.assert_allclose(actual[i], model(X)[i], atol=1e-12)
                np.testing.assert_allclose(actual[i], predict(model, X, 1)[i], atol=1e-12)
            changed = [video.copy() for video in X]
            changed[2] += 10
            altered = predict(model, changed)
            for i in (0, 1, 3, 4):
                np.testing.assert_allclose(altered[i], actual[i], atol=1e-12)
            self.assertIsNone(model.reservoir_state)

    def test_metrics_weight_real_frames(self):
        """Ensure metrics count each actual frame coordinate without padding."""
        # A large error in one short video must not get the weight of a long video.
        X = [np.zeros((1, 2)), np.zeros((4, 2))]
        y = [np.full((1, 3), 10.0), np.zeros((4, 3))]
        for model in self.models():
            model.w_out.weight[:] = 0
            model.w_out.bias[:] = 0
            self.assertEqual(evaluate(model, X, y), {"loss": 20.0, "mae": 2.0})

    def test_ragged_features_preserve_prediction_order(self):
        """Check cached variable-length features match each video's predictions."""
        # Concatenate only real frames, preserving the original video order.
        X = self.videos()
        for model in self.models():
            features = reservoir_features(model, X, 2)
            self.assertEqual([len(f) for f in features], list(map(len, X)))
            expected = model.readout(np.concatenate(features))
            actual = np.concatenate(predict(model, X, 2))
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_dense_and_list_inputs_agree(self):
        """Retain dense-array outputs while accepting equal-length lists too."""
        # The caller's input form determines the prediction return type.
        X = np.random.default_rng(70).normal(size=(3, 4, 2))
        for model in self.models():
            dense = predict(model, X)
            ragged = predict(model, list(X))
            self.assertEqual(dense.shape, (3, 4, 3))
            np.testing.assert_allclose(dense, np.stack(ragged), atol=1e-12)

    def test_npz_roundtrip_and_legacy_npy(self):
        """Preserve video ordering, lengths, and dtypes in pickle-free archives."""
        # More than ten entries detects accidental lexical sorting of video indices.
        videos = [np.full((i + 1, 2), i, dtype=np.float32) for i in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "videos.npz"
            save_videos(path, videos)
            restored = load_videos(path)
            for actual, expected in zip(restored, videos):
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.float32)
            dense_path = Path(directory) / "dense.npy"
            dense = np.zeros((2, 3, 2))
            np.save(dense_path, dense, allow_pickle=False)
            np.testing.assert_array_equal(load_videos(dense_path), dense)

    def test_invalid_ragged_pairs(self):
        """Reject mismatched video counts, per-video lengths, and vector widths."""
        # Equal total frame counts do not make swapped target lengths valid.
        X = [np.zeros((1, 2)), np.zeros((3, 2))]
        for invalid in ([np.zeros((4, 3))], [np.zeros((3, 3)), np.zeros((1, 3))],
                        [np.zeros((1, 3)), np.zeros((3, 4))],
                        [np.zeros((1, 3)), np.full((3, 3), np.nan)]):
            with self.assertRaises(ValueError):
                validate_pairs(X, invalid)
        for model in self.models():
            with self.assertRaises(ValueError):
                evaluate(model, X, [np.zeros((1, 1)), np.zeros((3, 1))])

    def test_main_loads_ragged_datasets(self):
        """Evaluate file-loaded and supplied variable-length datasets identically."""
        # Temporary paths exercise the same entry point used after training exports.
        X = self.videos()
        y = [np.zeros((len(v), 3)) for v in X]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            esn, quantum = self.models()
            esn.save(directory / "esn.npz")
            quantum.save(directory / "quantum.npz")
            dataset_path = directory / "final_dataset.pkl"
            with dataset_path.open("wb") as output:
                pickle.dump({"inputs": [[list(range(len(v))), v.tolist()] for v in X],
                             "targets": [[list(range(len(v))), v.tolist()] for v in y]}, output)
            with patch.multiple(main_numpy, ESN_PARAMS=directory / "esn.npz",
                                QUANTUM_PARAMS=directory / "quantum.npz",
                                DATASET_PATH=dataset_path,
                                USE_TEST_SPLIT=False), \
                    redirect_stdout(StringIO()):
                predictions, metrics = main_numpy.main()
                direct, direct_metrics = main_numpy.main(X, y)
            self.assertEqual(metrics, direct_metrics)
            for name in predictions:
                for actual, expected in zip(predictions[name], direct[name]):
                    np.testing.assert_allclose(actual, expected)
                self.assertEqual(metrics[name]["mse"], frame_metrics(predictions[name], y)["loss"])

    def test_notebook_pickle_preserves_values_and_frame_alignment(self):
        """Unwrap the notebook format and reject shifted action timestamps."""
        # Include unequal lengths and different input/output vector widths.
        X = self.videos()
        y = [np.full((len(v), 3), i) for i, v in enumerate(X)]
        dataset = {
            "inputs": [[list(range(1, len(v) + 1)), v.tolist()] for v in X],
            "targets": [[list(range(1, len(v) + 1)), v.tolist()] for v in y],
            "input_keys": ["x", "z"],
            "target_keys": ["a", "b", "c"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final_dataset.pkl"
            with path.open("wb") as output:
                pickle.dump(dataset, output)
            actual_X, actual_y = load_dataset(path, dtype=np.float32)
            for actual, expected in zip(actual_X + actual_y, X + y):
                np.testing.assert_array_equal(actual, expected.astype(np.float32))
                self.assertEqual(actual.dtype, np.float32)
            dataset["targets"][1][0][0] += 1
            with path.open("wb") as output:
                pickle.dump(dataset, output)
            with self.assertRaisesRegex(ValueError, "time steps"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
