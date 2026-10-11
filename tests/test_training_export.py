"""Check the training-to-NumPy handoff in an environment with training dependencies."""
import contextlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import main as training
from reservoirs import ESNReservoir, QiskitQuantumReservoir, check_readout_optimization
from utils import cached_metrics, make_loader, predict, reservoir_features, train_readout
from video_data import load_videos, split_video_indices


class TrainingExportTests(unittest.TestCase):
    def test_readout_diagnostic_checks_both_reservoirs(self):
        """Verify the diagnostic confirms changed readouts and fixed reservoirs."""
        # Independent decisions and unequal lengths exercise the real fitting path.
        rng = np.random.default_rng(49)
        X = [rng.normal(size=(t, 2)).astype(np.float32) for t in (2, 4, 3)]
        y = [rng.normal(size=(len(v), 3)).astype(np.float32) for v in X]
        models = [ESNReservoir(dict(input_dim=2, output_dim=3, reservoir_size=4)),
                  QiskitQuantumReservoir(2, 3, 2, damping=0.2)]
        for model in models:
            with self.subTest(model=type(model).__name__):
                report = check_readout_optimization(model, X, y, batch_size=2, epochs=40, learning_rate=0.01)
                self.assertTrue(report["passed"])
                self.assertTrue(report["fixed_parameters_unchanged"])
                self.assertGreater(report["weight_change"], 0)
                self.assertGreater(report["bias_change"], 0)
                self.assertLess(report["mse_after"], report["mse_before"])

    def test_readout_diagnostic_detects_missing_fit(self):
        """Reject a fitting function that leaves the random readout untouched."""
        # A negative control proves PASS is more than a requires_grad check.
        model = ESNReservoir(dict(input_dim=2, output_dim=3, reservoir_size=4))
        X, y = np.zeros((2, 3, 2), dtype=np.float32), np.full((2, 3, 3), 10, dtype=np.float32)
        with patch("utils.train_epoch"):
            report = check_readout_optimization(model, X, y)
        self.assertFalse(report["passed"])
        self.assertEqual(report["weight_change"], 0)
        self.assertEqual(report["bias_change"], 0)

    def test_adam_history_and_validation_selection(self):
        """Check genuine Adam updates, fixed reservoirs, and best-epoch restoration."""
        # Constant features and opposing targets make held-out selection observable.
        torch.manual_seed(42)
        model = ESNReservoir(dict(input_dim=2, output_dim=1, reservoir_size=4))
        with torch.no_grad():
            model.w_out.weight.zero_()
            model.w_out.bias.zero_()
        original = {key: value.clone() for key, value in model.state_dict().items()}
        features = torch.ones(20, 4)
        train = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(features, torch.ones(20, 1)), batch_size=5
        )
        validation = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(features, -torch.ones(20, 1)), batch_size=7
        )
        fitted = train_readout(model, train, validation, epochs=8,
                              learning_rate=0.01, verbose=False)
        history = fitted["history"]
        self.assertEqual(history["epoch"], list(range(9)))
        self.assertLess(history["train_loss"][-1], history["train_loss"][0])
        self.assertGreater(history["val_loss"][-1], history["val_loss"][0])
        self.assertEqual(fitted["best_epoch"], 0)
        self.assertEqual(fitted["validation"]["loss"], min(history["val_loss"]))
        self.assertEqual(cached_metrics(model, validation), fitted["validation"])
        for key, value in original.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_train_export_and_evaluate_without_training_dependencies(self):
        """Check training exports reproduce predictions in a NumPy-only interpreter."""
        rng = np.random.default_rng(19)
        X = rng.normal(size=(20, 4, 2)).astype(np.float32)
        y = rng.normal(size=(20, 4, 3)).astype(np.float32)
        self._check_handoff(X, y)

    def test_variable_length_training_export(self):
        """Train and export models on videos with unequal frame counts."""
        # Include one-frame videos and repeated lengths in non-sorted order.
        rng = np.random.default_rng(29)
        lengths = [1, 5, 3, 2, 4] * 4
        X = [rng.normal(size=(t, 2)).astype(np.float32) for t in lengths]
        y = [rng.normal(size=(t, 3)).astype(np.float32) for t in lengths]
        self._check_handoff(X, y)

    def test_notebook_pickle_training_export(self):
        """Load inputs and same-frame decisions from the notebook's pickle format."""
        # Default training must unwrap values rather than use time-step lists as features.
        rng = np.random.default_rng(39)
        lengths = [1, 5, 3, 2, 4] * 4
        X = [rng.normal(size=(t, 2)).astype(np.float32) for t in lengths]
        y = [rng.normal(size=(t, 3)).astype(np.float32) for t in lengths]
        self._check_handoff(X, y, from_file=True)

    def _check_handoff(self, X, y, from_file=False):
        """Verify fitted weights, video splits, and independent NumPy inference."""
        # Reuse the same export checks for dense and variable-length datasets.
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory) / "results"
            dataset_path = Path(directory) / "final_dataset.pkl"
            with dataset_path.open("wb") as output:
                pickle.dump({
                    "inputs": [[list(range(len(v))), v.tolist()] for v in X],
                    "targets": [[list(range(len(v))), v.tolist()] for v in y],
                }, output)
            with patch.multiple(training, RESULTS_DIR=results_dir, BATCH_SIZE=5,
                                ESN_SIZE=4, NUM_QUBITS=2, HZ=0.7, JZ=1.2,
                                TIME_STEP=1, SPECTRAL_RADIUS=0.9,
                                LEAK_RATE=0.2, EPOCHS=8, LEARNING_RATE=0.01, DATASET_PATH=dataset_path), \
                    contextlib.redirect_stdout(io.StringIO()):
                models, metrics = training.main() if from_file else training.main(X, y)

            train_indices, val_indices, test_indices = split_video_indices(len(X), training.SEED)
            X_test, y_test = [X[i] for i in test_indices], [y[i] for i in test_indices]
            all_indices = np.concatenate([train_indices, val_indices, test_indices])
            np.testing.assert_array_equal(np.sort(all_indices), np.arange(len(X)))
            settings = {"num_qubits": 2, "hz": 0.7, "Jz": 1.2, "leak_rate": 0.2,
                        "optimizer": "Adam", "epochs": 8, "learning_rate": 0.01,
                        "adam_betas": list(training.ADAM_BETAS), "adam_eps": training.ADAM_EPS,
                        "batch_size": 5, "seed": training.SEED}
            esn_dir = results_dir / "results_size_4_leak_rate_0.2"
            quantum_dir = results_dir / "results_n_2_hz_0.7_jz_1.2_leak_rate_0.2"
            self.assertEqual(set(results_dir.iterdir()), {esn_dir, quantum_dir})

            expected = {}
            for name, original_name, result_dir in (
                ("ESN", "Classical ESN", esn_dir),
                ("Quantum", "Quantum", quantum_dir),
            ):
                # Training saves only the final benchmark JSON and model parameters.
                self.assertEqual({p.name for p in result_dir.iterdir()}, {"params.npz", "results.json"})
                saved_results = json.loads((result_dir / "results.json").read_text())
                self.assertEqual(saved_results, json.loads(json.dumps(metrics[original_name])))
                for key, value in settings.items():
                    self.assertEqual(saved_results["params"][key], value)
                if name == "ESN":
                    self.assertEqual(saved_results["params"]["reservoir_size"], 4)
                model = models[original_name]
                expected[name] = [video.numpy() for video in predict(model, X_test)]
                with np.load(result_dir / "params.npz", allow_pickle=False) as archive:
                    # Both archives and metrics retain the actual configured settings.
                    metadata = json.loads(archive["metadata"].item())
                    self.assertEqual(metadata["hyperparameters"], settings)
                    for key, value in settings.items():
                        self.assertEqual(archive[key].tolist(), value)
                    config = metadata["model_config"]
                    if name == "ESN":
                        self.assertEqual(config["leak_rate"], settings["leak_rate"])
                    else:
                        for key in ("num_qubits", "hz", "Jz"):
                            self.assertEqual(config[key], settings[key])
                        self.assertEqual(config["damping"], settings["leak_rate"])
                    for key in ("w_in.weight", "w_in.bias", "w_out.weight", "w_out.bias",
                                "w_res" if name == "ESN" else "J"):
                        np.testing.assert_array_equal(archive[key], model.state_dict()[key].numpy())
                # Final metrics must describe the validation-selected exported weights.
                self.assertIn(saved_results["best_epoch"], range(9))
                cached = reservoir_features(
                    model, make_loader([X[i] for i in val_indices], [y[i] for i in val_indices], 5), 5
                )
                self.assertEqual(cached_metrics(model, cached), metrics[original_name]["validation"])

            # An isolated interpreter proves inference never imports Torch/Qiskit.
            script = '''
import importlib.abc
import json
from pathlib import Path
import sys

class BlockTrainingDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        """Reject training dependencies during isolated NumPy inference."""
        # Stop forbidden imports before Python resolves their modules.
        if fullname.split('.')[0] in {'torch', 'qiskit'}:
            raise ImportError(f'{fullname} is disabled during inference')
        return None

sys.meta_path.insert(0, BlockTrainingDependencies())
sys.path.insert(0, sys.argv[1])
import numpy as np
import main_numpy
from video_data import save_videos

root = Path(sys.argv[2])
main_numpy.ESN_PARAMS = root / 'results/results_size_4_leak_rate_0.2/params.npz'
main_numpy.QUANTUM_PARAMS = root / 'results/results_n_2_hz_0.7_jz_1.2_leak_rate_0.2/params.npz'
main_numpy.DATASET_PATH = root / 'final_dataset.pkl'
predictions, metrics = main_numpy.main()
for name, videos in predictions.items():
    save_videos(root / f'{name}_predictions.npz', videos)
(root / 'numpy_metrics.json').write_text(json.dumps(metrics))
assert not any(name.split('.')[0] in {'torch', 'qiskit'} for name in sys.modules)
'''
            subprocess.run(
                [sys.executable, "-c", script,
                 str(Path(__file__).resolve().parents[1] / "src"), directory],
                cwd=directory, check=True, capture_output=True, text=True,
            )
            actual_metrics = json.loads((Path(directory) / "numpy_metrics.json").read_text())
            for name, original_name in (("ESN", "Classical ESN"), ("Quantum", "Quantum")):
                predictions = load_videos(Path(directory) / f"{name}_predictions.npz")
                for actual, expected_video, target in zip(predictions, expected[name], y_test):
                    self.assertEqual(actual.shape, target.shape)
                    np.testing.assert_allclose(actual, expected_video, atol=1e-5)
                np.testing.assert_allclose(
                    actual_metrics[name]["mse"], metrics[original_name]["test"]["loss"], atol=1e-5
                )
                np.testing.assert_allclose(
                    actual_metrics[name]["mae"], metrics[original_name]["test"]["mae"], atol=1e-5
                )


if __name__ == "__main__":
    unittest.main()
