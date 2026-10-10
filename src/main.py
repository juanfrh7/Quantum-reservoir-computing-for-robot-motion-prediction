"""Train classical and quantum readouts on per-frame decision targets."""
import json
from pathlib import Path

import numpy as np
import torch
from reservoirs import ESNReservoir, QiskitQuantumReservoir
from utils import evaluate, export_numpy_checkpoint, fit_readout, make_loader, reservoir_features
from video_data import load_dataset, split_video_indices, validate_pairs

# All experiment settings are defined here.
BATCH_SIZE = 64
SEED = 42
RIDGE = 0.0  # 0 = least squares; positive values regularize the readout weights.
BASE_DIR = Path(__file__).resolve().parent
print(f"Working directory: {BASE_DIR}", flush=True)
RESULTS_DIR = BASE_DIR / "results"
DATASET_PATH = BASE_DIR / "dataset/final_dataset.pkl"
ESN_SIZE = 100
NUM_QUBITS = 4
LEAK_RATE = 0.1
hz = 1.0
Jz = 10


def main(X=None, y=None):
    """Train both readouts, export NumPy parameters, and report held-out metrics."""
    torch.manual_seed(SEED)

    # 1. Load aligned videos and decisions, then split by video.
    if X is None or y is None:
        dataset_X, dataset_y = load_dataset(DATASET_PATH, dtype=np.float32)
        X = dataset_X if X is None else X
        y = dataset_y if y is None else y
    X, y = validate_pairs(X, y, dtype=np.float32)
    train_indices, val_indices, test_indices = split_video_indices(len(X), SEED)
    X_train, y_train = [X[i] for i in train_indices], [y[i] for i in train_indices]
    X_val, y_val = [X[i] for i in val_indices], [y[i] for i in val_indices]
    X_test, y_test = [X[i] for i in test_indices], [y[i] for i in test_indices]
    print(
        f"Videos: N={len(X)}, frames={min(map(len, X))}..{max(map(len, X))}, "
        f"features={X[0].shape[1]}, decisions={y[0].shape[1]}; "
        f"splits={len(train_indices)}/{len(val_indices)}/{len(test_indices)}",
        flush=True,
    )

    # 2. Batch aligned videos and decision targets.
    train_loader = make_loader(X_train, y_train, BATCH_SIZE)
    input_dim = X[0].shape[-1]
    output_dim = y[0].shape[-1]

    # 3. Define the two reservoirs. Their inputs and dynamics are fixed.
    model_configs = {
        "Classical ESN": {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "reservoir_size": ESN_SIZE,
            "leak_rate": LEAK_RATE,
            "spectral_radius": 0.9,
            "time_step": 1,
        },
        "Quantum": {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "num_qubits": NUM_QUBITS,
            "hz": hz,
            "Jz": Jz,
            "interaction": "XX",
            "single_observables": ("Z", "X"),
            "pair_observables": ("ZZ", "XX"),
            "damping": LEAK_RATE,
            "coupling_seed": SEED,
        },
    }
    models = {
        "Classical ESN": ESNReservoir(model_configs["Classical ESN"]),
        "Quantum": QiskitQuantumReservoir(**model_configs["Quantum"]),
    }
    # Record the experiment settings in both exports; quantum uses leak_rate as damping.
    hyperparameters = {
        "num_qubits": NUM_QUBITS,
        "hz": hz,
        "Jz": Jz,
        "leak_rate": LEAK_RATE,
    }
    # Give each model's final parameters and benchmarks a descriptive folder.
    result_dirs = {
        "Classical ESN": RESULTS_DIR / f"results_size_{ESN_SIZE}_leak_rate_{LEAK_RATE}",
        "Quantum": RESULTS_DIR / (
            f"results_n_{NUM_QUBITS}_hz_{hz}_jz_{Jz}_leak_rate_{LEAK_RATE}"
        ),
    }
    results = {}

    for name, model in models.items():
        # 4. Only w_out is optimized. Simulate fixed features once.
        model.w_in.requires_grad_(False)
        print(f"\n{name}: computing reservoir features", flush=True)
        train = reservoir_features(model, train_loader, BATCH_SIZE)

        # 5. Solve for the optimal readout using only training frames/targets.
        features, targets = train.dataset.tensors
        fit_readout(model, features, targets, ridge=RIDGE)

        # 6. Evaluate the final readout on validation and test splits.
        results[name] = {
            "params": {**model_configs[name], **hyperparameters, "seed": SEED, "ridge": RIDGE},
            "validation": evaluate(model, X_val, y_val, BATCH_SIZE),
            "test": evaluate(model, X_test, y_test, BATCH_SIZE),
        }
        model_type = "esn" if name == "Classical ESN" else "quantum"
        result_dir = result_dirs[name]
        export_numpy_checkpoint(
            model, result_dir / "params.npz", model_type, model_configs[name],
            hyperparameters=hyperparameters,
        )
        (result_dir / "results.json").write_text(json.dumps(results[name], indent=2) + "\n")
        print(f"Saved parameters and final benchmarks: {result_dir}", flush=True)

    # 7. Print the comparison; return models and metrics for later use.
    print("\nModel            Val loss      Val MAE       Test loss     Test MAE")
    for name, metrics in results.items():
        val, test = metrics["validation"], metrics["test"]
        print(
            f"{name:<16} {val['loss']:<13.6g} {val['mae']:<13.6g} "
            f"{test['loss']:<13.6g} {test['mae']:.6g}"
        )
    return models, results


if __name__ == "__main__":
    main()
