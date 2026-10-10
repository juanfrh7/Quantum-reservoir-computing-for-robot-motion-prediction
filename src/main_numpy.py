"""Load saved NumPy reservoirs and evaluate a video feature array.

Call main() to recreate the training test split, or main(X, y) with
dense arrays or lists of (frames_i, vector_dim) arrays.
"""
from pathlib import Path

from reservoirs_numpy import load_model
from utils_numpy import predict
from video_data import frame_metrics, load_dataset, split_video_indices, validate_pairs

# All settings are here. Relative paths are resolved from this file's directory.
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
ESN_PARAMS = RESULTS_DIR / "results_size_8_leak_rate_0.3" / "params.npz"
QUANTUM_PARAMS = RESULTS_DIR / "results_n_3_hz_1.0_jz_10_leak_rate_0.3" / "params.npz"
DATASET_PATH = BASE_DIR / "dataset/final_dataset.pkl"
SEED = 42  # Match main.py when recreating its test split.
USE_TEST_SPLIT = True  # False evaluates every video in an independent dataset.
BATCH_SIZE = 64

def main(X=None, y=None):
    """Compare aligned decision predictions for dense or variable-length videos."""
    # Supplied arrays are evaluated in full; file loading can recreate the test split.
    select_test = X is None and y is None and USE_TEST_SPLIT
    if X is None or y is None:
        dataset_X, dataset_y = load_dataset(DATASET_PATH)
        X = dataset_X if X is None else X
        y = dataset_y if y is None else y
    if select_test:
        X, y = validate_pairs(X, y)
        _, _, indices = split_video_indices(len(X), SEED)
        X, y = [X[i] for i in indices], [y[i] for i in indices]

    predictions, metrics = {}, {}
    for name, params in (("ESN", ESN_PARAMS), ("Quantum", QUANTUM_PARAMS)):
        model = load_model(params)
        predictions[name] = predict(model, X, batch_size=BATCH_SIZE)
        scores = frame_metrics(predictions[name], y)
        metrics[name] = {"mse": scores["loss"], "mae": scores["mae"]}
        print(f"{name}: MSE={metrics[name]['mse']:.6g}, MAE={metrics[name]['mae']:.6g}")
        if isinstance(predictions[name], list):
            print(f"Predictions: {len(predictions[name])} videos, "
                  f"frames={min(map(len, predictions[name]))}..{max(map(len, predictions[name]))}")
        else:
            print(f"Predictions shape: {predictions[name].shape}")
        print(f"Video 0, frame 0 decision: {predictions[name][0][0]}")
        print()

    return predictions, metrics


if __name__ == "__main__":
    main()
