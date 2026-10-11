# hackathon

## Video inputs and decision targets

Videos can have different frame counts. Pass them as lists of NumPy arrays:

```python
X = [video_0, video_1, video_2]       # Each array: (frames_i, features)
y = [target_0, target_1, target_2]    # Each array: (frames_i, decision_vector)
```

`X[i]` and `y[i]` must have the same frame count. All input videos share one
feature width, and all targets share one decision-vector width. These widths
can differ. Every video must contain at least one frame.

For frame `t`, the model observes frames `0..t` and predicts `y[i][t]`.
State resets between videos. Equal-length videos are grouped internally for
batching, and outputs are restored to the original video order. No padding is
used. MSE and MAE average across all actual frame coordinates, so longer videos
contribute more frames to the overall score.

Regular arrays `(N,T,features)` and `(N,T,decision_vector)` still work.
Predictions match the input form: a dense array for dense inputs, or a list of
`(frames_i, decision_vector)` arrays for list inputs.

## Dataset files

Training and NumPy evaluation load `src/dataset/final_dataset.pkl` by default.
The notebook `src/dataset/dataset_generation.ipynb` saves this file next to its
source HDF5 dataset. It contains:

```python
{
    "inputs": [[time_steps, input_values], ...],
    "targets": [[time_steps, action_values], ...],
    "input_keys": [...],
    "target_keys": [...],
    "demo_names": [...],
}
```

`video_data.load_dataset` extracts each video's numeric value matrices and
checks matching time steps. The time-step lists are metadata, not model inputs.
The current Lift dataset has 200 videos, 36–64 frames per video, 19 input
features, and 7 action targets per frame.

For other datasets, the utility functions also support `.npz` archives with one numeric array per video,
keyed `video_000000`, `video_000001`, and so on. They load without pickle:

```python
from video_data import save_videos, load_videos

save_videos("features.npz", X)
save_videos("decision_targets.npz", y)
X = load_videos("features.npz")
y = load_videos("decision_targets.npz")
```

`load_videos` also accepts older dense `.npy` files. Load these yourself and pass
the resulting `X, y` arrays to the entry points. The input and target archives
must use the same video order.

## Train and export

Install `requirements.txt` in the training environment. Configure paths and
model settings at the top of `src/main.py`, then run it in your editor or call:

```python
from main import main as train

models, metrics = train(X, y)
```

Calling `train()` without arguments loads `src/dataset/final_dataset.pkl`.
Set `DATASET_PATH` in `src/main.py` to use another notebook-format pickle.
The script splits by video into 70% training, 15% validation, and 15% test sets:
For example, 200 videos give 140/30/30. Frames from a video stay in one split.

Only `w_out` and its bias are trained with Adam, minimizing MSE. Reservoir
features and aligned targets are cached once per split over real frames.
Set `EPOCHS`, `LEARNING_RATE`, `ADAM_BETAS`, `ADAM_EPS`, `BATCH_SIZE`, and `SEED`
at the top of `src/main.py`. Validation and test targets are excluded from updates.

Inputs are standardized per feature using the mean and standard deviation of
real training frames only. Constant features use a scale of 1. Both reservoirs
use these same statistics. Set `ESN_INPUT_GAIN` and `QUANTUM_INPUT_GAIN` to control
the strength of the standardized signal; the quantum gain controls input rotation
angles. `INPUT_STD_EPS` sets the threshold for treating a feature as constant.
Targets keep their original units.

Standardization and gain are folded into the fixed input weights and bias.
Always pass raw inputs to training and prediction, including direct model calls;
do not standardize them again. Each checkpoint saves `input_mean`, `input_std`,
and `input_preprocessing` metadata, also recorded in the results JSON. NumPy
evaluation uses the exported projection directly. Retrain existing models to
use the new preprocessing.

Training prints training/validation MSE and MAE at epoch 0 and after every epoch.
The weights with the lowest validation MSE are restored before scoring the test
split once. Epoch 0 can be selected if training does not improve validation MSE.

Source files use plain imports. For interactive calls, add `src/` to your
Python import path or open your interpreter there. Running `src/main.py` directly
from your editor also works.

Training saves two named checkpoints and one shared benchmark file directly under
`src/results/`. With the default settings:

```text
results/
  esn_size_100_leak_rate_0.3.npz
  qr_n_6_hz_1.0_jz_10_leak_rate_0.3.npz
  results.json
```

Each `.npz` contains fitted readouts, fixed reservoir parameters, and the model
configuration. Both checkpoints also retain `num_qubits`, `hz`, `Jz`, and
`leak_rate` as scalar arrays and in their JSON metadata. Quantum `damping` uses
`leak_rate`.

`results.json` contains entries for `Classical ESN` and `Quantum`, each with
`params` (model settings, split seed, and Adam settings), `best_epoch`, and final
`validation` and `test` benchmarks with MSE (`loss`) and MAE. Adam settings are
also recorded in the checkpoint metadata. A run saves only these three files;
the shared JSON describes the latest run.

## Evaluate with NumPy only

Install NumPy in the evaluation environment. Copy the selected `.npz` checkpoints
and dataset to the evaluation machine. Set `ESN_PARAMS` and `QUANTUM_PARAMS` in
`src/main_numpy.py` to the chosen `.npz` files. Set `DATASET_PATH` there to
the same `final_dataset.pkl`. It loads the original dataset
and recreates the test split using the same NumPy shuffle and `SEED` as training:

```python
from main_numpy import main as evaluate

predictions, metrics = evaluate()
```

For independent evaluation files, set `USE_TEST_SPLIT = False` to evaluate every
video. Or supply another dataset directly; supplied videos are always evaluated
in full:

```python
predictions, metrics = evaluate(X, y)
esn_predictions = predictions["ESN"]
quantum_predictions = predictions["Quantum"]
# For list input: esn_predictions[i].shape == (len(X[i]), decision_vector).
```

Paths and `BATCH_SIZE` are configured at the top of `src/main_numpy.py`; there
are no command-line arguments. Paths are based on the script's directory and
work independently of the current working directory.

Evaluation imports neither PyTorch nor Qiskit. Both NumPy reservoirs accept
video lists directly through `model(X)`. Use `utils_numpy.predict` for
batching and `utils_numpy.evaluate(model, X, y)` for metrics.
`utils_numpy.reservoir_features` also accepts video lists. Training happens in
`main.py`; NumPy inference applies the exported readout through matrix multiplication.

The quantum reservoir uses NumPy density matrices for input encoding, Ising
evolution, observable measurement, and damping. State memory scales as
`batch_size * 4**num_qubits`; use small qubit counts and bounded batches.
Float32 training and float64 inference may differ slightly due to rounding.

Existing model `.npz` checkpoints remain compatible. Run training again to
create the named checkpoints and shared benchmarks. Older runs used a Torch
shuffle; the recreated NumPy test split applies to new runs.

## Verification

Open `src/reservoir_training_curves.ipynb` to plot epoch versus MSE and MAE for
both reservoirs. Install `requirements_notebook.txt` in the notebook environment.
Its settings cell defines the dataset path, split fractions, seed, batch size,
Adam settings, and every ESN/quantum constructor hyperparameter. These settings
are independent of `main.py`; input and output widths come from the dataset.
It caches reservoir features once and uses the same Adam training function as
`main.py`. It plots training and validation MSE/MAE, restores the best validation
epoch, and evaluates the test split once. Histories stay in the notebook.

Run `src/reservoirs.py` in your editor to check whether both `w_out` readouts are
optimized. It uses the current settings in `main.py`, `final_dataset.pkl`, and
the same video split, without saving checkpoints or results. It prints:

- Training MSE before/after fitting and a mean-target baseline.
- Changes to readout weights and bias; whether fixed reservoir parameters stayed unchanged.
- Whether Adam reduced training MSE and changed the readout parameters.
- Validation/test MSE and PASS/FAIL for each readout.

Both models train their readout with Adam. PASS confirms parameter updates and
reduced training MSE while preserving the fixed reservoir. It does not guarantee
convergence; validation/test metrics measure performance on unseen videos.

From the project root, run `python -m unittest tests.test_numpy tests.test_variable_videos`
in a NumPy-only environment. In a training environment, also run
`python -m unittest tests.test_training_export`. The integration checks cover
dense and variable-length training, video splits, exported parameter equality,
and separate inference processes with Torch/Qiskit imports blocked.
