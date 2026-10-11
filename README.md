# Documentation

# 1. Input Data

## 1.1 Source: robomimic Lift (Proficient-Human), `low_dim_v15.hdf5`

The data comes from robomimic, a framework and benchmark for robot learning from demonstration developed by the Stanford Vision and Learning Lab as part of the ARISE Initiative. robomimic is released under the MIT License (Copyright © 2021 Stanford Vision and Learning Lab).

| | |
|---|---|
| Library | robomimic, <https://github.com/ARISE-Initiative/robomimic> |
| Dataset documentation | <https://robomimic.github.io/docs/v0.3/datasets/robomimic_v0.1.html> |
| Task | **Lift**: a Franka Panda robotic arm grasps a small cube on a table and lifts it |
| Demonstrator type | **Proficient-Human (PH)**: one experienced operator teleoperating the robot via RoboTurk; 200 successful demonstrations |
| File used | `lift/ph/low_dim_v15.hdf5` (low-dimensional observations, no images) |
| File size | ~21 MB |

**Reference.** If you use this dataset, please cite the original robomimic paper:

> A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín, "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation," *Conference on Robot Learning (CoRL)*, 2021. [arXiv:2108.03298](https://arxiv.org/abs/2108.03298)

**Version note.** The demonstrations were recorded in the robosuite simulator (MuJoCo physics), environment version 1.5.1, as stored in the file's env_args metadata. This is why the file is named `low_dim_v15.hdf5` rather than the `low_dim_v141.hdf5` listed on the v0.3 documentation page. Neither robomimic nor robosuite is a runtime dependency of this project: they were used only to obtain and inspect the original file, and the pipeline reads it directly with h5py.

## 1.2 The task

At the start of each episode the cube is placed at a slightly randomized position on the table. The operator moves the end effector (the gripper) above the cube, descends, closes the gripper, and lifts the cube. An episode counts as successful once the cube is raised above a height threshold.

- **Control frequency:** 20 Hz, so one time step is 50 ms. A demo of 48 steps (the average) lasts about 2.4 s.
- **Controller:** Operational Space Control (`OSC_POSE`) in **delta** mode. Each action is a small displacement and rotation of the end effector relative to its current pose, not an absolute target.
- **Reward:** sparse. The reward is 1 only while the cube is lifted. Every demo ends with 5–6 such success steps (`rewards` and `dones` sum to 5 or 6 per demo).

## 1.3 Original file structure (HDF5)

Besides arrays, HDF5 groups can carry small **metadata attributes** (marked "metadata:" in the tree). They describe the group itself rather than holding training data. This project only reads `env_args`, to confirm the task is `"Lift"`.

```
low_dim_v15.hdf5
├── data/                      metadata: environment settings (env_args, JSON text),
│   │                                    total time steps across all demos (total = 9666)
│   ├── demo_0/                metadata: time steps in this demo (num_samples = T),
│   │   │                                MuJoCo scene description (model_file, XML text)
│   │   ├── obs/               observations at time t
│   │   │   ├── robot0_eef_pos          (T, 3)
│   │   │   ├── robot0_eef_quat         (T, 4)
│   │   │   ├── robot0_gripper_qpos     (T, 2)
│   │   │   ├── object                  (T, 10)
│   │   │   ├── robot0_eef_quat_site    (T, 4)    ┐
│   │   │   ├── robot0_gripper_qvel     (T, 2)    │
│   │   │   ├── robot0_joint_pos        (T, 7)    │ present in the file,
│   │   │   ├── robot0_joint_pos_cos    (T, 7)    │ NOT used in this project
│   │   │   ├── robot0_joint_pos_sin    (T, 7)    │
│   │   │   └── robot0_joint_vel        (T, 7)    ┘
│   │   ├── next_obs/          same keys, shifted by one step (obs at t+1)
│   │   ├── actions            (T, 7)
│   │   ├── rewards            (T,)
│   │   ├── dones              (T,)
│   │   └── states             (T, 32)   full MuJoCo simulator state
│   ├── demo_1/
│   ├── ...
│   └── demo_199/
└── mask/                      predefined train/validation splits (lists of demo names)
    ├── train (180)            valid (20)
    ├── 50_percent (100)       50_percent_train (90)   50_percent_valid (10)
    └── 20_percent (40)        20_percent_train (36)   20_percent_valid (4)
```

Each demo is an independent episode with its own length `T`. The listing below shows a few demos and the arrays used by this project:

```
demo_1:   58 time steps | robot0_eef_pos (58, 3), robot0_eef_quat (58, 4), robot0_gripper_qpos (58, 2), object (58, 10), actions (58, 7)
demo_10:  54 time steps | robot0_eef_pos (54, 3), robot0_eef_quat (54, 4), robot0_gripper_qpos (54, 2), object (54, 10), actions (54, 7)
demo_100: 47 time steps | robot0_eef_pos (47, 3), robot0_eef_quat (47, 4), robot0_gripper_qpos (47, 2), object (47, 10), actions (47, 7)
demo_101: 43 time steps | robot0_eef_pos (43, 3), robot0_eef_quat (43, 4), robot0_gripper_qpos (43, 2), object (43, 10), actions (43, 7)
demo_102: 58 time steps | robot0_eef_pos (58, 3), robot0_eef_quat (58, 4), robot0_gripper_qpos (58, 2), object (58, 10), actions (58, 7)
```

Dataset-wide statistics:

| Quantity | Value |
|---|---|
| Number of demos | 200 (`demo_0` … `demo_199`) |
| Time steps per demo | min 36, max 64, mean 48.3 |
| Total time steps | 9,666 |
| Missing values (NaN) | none |

> **Ordering.** HDF5 lists keys alphabetically (`demo_1, demo_10, demo_100, demo_101, …`), which is the order in the listing above. The loading notebook re-sorts the demos numerically (`demo_0, demo_1, demo_2, …`).

## 1.4 Variables used

### Inputs: 19 features per time step (robot and object state)

| HDF5 key | Dim | Column names | Meaning |
|---|---|---|---|
| `robot0_eef_pos` | 3 | `eef_x, eef_y, eef_z` | End-effector (gripper) position in the world frame [m] |
| `robot0_eef_quat` | 4 | `eef_qx, eef_qy, eef_qz, eef_qw` | End-effector orientation as a unit quaternion (x, y, z, w) |
| `robot0_gripper_qpos` | 2 | `finger_1, finger_2` | Positions of the two finger joints [m]. They are approximately mirror images (`finger_2 ≈ −finger_1`); values range from about 0.02 to 0.04 in magnitude |
| `object[0:3]` | 3 | `cube_x, cube_y, cube_z` | Cube position [m] |
| `object[3:7]` | 4 | `cube_qx, cube_qy, cube_qz, cube_qw` | Cube orientation as a quaternion |
| `object[7:10]` | 3 | `grip2cube_x, grip2cube_y, grip2cube_z` | Relative vector from the gripper to the cube. Checked numerically to equal `cube_pos − eef_pos` exactly |

### Targets: 7 values per time step (the operator's action)

| Column | Meaning | Range in the data |
|---|---|---|
| `act_dx, act_dy, act_dz` | Commanded end-effector translation, normalized to [−1, 1] (±1 maps to ±5 cm per step) | [−1, 1] |
| `act_drx, act_dry, act_drz` | Commanded end-effector rotation (axis-angle), normalized to [−1, 1] (±1 maps to ±0.5 rad per step) | about [−1, 0.48] |
| `act_grip` | Gripper command: −1 = open, +1 = close | exactly {−1, +1} (binary) |

The learning problem is therefore **behavior cloning as sequence regression**: given the state sequence up to time *t*, predict the action the human took at time *t*. The gripper target is binary, while the other six are continuous.

**Redundancy note.** Some inputs are redundant: `grip2cube_*` is an exact linear combination of `cube_*` and `eef_*`, and the two finger coordinates are nearly mirror images, this means the 19 inputs carry fewer than 19 independent degrees of freedom. 

## 1.5 From HDF5 to nested lists (`dataset_generation.ipynb`)

The notebook `dataset_generation.ipynb` converts the HDF5 file into plain Python nested lists. It does **no padding and no masking**: every demo keeps its own length. Required packages: `h5py`, `numpy`, `pandas`.

### Settings

| Setting | Default | Effect |
|---|---|---|
| `DATASET` | `~/ionQ/robomimic/datasets/lift/ph/low_dim_v15.hdf5` | Path to the raw file |
| `N_DEMOS` | `200` | Number of demos to load |
| `SPLIT` | `None` | `None` = all 200 demos. Alternatively, one of the predefined masks (`"train"`, `"valid"`, `"20_percent_train"`, …) |
| `RANDOM`, `SEED` | `False`, `0` | Take the first `N_DEMOS` demos in numeric order, or draw them at random with the given seed |
| `TIME_STARTS_AT` | `1` | Time steps are numbered 1, 2, …, T (set to 0 for 0-based numbering) |
| `STATE_KEYS` | `eef_pos, eef_quat, gripper_qpos, object` | Observation groups concatenated into the 19 inputs, in this order |

### What the notebook `dataset_generation.ipynb` does

The notebook opens the HDF5 file, checks that it contains the Lift task, and selects which demos to load (by default all 200, in numeric order). For each demo it joins the four observation groups into one table of 19 input values per time step and takes the 7 action values as targets. Inputs and targets are paired at the same time step: the input is the state the operator saw, and the target is the action they then issued. It then checks that every demo is consistent (matching lengths, 19 inputs and 7 targets per step, no missing values) and saves the result, together with the column names and demo names, to a pickle file (`~/ionQ/lift_lists.pkl` by default).

### Resulting structure

```
inputs  = [video_1, video_2, ..., video_N]        # one element per demo (N = 200)
video_i = [time_steps, values]
    time_steps = [1, 2, ..., T_i]                 # T_i in [36, 64], differs per demo
    values     = [[19 floats at step 1],
                  [19 floats at step 2],
                  ...]                            # T_i rows

targets = [video_1, ..., video_N]                 # same structure, 7 floats per step

input_keys  = ["eef_x", "eef_y", "eef_z", "eef_qx", ..., "grip2cube_z"]   # 19 names
target_keys = ["act_dx", "act_dy", "act_dz", "act_drx", "act_dry", "act_drz", "act_grip"]
demo_names  = ["demo_0", "demo_1", ...]
```

Example (`demo_0`, 59 steps, first time step):

```
t = 1
inputs : [-0.0905, -0.0152, 1.0118, 0.9972, -0.0072, 0.0740, 0.0019, 0.0208, -0.0208,
           0.0264,  0.0270, 0.8314, 0.0000,  0.0000, 0.9691, 0.2466, 0.1169,  0.0422, -0.1804]
targets: [-0.0, 0.0, 0.0, 0.0038, 0.1482, 0.0145, -1.0]
```

## 1.6 Licensing

| Component | Role in this project | License |
|---|---|---|
| robomimic (ARISE-Initiative/robomimic) | Source of the Lift PH dataset and its format | MIT |
| robosuite (ARISE-Initiative/robosuite) | Simulator that generated the demonstrations | MIT |


## 2.Video inputs and decision targets

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
Video counts, frame counts, and vector widths come from the saved dataset.
Training prints these dimensions when loading it.

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
The script splits by video into 70% training, 15% validation, and 15% test sets.
For example, 200 videos give 140/30/30. Frames from a video stay in one split.

Only `w_out` and its bias are trained with Adam, minimizing MSE. Reservoir
features and aligned targets are cached once per split over real frames.
Set `EPOCHS`, `LEARNING_RATE`, `ADAM_BETAS`, `ADAM_EPS`, `BATCH_SIZE`, and `SEED`
at the top of `src/main.py`. Validation and test targets are excluded from updates.

`loss` means mean squared error: the average of `(prediction - target)**2`.
MAE is the average absolute error and is reported alongside the loss; Adam
optimizes MSE. Both metrics compare predictions with the imported decision
targets at the same frame, in their original units.

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

## Reservoir state and streaming

Each batch row has its own reservoir state. For a batch of complete videos,
state is initialized once before the first frame, carried through all frames
and internal steps, then cleared when the call finishes. It is never reset
between frames or predictions within that video. Independent videos start
from fresh states.

Training computes these sequences once when caching reservoir features.
Subsequent Adam batches and epochs update the readout using the cached features;
they do not rerun or reset the reservoir dynamics.

For streaming, call a loaded model with one frame per stream, shaped
`(batch_size, features)`:

```python
from reservoirs_numpy import load_model
from main_numpy import ESN_PARAMS

model = load_model(ESN_PARAMS)  # The same interface works for a quantum checkpoint.
model.initialize(len(X))       # X is a dense (N, T, features) array here.
for t in range(X.shape[1]):
    prediction = model(X[:, t, :])  # Shape: (N, decision_vector); state persists.
```

Keep the batch size and stream order fixed until calling `initialize` for new
videos. Passing `(N, 1, features)` instead invokes complete-video evaluation
and starts fresh state on every call. These state rules apply to both the
training and NumPy reservoir implementations.

## Proposed reservoir feature normalization

The current implementation standardizes **inputs**, as described above.
It does not yet standardize the measured quantum features or the ESN states
before the readout. Quantum observables lie in `[-1, 1]`, but their variances
can differ substantially, which can make Adam optimization uneven.

An optional improvement is to standardize each reservoir feature using mean
and standard deviation fitted on actual training frames only. Use those frozen
statistics for validation, test, and streaming, with a scale of 1 for nearly
constant features. Apply this after measurement and before `w_out`, so it
does not change reservoir evolution or memory. Compare enabled and disabled
normalization for both models; it may help optimization but adds no information
to the features.

If implemented, normalization can be folded into the exported readout. For a
trained weight matrix `W` shaped `(decisions, reservoir_features)`:

```python
W_export = W / feature_std[None, :]
b_export = b - W_export @ feature_mean
# Inference on unnormalized reservoir features:
prediction = features @ W_export.T + b_export
```

This would preserve NumPy inference through matrix multiplication and bias
addition. These readout transformations are a proposal, not current behavior.

## Verification

Open `src/reservoir_training.ipynb` to plot epoch versus MSE and MAE for
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
