# Quantum reservoir computing for robot motion prediction

A trapped-ion-style quantum reservoir computer used as the memory of a robot controller. The reservoir is
benchmarked against classical baselines on a robot manipulation task, and the trained controller runs as
Qiskit circuits on Qollab's simulators.

This effort is supported by Qollab & IonQ.

## Introduction

### Quantum reservoir computing

Reservoir computing is a way to learn from time-dependent data, such as time series, signals or control tasks. The input sequence drives a fixed dynamical system, the *reservoir*, and we only train a simple linear *readout* on top of its state. Classically, the reservoir is a random recurrent neural network called an Echo State Network (ESN). A *quantum* reservoir swaps that network for a set of interacting qubits and uses their measured expectation values as features.

First, n qubits live in a 2<sup>n</sup>-dimensional state space, so a handful of them might do the job of a much larger classical network ([Fujii and Nakajima, 2017](https://arxiv.org/abs/1602.08159), found that up to seven qubits could rival recurrent networks with 500 nodes). Second, the quantum part is never trained, so there's no costly circuit optimisation. Third, the dissipation that usually hurts quantum hardware becomes useful here: it gives the reservoir its fading memory. Whether all this adds up to a real advantage is still an open question, and that's what this project sets out to test.

### The task: imitating a robot arm

We use the **Lift** task from the robomimic benchmark, where a Franka Panda robot arm picks up a cube from a table. The dataset (`low_dim_v15.hdf5`) has 200 demonstrations recorded by a human operator; this project uses the first 100 of them. At each time step, 19 numbers describe the scene: where the gripper is and how it's oriented, how open its fingers are, and where the cube is relative to the gripper. The target is the operator's action at that step, 7 numbers that say how to move and rotate the gripper and whether to open or close it. In short, the model watches the states so far and must guess what the human did next.

### Why quantum reservoirs for this dataset?

This task is a natural fit for reservoir computing. The right action depends not just on the current state but on where we are in the motion (approaching, grasping or lifting), so the model needs memory. The mapping is also partly nonlinear: the gripper command, for example, flips suddenly from "open" to "close" once the gripper reaches the cube, a jump that a weighted sum of the inputs can't reproduce. And with only 19 inputs, the problem is small enough to encode into a few qubits, which makes it a realistic target for today's quantum hardware.

### The reservoir used here

At every time step of a demonstration:

1. The 19 observations are standardised and projected to one rotation angle per qubit (RY).
2. Every qubit gets an RZ rotation, and every pair of qubits an XX rotation with a fixed random strength.
3. Every qubit loses a fraction 0.3 of its excitation (amplitude damping), which makes old inputs fade.
4. The averages of Z and X on every qubit, and of ZZ and XX on every pair, are the features for the readout.

The state is carried from one time step to the next within a demonstration. We compare this with a classical
reservoir (echo state network) and a plain linear model, and run the trained controller as circuits.

## Run it on Qollab

Press **Run**. `main.py` loads the trained 6-qubit policy from `lift_policy.npz` and does three things:

- It simulates the reservoir exactly along one held-out demonstration and prints the error against the human.
- For a few frames it builds the same reservoir as a circuit, runs it on the selected backend, and recomputes
  the action from the measured counts. The figure shows the human (grey), the exact policy (blue) and the
  measured actions (orange dots).
- It prints what `src/sweep.py` ran offline (its settings) and what that gave (MSE and MAE of every trained
  reservoir, from `src/results/sweep/results.json`), and draws the size sweep in the lower half of the figure.

| Backend | What is sent |
|---|---|
| Built-in simulator | 12 circuits of 7 qubits: 6 reservoir qubits and one ancilla, reset after every use |
| IonQ simulators | one circuit of 24 qubits for one frame, because IonQ backends have no reset and take one job per run |

Amplitude damping is built from gates: a controlled rotation moves part of each qubit's excitation onto an
ancilla. The Run settings are `shots`, `replay` (how many past frames each circuit replays; the reservoir
forgets older ones) and `every` (which frames are measured).

Tick `retrain` to train the reservoirs again inside the run instead of only reading the stored results. It
runs the code and data in `src/` (NumPy, in the browser): every classical reservoir, the quantum reservoirs up
to `max_qubits`, and the policy of the demo. The retrained scores are printed next to the stored ones and
marked with rings in the figure. Up to 6 qubits this took about 7 seconds on Qollab and gave the stored scores
back to rounding error; 8 qubits takes longer, and 10 or 12 are better run offline with `src/sweep.py`.

## Results

Test loss is the mean squared error against the human's actions on 15 held-out demonstrations (lower is better).

| Model | Features | Adam readout | Exact (ridge) readout |
|---|---|---|---|
| Classical reservoir, 10 nodes | 10 | 0.0565 | 0.0564 |
| Classical reservoir, 20 nodes | 20 | 0.0481 | 0.0478 |
| Classical reservoir, 50 nodes | 50 | 0.0459 | 0.0456 |
| Classical reservoir, 100 nodes | 100 | 0.0376 | 0.0370 |
| Quantum reservoir, 4 qubits | 20 | 0.0985 | 0.0834 |
| Quantum reservoir, 6 qubits | 42 | 0.0605 | 0.0590 |
| Quantum reservoir, 8 qubits | 72 | 0.0521 | 0.0483 |
| Quantum reservoir, 10 qubits | 110 | 0.0503 | 0.0453 |
| Quantum reservoir, 12 qubits | 156 | 0.0474 | 0.0432 |

For reference, always predicting the average action gives 0.1740 and a linear model on the current
observation gives 0.0448.

![Test error against reservoir size](src/results/sweep/size_sweep.png)

- The quantum reservoir improves steadily with size and reaches the linear model at 10 to 12 qubits.
- The 100-node classical reservoir is clearly ahead. We found no quantum advantage on this task.
- The policy in the Qollab demo also shows the readout the current observation. It scores 0.0430, and 0.0439
  when its features are measured with 1,000 shots.

Limits of these numbers: one random reservoir per size and 15 test demonstrations, so small gaps between rows
are within noise. The Adam readout has not fully converged for the quantum reservoirs. The error is measured
against the human's actions; the policy was not run in the robot simulator, so there is no success rate.

## Files

There are two files called `main.py`. The one at the top is what Qollab runs. `src/main.py` is the PyTorch
training script.

| File | What it does | Needs |
|---|---|---|
| `main.py` | Qollab entry point: runs the trained policy as circuits | Qollab |
| `lift_policy.npz` | the trained policy and one held-out demonstration | |
| `src/sweep.py` | trains both reservoirs at every size (the table above) | NumPy |
| `src/sweep_plots.ipynb` | plots the size sweep | NumPy, Matplotlib |
| `src/train_readout.py` | makes `lift_policy.npz` from the 6-qubit reservoir of the sweep | NumPy |
| `src/main.py` | trains one configuration | PyTorch, Qiskit |
| `src/reservoirs.py`, `src/utils.py` | the reservoir models and training code for `src/main.py` | PyTorch, Qiskit |
| `src/reservoirs_numpy.py`, `src/utils_numpy.py`, `src/main_numpy.py` | NumPy copies of the models, and evaluation of saved parameter files | NumPy |
| `src/video_data.py` | dataset loading, split and normalisation | NumPy |
| `src/scaling_tuned.py`, `src/scaling_core_all_w2.py` | tuned comparison across sets of measured observables | PyTorch, Qiskit, Matplotlib |
| `src/reservoir_training.ipynb` | training curves for one configuration | PyTorch, Qiskit |
| `src/dataset/` | `final_dataset.pkl` (the 100 demonstrations used) and the notebook that makes it | h5py, pandas for the notebook |
| `src/results/sweep/` | scores (`results.json`), trained parameters and the figure of the size sweep | |
| `tests/` | unit tests | |

`requirements.txt` is what Qollab installs and what the NumPy scripts need (NumPy and Qiskit).
`requirements_torch.txt` adds what the PyTorch scripts need, and `requirements_notebook.txt` what the notebooks need.

## Reproduce it

```
pip install -r requirements.txt
cd src
python sweep.py            # the size sweep; the 12-qubit run took about 90 minutes on a laptop
python train_readout.py    # writes lift_policy.npz
```

`sweep.py` skips runs that are already in `src/results/sweep/results.json`, so delete that folder to start
again. To rebuild `final_dataset.pkl`, run `src/dataset/dataset_generation.ipynb` next to robomimic's
`low_dim_v15.hdf5` (see *Input Data* below).

## How it was checked

- The simulation in `sweep.py` gives the same features as `reservoirs_numpy.QuantumReservoir` to 13 decimal
  places, and the same test scores as a parameter file exported by `src/main.py` to 9 decimal places.
- The fast simulation in `scaling_tuned.py` gives the same results as the Qiskit reservoir in
  `src/reservoirs.py` to 7 decimal places on a reduced run.
- On Qollab, the actions recomputed from measured counts match an exact simulation of the same circuits to a
  mean squared difference of 0.001 to 0.003 at 500 shots, on the Built-in simulator and on an IonQ simulator.
- Retraining inside a Qollab run (up to 6 qubits) gives the stored test scores back to within 1e-15.
- `python -m unittest tests.test_numpy tests.test_variable_videos` runs the NumPy tests from the top folder.

The rest of this page documents the data and the training code in detail.

## Input Data

### Source: robomimic Lift (Proficient-Human), `low_dim_v15.hdf5`

The data comes from robomimic, a framework and benchmark for robot learning from demonstration developed by the Stanford Vision and Learning Lab as part of the ARISE Initiative. 

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


### Licensing

| Component | Role in this project | License |
|---|---|---|
| robomimic (ARISE-Initiative/robomimic) | Source of the Lift PH dataset and its format | MIT |
| robosuite (ARISE-Initiative/robosuite) | Simulator that generated the demonstrations | MIT |

### Original file structure (HDF5)

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

### Variables used

#### Inputs: 19 features per time step (robot and object state)

| HDF5 key | Dim | Column names | Meaning |
|---|---|---|---|
| `robot0_eef_pos` | 3 | `eef_x, eef_y, eef_z` | End-effector (gripper) position in the world frame [m] |
| `robot0_eef_quat` | 4 | `eef_qx, eef_qy, eef_qz, eef_qw` | End-effector orientation as a unit quaternion (x, y, z, w) |
| `robot0_gripper_qpos` | 2 | `finger_1, finger_2` | Positions of the two finger joints [m]. They are approximately mirror images (`finger_2 ≈ −finger_1`); values range from about 0.02 to 0.04 in magnitude |
| `object[0:3]` | 3 | `cube_x, cube_y, cube_z` | Cube position [m] |
| `object[3:7]` | 4 | `cube_qx, cube_qy, cube_qz, cube_qw` | Cube orientation as a quaternion |
| `object[7:10]` | 3 | `grip2cube_x, grip2cube_y, grip2cube_z` | Relative vector from the gripper to the cube. Checked numerically to equal `cube_pos − eef_pos` exactly |

#### Targets: 7 values per time step (the operator's action)

| Column | Meaning | Range in the data |
|---|---|---|
| `act_dx, act_dy, act_dz` | Commanded end-effector translation, normalized to [−1, 1] (±1 maps to ±5 cm per step) | [−1, 1] |
| `act_drx, act_dry, act_drz` | Commanded end-effector rotation (axis-angle), normalized to [−1, 1] (±1 maps to ±0.5 rad per step) | about [−1, 0.48] |
| `act_grip` | Gripper command: −1 = open, +1 = close | exactly {−1, +1} (binary) |

The learning problem is therefore **behavior cloning as sequence regression**: given the state sequence up to time *t*, predict the action the human took at time *t*. The gripper target is binary, while the other six are continuous.

**Redundancy note.** Some inputs are redundant: `grip2cube_*` is an exact linear combination of `cube_*` and `eef_*`, and the two finger coordinates are nearly mirror images, this means the 19 inputs carry fewer than 19 independent degrees of freedom. 

### From HDF5 to nested lists (`dataset_generation.ipynb`)

The notebook `dataset_generation.ipynb` (needs `h5py`, `numpy`, `pandas`) turns the HDF5 file into plain Python nested lists, keeping each demo at its own length with no padding. It loads the first 100 of the 200 Lift demos (`N_DEMOS = 100`), joins the four observation groups into 19 inputs per time step, and pairs each step with the 7-value action the operator took at that same step. After checking that every demo is consistent (matching lengths, no missing values), it saves the lists, column names and demo names to a pickle file (`final_dataset.pkl`, next to the HDF5 file).

#### Resulting structure

```
inputs  = [video_1, video_2, ..., video_N]        # one element per demo (N = 100)
video_i = [time_steps, values]
    time_steps = [1, 2, ..., T_i]                 # T_i in [36, 59], differs per demo
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

Install `requirements_torch.txt` in the training environment. Configure paths and
model settings at the top of `src/main.py`, then run it in your editor or call:

```python
from main import main as train

models, metrics = train(X, y)
```

Calling `train()` without arguments loads `src/dataset/final_dataset.pkl`.
Set `DATASET_PATH` in `src/main.py` to use another notebook-format pickle.
The script splits by video into 70% training, 15% validation, and 15% test sets.
The 100 videos here give 70/15/15. Frames from a video stay in one split.

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

Install NumPy in the evaluation environment. By default `ESN_PARAMS` and `QUANTUM_PARAMS`
point at two files of the size sweep in `src/results/sweep/`. Copy the selected `.npz` checkpoints
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

## Credits and disclosure

- **Data:** the Lift demonstrations come from robomimic (Mandlekar et al., CoRL 2021; reference and licences
  under *Input Data* above), recorded in the robosuite simulator. `src/dataset/final_dataset.pkl` and the
  demonstration inside `lift_policy.npz` are derived from that data.
- **Libraries:** Qiskit, NumPy, Matplotlib and PyTorch.
- **Background:** quantum reservoir computing was introduced by Fujii and Nakajima, "Harnessing
  Disordered-Ensemble Quantum Dynamics for Machine Learning", Phys. Rev. Applied 8, 024030 (2017).
- **AI assistance:** code and analysis were developed with the help of AI assistants (Claude and ChatGPT).
  The team ran and checked the results.
- **Build window:** all code here was written during the hackathon, 9 to 11 October 2026.

The code is released under the MIT licence (see `LICENSE`).
