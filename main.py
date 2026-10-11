# Lift robot policy driven by a 6-qubit quantum reservoir with amplitude damping.
#
# 1. Exact simulation: the reservoir runs through one held-out demonstration and a linear
#    readout turns [observation, reservoir features] into the robot's 7 action values.
# 2. Backend: for a few frames the same reservoir is built as a circuit, measured in the
#    Z and X bases, and the action is recomputed from the measured counts.
#
# `backend` is supplied by Qollab. On the Built-in simulator each circuit has 7 qubits (one ancilla, reset
# after every use). IonQ backends have no reset and take one circuit per run, so there a single 24-qubit
# circuit with fresh ancillas computes one frame.
#
# lift_policy.npz holds the trained policy and the demonstration. It is made by src/train_readout.py from the
# 6-qubit reservoir of the size sweep (src/sweep.py). src/main.py is a different file: the PyTorch training script.
import json
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

# The training code in src/ (NumPy), for the `retrain` setting. It is imported by its full path so that Qollab
# loads the files. The modules there import each other by plain name, so each is registered under that name too.
sweep = train_readout = None
try:
    import sys
    import src.video_data
    sys.modules["video_data"] = src.video_data
    import src.reservoirs_numpy
    sys.modules["reservoirs_numpy"] = src.reservoirs_numpy
    import src.utils_numpy
    sys.modules["utils_numpy"] = src.utils_numpy
    import src.sweep
    sys.modules["sweep"] = sweep = src.sweep
    import src.train_readout
    train_readout = src.train_readout
except Exception as error:
    print(f"Not all of the training code in src/ was loaded: {type(error).__name__}: {error}")

policy = np.load("lift_policy.npz")
OBS = policy["demo_obs"]                      # (T, 19) standardised observations of one demonstration
EXPERT = policy["demo_actions"]               # (T, 7) what the human demonstrator did
ANGLES = OBS @ policy["w_in"].T + policy["b_in"]   # (T, 6) one RY angle per qubit and frame
J = policy["J"]                               # fixed XX couplings
HZ = float(policy["hz"])
GAMMA = float(policy["damping"])
W, B = policy["readout_W"], policy["readout_b"]

N = ANGLES.shape[1]
PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
DAMP_ANGLE = 2 * np.arcsin(np.sqrt(GAMMA))
ACTION_NAMES = ["move x", "move y", "move z", "rotate x", "rotate y", "rotate z", "gripper"]


# ---------------- exact simulation (density matrix) ----------------
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Z = np.diag([1, -1]).astype(complex)

def on_qubit(op, q):
    return reduce(np.kron, [op if i == q else np.eye(2) for i in range(N)])

def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])

def rotation(pauli, theta):
    return np.cos(theta / 2) * np.eye(len(pauli)) - 1j * np.sin(theta / 2) * pauli

EVOLVE = reduce(np.matmul, [rotation(on_qubit(PAULI_X, i) @ on_qubit(PAULI_X, j), J[i, j]) for i, j in PAIRS]
                + [rotation(on_qubit(PAULI_Z, q), HZ) for q in range(N)])
KRAUS = [(on_qubit(np.diag([1, np.sqrt(1 - GAMMA)]), q), on_qubit(np.array([[0, np.sqrt(GAMMA)], [0, 0]]), q))
         for q in range(N)]
OBSERVABLES = ([on_qubit(PAULI_Z, q) for q in range(N)] + [on_qubit(PAULI_X, q) for q in range(N)]
               + [on_qubit(PAULI_Z, i) @ on_qubit(PAULI_Z, j) for i, j in PAIRS]
               + [on_qubit(PAULI_X, i) @ on_qubit(PAULI_X, j) for i, j in PAIRS])

def exact_features(angles):
    # start in |0...0>, feed in the rows of `angles` one by one, return the features after each row
    rho = np.zeros((2**N, 2**N), dtype=complex)
    rho[0, 0] = 1
    features = []
    for theta in angles:
        step = EVOLVE @ reduce(np.kron, [ry(t) for t in theta])
        rho = step @ rho @ step.conj().T
        for k0, k1 in KRAUS:
            rho = k0 @ rho @ k0.conj().T + k1 @ rho @ k1.conj().T
        features.append([np.real(np.trace(rho @ obs)) for obs in OBSERVABLES])
    return np.array(features)

def actions_from(obs, features):
    return np.column_stack([obs, features]) @ W.T + B


# ---------------- the same reservoir as a circuit ----------------
def replay_reservoir(qc, angles, qubits, ancillas, basis, reset):
    # Replay the rows of `angles` on `qubits`, then rotate into the measurement basis.
    # Damping moves a fraction GAMMA of each qubit's |1> onto an ancilla. With reset, one ancilla is reused;
    # without it, every damping event takes a fresh ancilla.
    # The damping after the last row is not in the circuit: it is applied to the measured averages instead.
    ancillas = list(ancillas)
    for step, theta in enumerate(angles):
        for k, q in enumerate(qubits):
            qc.ry(float(theta[k]), q)
            qc.rz(HZ, q)
        for i, j in PAIRS:
            qc.rxx(float(J[i, j]), qubits[i], qubits[j])
        if step < len(angles) - 1:
            for q in qubits:
                ancilla = ancillas[0] if reset else ancillas.pop()
                qc.cry(DAMP_ANGLE, q, ancilla)
                qc.cx(ancilla, q)
                if reset:
                    qc.reset(ancilla)
    if basis == "X":
        for q in qubits:
            qc.h(q)

def circuit_with_reset(angles, basis):
    # 7 qubits: the reservoir and one reusable ancilla
    qc = QuantumCircuit(N + 1, N)
    replay_reservoir(qc, angles, range(N), [N], basis, reset=True)
    qc.measure(range(N), range(N))
    return qc

def circuit_without_reset(angles):
    # one circuit holding two copies of the reservoir, measured in Z and in X, each with its own fresh ancillas
    copy = N + N * (len(angles) - 1)
    qc = QuantumCircuit(2 * copy, 2 * N)
    for c, basis in enumerate("ZX"):
        first = c * copy
        replay_reservoir(qc, angles, range(first, first + N), range(first + N, first + copy), basis, reset=False)
        qc.measure(range(first, first + N), range(c * N, (c + 1) * N))
    return qc

def averages(counts, first=0):
    # <P_q> and <P_i P_j> for the N measured bits starting at classical bit `first` (bit 0 is the last character)
    signs = np.array([[1 - 2 * int(bit) for bit in key[::-1][first:first + N]] for key in counts])
    weights = np.array(list(counts.values())) / sum(counts.values())
    return weights @ signs, np.array([weights @ (signs[:, i] * signs[:, j]) for i, j in PAIRS])

def measured_features(z_averages, x_averages):
    # apply the last damping step to the averages: Z -> (1-g) Z + g,  X -> sqrt(1-g) X  on every qubit
    z, zz = z_averages
    x, xx = x_averages
    g = GAMMA
    zz = (1 - g)**2 * zz + g * (1 - g) * np.array([z[i] + z[j] for i, j in PAIRS]) + g**2
    return np.concatenate([(1 - g) * z + g, np.sqrt(1 - g) * x, zz, (1 - g) * xx])


# ---------------- the size sweep that src/sweep.py ran offline ----------------
def print_sweep_settings():
    # what was run: the settings block at the top of src/sweep.py, printed from the file itself
    print("\nBenchmark. What was run: src/sweep.py (NumPy, offline) with these settings")
    try:
        source = open(sweep.__file__).read()
        print(source.split("# ---------------- settings ----------------\n")[1].split("\n\n\n")[0])
    except OSError:                                       # the file cannot be read: print the values instead
        for name in ("SEED", "TRAIN_FRACTION", "VAL_FRACTION", "BATCH_SIZE", "EPOCHS", "LEARNING_RATE", "ADAM_BETAS",
                     "ADAM_EPS", "INPUT_STD_EPS", "RIDGES", "ESN_CONFIG", "QUANTUM_CONFIG"):
            print(f"{name} = {getattr(sweep, name)}")

def print_sweep_scores(results, title):
    # the test scores of every reservoir in `results`
    print(f"\n{title}: test MSE and MAE against the expert on the 15 held-out demonstrations")
    print(f"{'model':<30} {'readout':>8} {'features':>9} {'selected':>13} {'MSE':>8} {'MAE':>8} {'seconds':>8}")
    for name, r in results.items():
        if "model" not in r:                              # the two reference rows
            model, readout, selected, seconds = name.replace("_", " "), "", "", ""
        else:
            quantum = r["model"] == "quantum"
            size = r["config"]["num_qubits" if quantum else "reservoir_size"]
            model = f"quantum reservoir, {size} qubits" if quantum else f"classical reservoir, {size} nodes"
            readout, seconds = r["readout"], f"{r['seconds']:.0f}"
            selected = f"epoch {r['best_epoch']}" if readout == "adam" else f"ridge {r['ridge']:g}"
        print(f"{model:<30} {readout:>8} {r['features']:>9} {selected:>13} {r['test']['loss']:>8.4f} "
              f"{r['test']['mae']:>8.4f} {seconds:>8}")

def retrain_sweep(max_qubits):
    # Trains the reservoirs again, here and now, with the code, data, seeds and settings of src/sweep.py:
    # every classical size and the quantum sizes up to max_qubits. The scores should equal the stored ones.
    import time
    if sweep is None:
        raise RuntimeError("the training code in src/ was not loaded")
    X, Y = sweep.load_dataset("src/dataset/final_dataset.pkl")
    train, val, test = sweep.split_video_indices(len(X), sweep.SEED, sweep.TRAIN_FRACTION, sweep.VAL_FRACTION)
    mean, std = sweep.fit_input_normalization([X[i] for i in train], sweep.INPUT_STD_EPS)
    split = lambda videos: [np.concatenate([videos[i] for i in part]) for part in (train, val, test)]
    Y_train, Y_val, Y_test = split(Y)
    live = {}

    # the two reference rows: a readout on no features, and on the standardised observation alone
    for name, F in (("average_action", [video[:, :0] for video in X]),
                    ("linear_on_observation", [(video - mean) / std for video in X])):
        F_train, F_val, F_test = split(F)
        weights, bias, selected = sweep.ridge_fit(F_train, Y_train, F_val, Y_val)
        live[name] = {**selected, "features": F_train.shape[1], "test": sweep.scores(F_test, Y_test, weights, bias)}

    runs = list(sweep.combinations("esn", sweep.ESN_CONFIG))
    runs += [run for run in sweep.combinations("quantum", sweep.QUANTUM_CONFIG) if run[2]["num_qubits"] <= max_qubits]
    print(f"\nRetraining {len(runs)} reservoirs now with the code and data in src/ (the steps of run() in src/sweep.py)")
    for name, model, config in runs:
        start = time.time()
        gain = config.pop("input_gain")
        config.update(input_dim=X[0].shape[1], output_dim=Y[0].shape[1])
        if model == "esn":
            reservoir = sweep.ESNReservoir(config)
            w_in, w_out = reservoir.w_in, reservoir.w_out
        else:
            w_in, w_out, _, *operators = sweep.quantum_reservoir(config)
        w_in.weight = w_in.weight * gain / std            # fold the standardisation into the input weights
        w_in.bias = w_in.bias - w_in.weight @ mean
        if model == "esn":
            F = sweep.reservoir_features(reservoir, X, sweep.BATCH_SIZE)
        else:
            F = [sweep.quantum_features(video, config, w_in, *operators) for video in X]
        F_train, F_val, F_test = split(F)
        fits = {"adam": sweep.adam_fit(F_train, Y_train, F_val, Y_val, w_out.weight, w_out.bias),
                "ridge": sweep.ridge_fit(F_train, Y_train, F_val, Y_val)}
        seconds = time.time() - start
        for readout, (weights, bias, selected) in fits.items():
            live[f"{name}_{readout}"] = {"model": model, "config": config, "readout": readout, **selected,
                                         "features": F_train.shape[1], "seconds": seconds,
                                         "test": sweep.scores(F_test, Y_test, weights, bias)}
        print(f"  {name}: {seconds:.0f} s")
    return live

def retrain_policy():
    # trains the policy of the demo again with src/train_readout.py and compares it with the one in lift_policy.npz
    if train_readout is None:
        raise RuntimeError("src/train_readout.py was not loaded")
    print("\nRetraining the policy of the demo (train_policy() in src/train_readout.py)")
    again = train_readout.train_policy("src/dataset/final_dataset.pkl")
    print(f"Largest difference between its readout weights and the ones this run started with: "
          f"{np.abs(again['readout_W'] - W).max():.1e}")

def plot_sweep(fig, results, live):
    # test error against reservoir size, the figure of src/sweep_plots.ipynb; rings mark what this run retrained
    models = [("quantum", "num_qubits", "Number of qubits", "Quantum reservoir"),
              ("esn", "reservoir_size", "Reservoir size (nodes)", "Classical reservoir (ESN)")]
    readouts = {"adam": ("Adam readout", "#2a78d6"), "ridge": ("Ridge readout", "#eb6834")}
    axes = fig.subplots(2, 2, sharey="row")
    for row, (metric, metric_name) in enumerate([("loss", "MSE"), ("mae", "MAE")]):
        for ax, (model, size_key, xlabel, title) in zip(axes[row], models):
            for readout, (label, color) in readouts.items():
                runs = [r for r in results.values() if r.get("model") == model and r["readout"] == readout]
                sizes = sorted({r["config"][size_key] for r in runs})
                scores = [np.mean([r["test"][metric] for r in runs if r["config"][size_key] == size]) for size in sizes]
                ax.plot(sizes, scores, "o-", color=color, linewidth=2, markersize=8, markeredgecolor="white", label=label)
                again = [r for r in (live or {}).values() if r.get("model") == model and r["readout"] == readout]
                ax.plot([r["config"][size_key] for r in again], [r["test"][metric] for r in again], "o", markersize=15,
                        markerfacecolor="none", markeredgecolor="#0b0b0b", markeredgewidth=1.5,
                        label="retrained in this run" if again and readout == "ridge" else None)
            ax.axhline(results["linear_on_observation"]["test"][metric], color="#73726c", linestyle="--",
                       linewidth=1.5, label="Linear model on the current observation")
            if model == "esn":
                ax.set_xscale("log")
            ax.set_xticks(sizes)
            ax.set_xticklabels([str(size) for size in sizes])
            ax.minorticks_off()
            ax.set_xlabel(xlabel)
            ax.set_title(f"{title}: test {metric_name}", fontsize=11, loc="left")
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Benchmark (src/sweep.py): test MSE and MAE on 15 held-out demonstrations against reservoir size, "
                 "lower is better", x=0.01, ha="left", fontsize=13)


def main(shots: int = 500, replay: int = 6, every: int = 8, retrain: bool = False, max_qubits: int = 6):
    # exact reservoir through the whole demonstration
    exact = actions_from(OBS, exact_features(ANGLES))
    print(f"Demonstration with {len(OBS)} frames, {N} reservoir qubits, damping {GAMMA}")
    print(f"Exact reservoir policy vs expert on this demonstration: MSE {np.mean((exact - EXPERT)**2):.4f}, "
          f"MAE {np.mean(np.abs(exact - EXPERT)):.4f}")

    # a few frames on the backend: replay the last `replay` frames, measure Z and X
    frames = list(range(every - 1, len(OBS), every))
    rows = lambda t: ANGLES[max(0, t - replay + 1):t + 1]
    measured = None
    try:
        if "ionq" in str(backend).lower():
            # IonQ: no reset, and one circuit per run. One frame, two frames replayed, 24 qubits.
            replay, frames = min(replay, 2), frames[len(frames) // 2:][:1]
            circuit = circuit_without_reset(rows(frames[0]))
            print(f"\nRunning 1 circuit of {circuit.num_qubits} qubits on the backend ...")
            counts = backend.run(circuit, shots=shots).result().get_counts()
            features = np.array([measured_features(averages(counts, 0), averages(counts, N))])
        else:
            circuits = [circuit_with_reset(rows(t), basis) for t in frames for basis in "ZX"]
            print(f"\nRunning {len(circuits)} circuits of {circuits[0].num_qubits} qubits on the backend ...")
            counts = backend.run(circuits, shots=shots).result().get_counts()
            features = np.array([measured_features(averages(counts[2 * k]), averages(counts[2 * k + 1]))
                                 for k in range(len(frames))])
        measured = actions_from(OBS[frames], features)
        print(f"Backend: {shots} shots per circuit, last {replay} frames replayed per circuit")
        # "exact" is the full-history simulation; "same circuit" is an exact simulation of what was sent
        replayed = actions_from(OBS[frames], [exact_features(rows(t))[-1] for t in frames])
        print(f"{'frame':>5s} {'action':>9s} {'expert':>8s} {'exact':>8s} {'same circuit':>13s} {'measured':>9s}")
        for k, t in enumerate(frames):
            for a in (2, 6):
                print(f"{t:>5d} {ACTION_NAMES[a]:>9s} {EXPERT[t, a]:>8.2f} {exact[t, a]:>8.2f} {replayed[k, a]:>13.2f} {measured[k, a]:>9.2f}")
        print(f"Measured vs exact simulation of the same circuits, mean squared difference: {np.mean((measured - replayed)**2):.4f}")
        print(f"Replaying {replay} frames vs the full history, mean squared difference: {np.mean((replayed - exact[frames])**2):.4f}")
    except Exception as error:
        print(f"\nThis backend could not run the reservoir circuit: {type(error).__name__}: {error}")

    # the size sweep: what was run, what it gave and, in the lower half of the figure, its plot
    try:
        print_sweep_settings()
    except Exception as error:
        print(f"\nSweep settings not shown: {type(error).__name__}: {error}")
    stored = None
    try:
        stored = json.load(open("src/results/sweep/results.json"))
        print_sweep_scores(stored, "What it gave (stored results)")
        print("Trained parameters: lift_policy.npz for the policy above, src/results/sweep/<run>_<readout>.npz for each row")
    except Exception as error:
        print(f"\nSweep results not shown: {type(error).__name__}: {error}")

    # optionally train the reservoirs, and then the policy of the demo, again in this run
    live = None
    if retrain:
        try:
            live = retrain_sweep(max_qubits)
            print_sweep_scores(live, "Retrained in this run")
            if stored:
                difference = max(abs(live[key]["test"]["loss"] - stored[key]["test"]["loss"]) for key in live if key in stored)
                print(f"Largest difference between the retrained and the stored test MSE: {difference:.1e}")
        except Exception as error:
            print(f"\nRetraining the reservoirs stopped: {type(error).__name__}: {error}")
        try:
            retrain_policy()
        except Exception as error:
            print(f"\nRetraining the policy stopped: {type(error).__name__}: {error}")
    results = stored or live

    # top: expert (grey), exact reservoir policy (blue), backend-measured (orange dots)
    fig = plt.figure(figsize=(13, 12.5 if results else 5.5), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[5.5, 7]) if results else (fig, None)
    axes = top.subplots(2, 4, sharex=True)
    for a, ax in enumerate(axes.flat[:7]):
        ax.plot(EXPERT[:, a], color="#73726c", linewidth=2, label="expert")
        ax.plot(exact[:, a], color="#2a78d6", linewidth=2, label="reservoir policy, exact")
        if measured is not None:
            ax.plot(frames, measured[:, a], "o", color="#eb6834", markersize=7,
                    markeredgecolor="white", label=f"reservoir policy, {shots} shots")
        ax.set_title(ACTION_NAMES[a], fontsize=11, loc="left")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("frame")
    axes.flat[7].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[7].legend(handles, labels, loc="center", frameon=False)
    top.suptitle("Lift: action commands along one held-out demonstration", x=0.01, ha="left", fontsize=13)
    if results:
        try:
            plot_sweep(bottom, results, live if stored else None)
        except Exception as error:
            print(f"\nSweep figure not drawn: {type(error).__name__}: {error}")
    plt.show()
