# Train the classical and quantum reservoir readouts for every combination of the settings below.
# Any setting written as a list is swept, e.g. "num_qubits": [4, 6, 8, 10, 12].
#
# NumPy only. Put this file in src/ and run it:  python sweep.py
#
# Everything is saved in RESULTS_DIR after each run:
#   results.json              scores of every trained readout
#   <run>_<readout>.npz       its parameters, readable with reservoirs_numpy.load_model
# Runs go from cheapest to most expensive. Ctrl+C stops the sweep and prints the table of what is done;
# runs already in results.json are skipped, so starting it again carries on where it stopped.
# Use a new RESULTS_DIR after changing the dataset or any setting that is not swept.

import json
import time
from functools import reduce
from itertools import product
from pathlib import Path
import numpy as np
from reservoirs_numpy import ESNReservoir, Linear
from utils_numpy import reservoir_features
from video_data import fit_input_normalization, load_dataset, split_video_indices

# ---------------- settings ----------------
SRC_DIR = Path(__file__).resolve().parent
DATASET_PATH = SRC_DIR / "dataset" / "final_dataset.pkl"
RESULTS_DIR = SRC_DIR / "results" / "sweep"
SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15  # Test uses the remaining fraction.
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 1e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
INPUT_STD_EPS = 1e-8
RIDGES = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]   # ridge strengths tried; the best on validation is kept
leak_rate = 0.3

# "input_gain" multiplies the standardised inputs (ESN_INPUT_GAIN and QUANTUM_INPUT_GAIN in main.py).
# "seed" draws the fixed random weights (the quantum couplings use "coupling_seed").
ESN_CONFIG = {"reservoir_size": [10, 20, 50, 100], "leak_rate": leak_rate, "spectral_radius": 1.0,
              "reservoir_connectivity": 1.0, "time_step": 1, "seed": SEED, "input_gain": 1.0}
QUANTUM_CONFIG = {"num_qubits": [4, 6, 8, 10, 12], "hz": 1, "Jz": 3, "reservoir_time": 1.0, "interaction": "XX",
                  "connectivity": "all_to_all", "single_observables": ("Z", "X"), "pair_observables": ("ZZ", "XX"),
                  "damping": leak_rate, "time_step": 1, "coupling_seed": 42, "seed": SEED, "input_gain": 1.0}


# ---------------- quantum reservoir ----------------
# Same model as reservoirs_numpy.QuantumReservoir (same weights, same features), simulated one demonstration
# at a time without building any 2^n x 2^n gate, which is what makes 10 and 12 qubits practical.
# It needs the pair interaction to be XX, YY or ZZ.
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
BASIS = {"X": H, "Y": np.diag([1, 1j]) @ H, "Z": np.eye(2)}          # P = B Z B^dagger for each Pauli P
COLUMN = {"I": [1, 1], "X": [1, 1], "Y": [1j, -1j], "Z": [1, -1]}    # the non-zero entry in each column of a Pauli

def kron_all(matrices):
    return reduce(np.kron, matrices, np.eye(1))

def kron_apply(A, B, M):
    # (A kron B) @ M without building the big matrix: A acts on the first half of the qubits, B on the rest
    a, b = len(A), len(B)
    M = B @ M.reshape(a, b, -1)
    return (A @ M.reshape(a, -1)).reshape(a * b, -1)

def quantum_reservoir(config):
    n, t = config["num_qubits"], config["reservoir_time"]
    if config["interaction"] not in ("XX", "YY", "ZZ"):
        raise ValueError("this simulation needs an XX, YY or ZZ interaction")

    # fixed random couplings and input weights
    rng = np.random.default_rng(config["coupling_seed"])
    J = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = rng.uniform(-config["Jz"] / 2, config["Jz"] / 2)
    rng = np.random.default_rng(config["seed"])
    w_in = Linear(config["input_dim"], n, rng)

    # pair rotations exp(-i J_ij t/2 P_i P_j): once every qubit is rotated to the basis of P they are all
    # diagonal, and together they multiply basis state k by phase[k]
    if config["connectivity"] == "all_to_all":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairs = [(i, i + 1) for start in (0, 1) for i in range(start, n - 1, 2)]
    z = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1)   # +1/-1 for each qubit of state k; wire 0 is the leftmost bit
    angle = np.zeros(2**n)
    for i, j in pairs:
        angle += float(J[i, j]) * t * z[:, i] * z[:, j]
    phase = np.exp(-0.5j * angle)

    # measured Pauli strings. Column k of a Pauli string has one non-zero entry, values[k] in row rows[k],
    # so <P> = Tr(rho P) = sum_k rho[k, rows[k]] * values[k]
    observables = [{q: p} for p in config["single_observables"] for q in range(n)]
    observables += [{i: p[0], j: p[1]} for p in config["pair_observables"] for i in range(n) for j in range(i + 1, n)]
    rows = np.array([np.arange(2**n) ^ sum(1 << (n - 1 - q) for q, p in o.items() if p in "XY") for o in observables])
    values = np.array([reduce(np.kron, [COLUMN[o.get(q, "I")] for q in range(n)]) for o in observables])

    w_out = Linear(len(observables) * config["time_step"], config["output_dim"], rng)
    return w_in, w_out, J, phase, rows, values

def quantum_features(video, config, w_in, phase, rows, values):
    # features after every frame of one demonstration; the state starts in |00...0> and is never reset
    n, damping, t = config["num_qubits"], config["damping"], config["reservoir_time"]
    half = n // 2
    basis = BASIS[config["interaction"][0]]
    rz = np.diag(np.exp([-0.5j * config["hz"] * t, 0.5j * config["hz"] * t]))
    back = kron_all([basis] * half), kron_all([basis] * (n - half))
    rho = np.zeros((2**n, 2**n), dtype=complex)
    rho[0, 0] = 1
    features = []
    for theta in w_in(video):
        # on each qubit: RY (the input), RZ (the field), then the rotation to the basis of the interaction
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        gates = [basis.conj().T @ rz @ np.array([[c[q], -s[q]], [s[q], c[q]]]) for q in range(n)]
        gates = kron_all(gates[:half]), kron_all(gates[half:])
        frame = []
        for _ in range(config["time_step"]):
            rho = kron_apply(*gates, kron_apply(*gates, rho).conj().T)   # U rho U^dagger, using that rho is Hermitian
            rho *= phase[:, None]                                        # pair interactions
            rho *= phase.conj()
            rho = kron_apply(*back, kron_apply(*back, rho).conj().T)     # back to the computational basis
            for q in range(n):
                # amplitude damping on qubit q: block[:, a, :, :, b] has row bit a and column bit b of that qubit
                block = rho.reshape(2**q, 2, 2**(n - q - 1), 2**q, 2, 2**(n - q - 1))
                block[:, 0, :, :, 0] += damping * block[:, 1, :, :, 1]
                block[:, 1] *= np.sqrt(1 - damping)
                block[:, :, :, :, 1] *= np.sqrt(1 - damping)
            frame.append(np.sum(values * rho[np.arange(2**n), rows], axis=1).real)
        features.append(np.concatenate(frame))
    return np.array(features)


# ---------------- readout ----------------
def scores(F, Y, W, b):
    error = F @ W.T + b - Y
    return {"loss": float(np.mean(error**2)), "mae": float(np.mean(np.abs(error)))}

def adam_fit(F, Y, F_val, Y_val, W, b):
    # the training of utils.train_readout: Adam on the mean squared error from the initial weights W, b,
    # keeping the epoch with the lowest validation loss
    A = np.column_stack([F, np.ones(len(F))])
    C = np.column_stack([W, b])                      # weights, with the bias as the last column
    m, v, step = np.zeros_like(C), np.zeros_like(C), 0
    rng = np.random.default_rng(SEED)
    best = (scores(F_val, Y_val, W, b)["loss"], 0, C.copy())
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(A))
        for start in range(0, len(A), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            error = A[batch] @ C.T - Y[batch]
            gradient = 2 * error.T @ A[batch] / error.size
            step += 1
            m = ADAM_BETAS[0] * m + (1 - ADAM_BETAS[0]) * gradient
            v = ADAM_BETAS[1] * v + (1 - ADAM_BETAS[1]) * gradient**2
            C -= LEARNING_RATE * (m / (1 - ADAM_BETAS[0]**step)) / (np.sqrt(v / (1 - ADAM_BETAS[1]**step)) + ADAM_EPS)
        loss = scores(F_val, Y_val, C[:, :-1], C[:, -1])["loss"]
        if loss < best[0]:
            best = (loss, epoch, C.copy())
    return best[2][:, :-1], best[2][:, -1], {"best_epoch": best[1]}

def ridge_fit(F, Y, F_val, Y_val):
    # exact least squares with a penalty on the weights (not the bias), keeping the penalty with the lowest validation loss
    A = np.column_stack([F, np.ones(len(F))])
    best = (np.inf, None, None)
    for ridge in RIDGES:
        penalty = np.sqrt(ridge) * np.eye(A.shape[1])[:-1]
        C = np.linalg.lstsq(np.vstack([A, penalty]), np.vstack([Y, np.zeros((len(penalty), Y.shape[1]))]), rcond=None)[0]
        loss = scores(F_val, Y_val, C[:-1].T, C[-1])["loss"]
        if loss < best[0]:
            best = (loss, ridge, C)
    return best[2][:-1].T, best[2][-1], {"ridge": best[1]}


# ---------------- sweep ----------------
def combinations(model, config):
    # one (name, model, settings) per combination of the settings written as lists
    swept = [key for key, value in config.items() if isinstance(value, list)]
    for values in product(*[config[key] for key in swept]):
        name = "_".join([model] + [f"{key}_{value}" for key, value in zip(swept, values)])
        yield name, model, {**config, **dict(zip(swept, values))}

def print_table(results):
    width = max(map(len, results))
    print(f"\n{'trained readout':<{width}} {'features':>8} {'selected':>12} {'val loss':>9} {'test loss':>9} {'test MAE':>9}")
    for name, r in sorted(results.items(), key=lambda item: item[1]["validation"]["loss"]):
        selected = f"epoch {r['best_epoch']}" if "best_epoch" in r else f"ridge {r['ridge']:g}"
        print(f"{name:<{width}} {r['features']:>8} {selected:>12} {r['validation']['loss']:>9.4f} "
              f"{r['test']['loss']:>9.4f} {r['test']['mae']:>9.4f}")


if __name__ == "__main__":
    X, Y = load_dataset(DATASET_PATH)
    train, val, test = split_video_indices(len(X), SEED, TRAIN_FRACTION, VAL_FRACTION)
    mean, std = fit_input_normalization([X[i] for i in train], INPUT_STD_EPS)
    Y_train, Y_val, Y_test = [np.concatenate([Y[i] for i in split]) for split in (train, val, test)]
    print(f"{len(X)} demonstrations ({len(train)}/{len(val)}/{len(test)}), {sum(map(len, X))} frames")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    def fit_readouts(F, initial=None):
        # F: features of every demonstration. Returns {readout: (W, b, scores)}, fitted on the training demonstrations.
        F_train, F_val, F_test = [np.concatenate([F[i] for i in split]) for split in (train, val, test)]
        fits = {}
        if initial is not None:
            fits["adam"] = adam_fit(F_train, Y_train, F_val, Y_val, initial.weight, initial.bias)
        fits["ridge"] = ridge_fit(F_train, Y_train, F_val, Y_val)
        for W, b, record in fits.values():
            record.update(features=F_train.shape[1], train=scores(F_train, Y_train, W, b),
                          validation=scores(F_val, Y_val, W, b), test=scores(F_test, Y_test, W, b))
        return fits

    def run(name, model, config):
        start = time.time()
        gain = config.pop("input_gain")
        config.update(input_dim=X[0].shape[1], output_dim=Y[0].shape[1])
        if model == "esn":
            reservoir = ESNReservoir(config)
            w_in, w_out, fixed = reservoir.w_in, reservoir.w_out, {"w_res": reservoir.w_res}
        else:
            w_in, w_out, J, *operators = quantum_reservoir(config)
            fixed = {"J": J}

        # fold the standardisation into w_in so that it takes raw inputs:  W (gain (x - mean) / std) + b = W' x + b'
        w_in.weight = w_in.weight * gain / std
        w_in.bias = w_in.bias - w_in.weight @ mean

        if model == "esn":
            F = reservoir_features(reservoir, X, BATCH_SIZE)
        else:
            F = []
            for video in X:
                F.append(quantum_features(video, config, w_in, *operators))
                if len(F) % 10 == 0:
                    print(".", end="", flush=True)   # a dot every 10 demonstrations

        # parameter files in the layout of utils.export_numpy_checkpoint
        preprocessing = {"method": "training_zscore", "folded_into": "w_in", "mean": mean.tolist(),
                         "std": std.tolist(), "input_gain": gain}
        metadata = {"format_version": 1, "model_type": model, "model_config": config, "input_preprocessing": preprocessing}
        for readout, (W, b, record) in fit_readouts(F, w_out).items():
            parameters = {"w_in.weight": w_in.weight, "w_in.bias": w_in.bias, "w_out.weight": W, "w_out.bias": b}
            np.savez_compressed(RESULTS_DIR / f"{name}_{readout}.npz", metadata=np.array(json.dumps(metadata)),
                                input_mean=mean, input_std=std, **parameters, **fixed)
            results[f"{name}_{readout}"] = {"model": model, "config": config, "input_gain": gain, "readout": readout,
                                            "seconds": round(time.time() - start, 1), **record}
        results_path.write_text(json.dumps(results, indent=2) + "\n")

    # references: a readout on no features (the average action) and on the standardised observation alone
    results["average_action"] = fit_readouts([np.zeros((len(video), 0)) for video in X])["ridge"][2]
    results["linear_on_observation"] = fit_readouts([(video - mean) / std for video in X])["ridge"][2]

    runs = list(combinations("esn", ESN_CONFIG))
    runs += sorted(combinations("quantum", QUANTUM_CONFIG), key=lambda run: run[2]["num_qubits"])   # cheapest first
    for number, (name, model, config) in enumerate(runs, 1):
        if f"{name}_ridge" in results:
            print(f"[{number}/{len(runs)}] {name}: already done")
            continue
        print(f"[{number}/{len(runs)}] {name}:", end="", flush=True)
        try:
            run(name, model, config)
        except KeyboardInterrupt:
            print(" stopped")
            break
        except Exception as error:
            print(f" failed with {type(error).__name__}: {error}")
            continue
        adam, ridge = results[f"{name}_adam"], results[f"{name}_ridge"]
        print(f" {ridge['seconds']:.0f} s, test loss {adam['test']['loss']:.4f} with Adam "
              f"(best epoch {adam['best_epoch']} of {EPOCHS}), {ridge['test']['loss']:.4f} with ridge")

    print_table(results)
