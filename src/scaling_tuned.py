"""
scaling_tuned.py
================
Quantum reservoir vs classical ESN scaling, with every point re-tuned.

For every observable set in OBS_SETS and every qubit number n in QUBITS:
    1. tune ALL quantum hyper-parameters (input gain, damping, hz, Jz, TIME_STEP,
       readout learning rate) by coordinate descent on the validation MSE,
       averaged over SEEDS (warm-started from the best settings of n-1);
    2. report test MSE/MAE (mean +- std over SEEDS) for the tuned settings.
The ESN is tuned the same way for every reservoir size N (input gain, leak rate,
spectral radius, readout learning rate). A linear readout on the raw
(standardized) inputs is the baseline.

Outputs (in results/scaling_tuned/):
    scaling_tuned_<obs_set>.png   one scaling.png-style figure per observable set
    scaling_tuned_all_obs.png     all observable sets on one figure
    tuned_points.json             tuned settings + metrics for every point
    tuned_points.csv              same, as a flat table
    eval_cache.json               every (config, seed) evaluation -> safe to stop and rerun

Uses the CURRENT main.py / utils.py / reservoirs.py / video_data.py unchanged:
  - data loading, the 70/15/15 split and z-scoring are the same as main.py;
  - models are the same classes (ESNReservoir, QiskitQuantumReservoir);
  - the readout is trained with utils.train_readout (Adam, best-val epoch), as in main.py.
Put this file next to main.py and run:  python scaling_tuned.py
"""
import csv
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from torch.utils.data import DataLoader, TensorDataset

import main as M  # only reads constants; main() is not executed
from reservoirs import ESNReservoir, QiskitQuantumReservoir
from utils import (cached_metrics, configure_input_normalization, make_loader,
                   reservoir_features, train_readout)
from video_data import fit_input_normalization, load_dataset, split_video_indices, validate_pairs

# =============================================================================
# SETTINGS
# =============================================================================
QUICK = False            # True = small smoke test (few videos, n=2..3, tiny grids). Run this first.

SEEDS = [42, 1, 2]       # model seeds: w_in, Ising couplings J, ESN w_res, readout init + shuffling
SPLIT_SEED = M.SEED      # the train/val/test video split is kept FIXED across seeds
QUBITS = [2, 3, 4, 5, 6]
ESN_SIZES = [5, 10, 15, 20, 30, 45, 66, 100, 150, 200, 300]
TUNE_ROUNDS = 2          # coordinate-descent passes over the grid (stops early if nothing changes)

# "adam"  = utils.train_readout exactly as main.py (Adam, EPOCHS, best-val epoch). Faithful but slower.
# "lstsq" = closed-form least squares on standardized features, with a "ridge" knob. ~10x faster.
READOUT = "adam"

# Fast, exact re-implementation of QiskitQuantumReservoir's step in batched torch.
# It is checked against the real Qiskit model at start-up for every n (VERIFY_FAST_SIM);
# the script stops if they disagree. USE_FAST_SIM=False uses the Qiskit model directly (very slow).
USE_FAST_SIM = True
VERIFY_FAST_SIM = True

OUT_DIR = M.RESULTS_DIR / "scaling_tuned"

# ----- observable sets: one figure per entry --------------------------------
PAIRS_ALL = tuple(a + b for a in "XYZ" for b in "XYZ")
OBS_SETS = OrderedDict([
    ("main_default", dict(single=("Z", "X"), pair=("ZZ", "XX"), label="Z,X + ZZ,XX (main.py default)")),
    ("local_Z",      dict(single=("Z",),     pair=(),           label="local Z only")),
    ("local_XYZ",    dict(single=("X", "Y", "Z"), pair=(),      label="local X,Y,Z")),
    ("Z_ZZ",         dict(single=("Z",),     pair=("ZZ",),      label="Z + ZZ")),
    ("all_w2",       dict(single=("X", "Y", "Z"), pair=PAIRS_ALL, label="all weight<=2 Paulis")),
])

# ----- quantum: starting point (n=2) and grid --------------------------------
# input_gain = QUANTUM_INPUT_GAIN in main.py; damping = what main.py passes as LEAK_RATE.
Q_BASE = dict(input_gain=0.03, damping=0.1, hz=2.0, Jz=10.0, time_step=1,
              interaction=M.INTERACTION, connectivity="all_to_all", lr=1e-3, ridge=0.0)
Q_GRID = OrderedDict([
    ("input_gain", [0.01, 0.03, 0.1, 0.3, 1.0]),
    ("damping",    [0.01, 0.03, 0.1, 0.3]),
    ("hz",         [0.5, 1.0, 2.0, 4.0]),
    ("Jz",         [1.0, 3.0, 10.0]),
    ("time_step",  [1, 2]),
    # ("interaction", ["XX", "ZZ"]),            # uncomment to also tune the coupling type
    # ("connectivity", ["all_to_all", "nearest_neighbour"]),
])
# ----- ESN: starting point (smallest N) and grid ------------------------------
E_BASE = dict(input_gain=M.ESN_INPUT_GAIN, leak_rate=M.LEAK_RATE,
              spectral_radius=M.SPECTRAL_RADIUS, time_step=1, lr=1e-3, ridge=0.0)
E_GRID = OrderedDict([
    ("input_gain",      [0.1, 0.3, 1.0, 3.0]),
    ("leak_rate",       [0.1, 0.3, 0.6, 1.0]),
    ("spectral_radius", [0.5, 0.9, 1.2]),
])
# readout knob, appended to both grids
READOUT_GRID = {"adam": ("lr", [1e-3, 3e-3, 1e-2]),
                "lstsq": ("ridge", [0.0, 1e-4, 1e-2])}[READOUT]

EPOCHS = M.EPOCHS
QUICK_VIDEOS = (24, 8, 8)
if QUICK:
    SEEDS, QUBITS, ESN_SIZES, TUNE_ROUNDS, EPOCHS = [42, 1], [2, 3], [5, 15, 30], 1, 20
    Q_GRID = OrderedDict([("input_gain", [0.03, 0.3]), ("damping", [0.03, 0.1])])
    E_GRID = OrderedDict([("leak_rate", [0.3, 1.0])])
    OBS_SETS = OrderedDict((k, OBS_SETS[k]) for k in ("main_default", "all_w2"))
    OUT_DIR = M.RESULTS_DIR / "scaling_tuned_quick"
Q_GRID[READOUT_GRID[0]] = READOUT_GRID[1]
E_GRID[READOUT_GRID[0]] = READOUT_GRID[1]
READOUT_KEYS = ("lr", "ridge")  # do not affect the reservoir simulation

# =============================================================================
# DATA (identical to main.py steps 1-3)
# =============================================================================


def load_data():
    X, y = load_dataset(M.DATASET_PATH, dtype=np.float32)
    X, y = validate_pairs(X, y, dtype=np.float32)
    idx = split_video_indices(len(X), SPLIT_SEED)
    data = {}
    for name, ids, k in zip(("train", "val", "test"), idx, QUICK_VIDEOS):
        ids = list(ids)[:k] if QUICK else list(ids)
        data[name] = ([X[i] for i in ids], [y[i] for i in ids])
    mean, std = fit_input_normalization(data["train"][0], M.INPUT_STD_EPS)
    loaders = {k: make_loader(*v, M.BATCH_SIZE) for k, v in data.items()}
    print("Videos train/val/test: " + "/".join(str(len(v[0])) for v in data.values()), flush=True)
    return dict(data=data, loaders=loaders, mean=mean, std=std,
                input_dim=X[0].shape[-1], output_dim=y[0].shape[-1])


# =============================================================================
# OBSERVABLES
# =============================================================================


def obs_labels(n, single, pair):
    """Same label construction/order as QiskitQuantumReservoir.__init__."""
    labels = []
    for op in single:
        for w in range(n):
            lab = ["I"] * n
            lab[w] = op
            labels.append("".join(lab))
    for op in pair:
        for i in range(n):
            for j in range(i + 1, n):
                lab = ["I"] * n
                lab[i], lab[j] = op
                labels.append("".join(lab))
    return labels


FULL_SINGLE, FULL_PAIR = ("X", "Y", "Z"), PAIRS_ALL  # superset simulated once, then sliced


def obs_columns(n, obs_name, time_step):
    full = obs_labels(n, FULL_SINGLE, FULL_PAIR)
    spec = OBS_SETS[obs_name]
    pos = [full.index(lab) for lab in obs_labels(n, spec["single"], spec["pair"])]
    return [r * len(full) + p for r in range(time_step) for p in pos]


# =============================================================================
# MODELS
# =============================================================================


def build_quantum(D, n, cfg, seed):
    torch.manual_seed(seed)
    model = QiskitQuantumReservoir(
        D["input_dim"], D["output_dim"], num_qubits=n, hz=cfg["hz"], Jz=cfg["Jz"],
        interaction=cfg["interaction"], connectivity=cfg["connectivity"],
        single_observables=FULL_SINGLE, pair_observables=FULL_PAIR,
        damping=cfg["damping"], time_step=cfg["time_step"], coupling_seed=seed)
    configure_input_normalization(model, D["mean"], D["std"], cfg["input_gain"])
    return model


def build_esn(D, N, cfg, seed):
    torch.manual_seed(seed)
    model = ESNReservoir({"input_dim": D["input_dim"], "output_dim": D["output_dim"],
                          "reservoir_size": N, "leak_rate": cfg["leak_rate"],
                          "spectral_radius": cfg["spectral_radius"],
                          "time_step": cfg["time_step"]})
    configure_input_normalization(model, D["mean"], D["std"], cfg["input_gain"])
    return model


class FastQuantumSim:
    """Batched torch version of QiskitQuantumReservoir.reservoir_step.

    Per step: rho -> Damp( U_ising R(theta) rho R(theta)^dag U_ising^dag ),
    R = RY(theta_0) (x) ... (x) RY(theta_{n-1}) (wire 0 = leftmost factor, as in the class),
    U_ising taken from Qiskit itself (model.apply_ising_evolution), so conventions match.
    """

    def __init__(self, model):
        self.n, self.d, self.ts = model.num_qubits, model.reservoir_dim, model.time_step
        qc = QuantumCircuit(self.n)
        model.apply_ising_evolution(qc)
        self.U = torch.as_tensor(Operator(qc).data, dtype=torch.complex128)
        g = model.damping
        self.K = None if g == 0 else [
            torch.tensor([[1, 0], [0, np.sqrt(1 - g)]], dtype=torch.complex128),
            torch.tensor([[0, np.sqrt(g)], [0, 0]], dtype=torch.complex128)]
        self.obs = model.observables.to(torch.complex128)
        self.W = model.w_in.weight.detach().double()
        self.b = model.w_in.bias.detach().double()

    def _ry(self, a):  # (B,n) -> (B,d,d)
        c, s = torch.cos(a / 2), torch.sin(a / 2)
        R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (B,n,2,2)
        out = R[:, 0]
        for w in range(1, self.n):
            out = torch.einsum("bij,bkl->bikjl", out, R[:, w]).reshape(len(a), 2 ** (w + 1), 2 ** (w + 1))
        return out.to(torch.complex128)

    def _damp(self, rho):
        B, n = rho.shape[0], self.n
        t = rho.reshape(B, *([2] * (2 * n)))
        for w in range(n):
            acc = 0
            for K in self.K:
                x = (t.movedim(1 + w, -1) @ K.T).movedim(-1, 1 + w)
                x = (x.movedim(1 + n + w, -1) @ K.conj().T).movedim(-1, 1 + n + w)
                acc = acc + x
            t = acc
        return t.reshape(B, self.d, self.d)

    def step(self, rho, angles):
        V = self.U @ self._ry(angles)
        rho = V @ rho @ V.conj().transpose(-1, -2)
        return self._damp(rho) if self.K is not None else rho

    @torch.no_grad()
    def features(self, loader):
        """Same output (order, time_step concat, float32) as utils.reservoir_features."""
        feats, targs = [], []
        for X, y in loader:
            ang = X.double() @ self.W.T + self.b  # (B,T,n)
            rho = torch.zeros(len(X), self.d, self.d, dtype=torch.complex128)
            rho[:, 0, 0] = 1
            steps = []
            for t in range(X.shape[1]):
                per = []
                for _ in range(self.ts):
                    rho = self.step(rho, ang[:, t])
                    per.append(torch.einsum("bij,kji->bk", rho, self.obs).real)
                steps.append(torch.cat(per, -1))
            feats.append(torch.stack(steps, 1).flatten(0, 1).float())
            targs.append(y.flatten(0, 1))
        return torch.cat(feats), torch.cat(targs)


def verify_fast_sim(D, n):
    """Compare FastQuantumSim with the real Qiskit model on 2 short videos."""
    cfg = dict(Q_BASE, input_gain=0.5, damping=0.2, hz=1.3, Jz=4.0, time_step=2)
    model = build_quantum(D, n, cfg, seed=7)
    Xs, ys = D["data"]["train"]
    T = min(6, len(Xs[0]), len(Xs[1]))
    loader = make_loader([Xs[0][:T], Xs[1][:T]], [ys[0][:T], ys[1][:T]], 2)
    ref = reservoir_features(model, loader, 2).dataset.tensors[0]
    fast = FastQuantumSim(model).features(loader)[0]
    err = (ref - fast).abs().max().item()
    print(f"  fast-sim check n={n}: max |qiskit - fast| = {err:.2e}", flush=True)
    if err > 1e-4:
        raise RuntimeError("Fast simulator disagrees with QiskitQuantumReservoir; set USE_FAST_SIM=False.")


def compute_features(D, kind, size, cfg, seed):
    """Full-observable features for train/val/test: {split: (F, Y)}."""
    if kind == "linear":
        out = {}
        for k, (X, y) in D["data"].items():
            Xf = np.concatenate(X)
            F = (Xf - D["mean"]) / D["std"]
            out[k] = (torch.as_tensor(F, dtype=torch.float32), torch.as_tensor(np.concatenate(y)))
        return out
    model = build_quantum(D, size, cfg, seed) if kind == "quantum" else build_esn(D, size, cfg, seed)
    if kind == "quantum" and USE_FAST_SIM:
        sim = FastQuantumSim(model)
        return {k: sim.features(L) for k, L in D["loaders"].items()}
    return {k: tuple(reservoir_features(model, L, M.BATCH_SIZE).dataset.tensors)
            for k, L in D["loaders"].items()}


# =============================================================================
# READOUT
# =============================================================================


class ReadoutHead(torch.nn.Module):
    """Minimal object with the attributes utils.train_readout / cached_metrics use."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.w_in = torch.nn.Linear(1, 1)  # dummy; train_readout freezes model.w_in
        self.w_out = torch.nn.Linear(n_in, n_out)


def fit_readout(F, lr, ridge, seed):
    (Ftr, Ytr), (Fva, Yva), (Fte, Yte) = F["train"], F["val"], F["test"]
    if READOUT == "adam":
        mk = lambda a, b: DataLoader(TensorDataset(a, b.float()), batch_size=M.BATCH_SIZE)
        torch.manual_seed(seed)
        head = ReadoutHead(Ftr.shape[1], Ytr.shape[1])
        fit = train_readout(head, mk(Ftr, Ytr), mk(Fva, Yva), EPOCHS, lr,
                            M.ADAM_BETAS, M.ADAM_EPS, seed, verbose=False)
        return dict(val=fit["validation"], test=cached_metrics(head, mk(Fte, Yte)),
                    best_epoch=fit["best_epoch"])
    # closed-form ridge on standardized features (fair across feature scales)
    Ftr, Fva, Fte = Ftr.double(), Fva.double(), Fte.double()
    mu, sd = Ftr.mean(0), Ftr.std(0)
    sd[sd < 1e-12] = 1.0
    A = lambda Z: torch.cat([(Z - mu) / sd, torch.ones(len(Z), 1, dtype=Z.dtype)], 1)
    Atr = A(Ftr)
    reg = ridge * len(Atr) * torch.eye(Atr.shape[1], dtype=Atr.dtype)
    reg[-1, -1] = 0
    Wt = torch.linalg.lstsq(Atr.T @ Atr + reg, Atr.T @ Ytr.double()).solution
    met = lambda Z, Y: (lambda r: {"loss": r.square().mean().item(), "mae": r.abs().mean().item()})(A(Z) @ Wt - Y.double())
    return dict(val=met(Fva, Yva), test=met(Fte, Yte), best_epoch=None)


# =============================================================================
# CACHED EVALUATION
# =============================================================================


class Evaluator:
    def __init__(self, D, path):
        self.D, self.path = D, path
        self.cache = json.loads(path.read_text()) if path.exists() else {}
        self.feat = OrderedDict()  # small LRU of simulated features
        self.new = 0

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache))
        tmp.replace(self.path)

    def run(self, kind, size, cfg, seed, obs=None):
        key = json.dumps([READOUT, EPOCHS, QUICK, kind, size, obs, seed, sorted(cfg.items())])
        if key in self.cache:
            return self.cache[key]
        dyn = {k: v for k, v in cfg.items() if k not in READOUT_KEYS}
        fkey = json.dumps([kind, size, seed, sorted(dyn.items())])
        if fkey not in self.feat:
            self.feat[fkey] = compute_features(self.D, kind, size, dyn, seed)
            if len(self.feat) > 3:
                self.feat.popitem(last=False)
        self.feat.move_to_end(fkey)
        F = self.feat[fkey]
        if kind == "quantum":
            cols = obs_columns(size, obs, cfg["time_step"])
            F = {k: (v[0][:, cols], v[1]) for k, v in F.items()}
        res = fit_readout(F, cfg.get("lr", 1e-3), cfg.get("ridge", 0.0), seed)
        res["n_features"] = int(F["train"][0].shape[1])
        self.cache[key] = res
        self.new += 1
        if self.new % 5 == 0:
            self._save()
        return res

    def val_score(self, kind, size, cfg, obs=None):
        return float(np.mean([self.run(kind, size, cfg, s, obs)["val"]["loss"] for s in SEEDS]))

    def summary(self, kind, size, cfg, obs=None):
        rs = [self.run(kind, size, cfg, s, obs) for s in SEEDS]
        out = {"config": cfg, "n_features": rs[0]["n_features"], "seeds": SEEDS}
        for split in ("val", "test"):
            for m in ("loss", "mae"):
                v = [r[split][m] for r in rs]
                out[f"{split}_{'mse' if m == 'loss' else 'mae'}_mean"] = float(np.mean(v))
                out[f"{split}_{'mse' if m == 'loss' else 'mae'}_std"] = float(np.std(v))
        return out


def tune(ev, kind, size, base, grid, obs=None, tag=""):
    """Coordinate descent on mean validation MSE over SEEDS."""
    best = dict(base)
    best_s = ev.val_score(kind, size, best, obs)
    t0 = time.time()
    print(f"\n[{tag}] start val MSE={best_s:.5f}  {best}", flush=True)
    for rnd in range(TUNE_ROUNDS):
        changed = False
        for p, values in grid.items():
            scores = {}
            for v in values:
                scores[v] = best_s if v == best[p] else ev.val_score(kind, size, {**best, p: v}, obs)
            v_best = min(scores, key=scores.get)
            print(f"  [{tag}] round {rnd + 1} {p}: " +
                  "  ".join(f"{v}->{s:.5f}" for v, s in scores.items()) + f"  => {v_best}", flush=True)
            if v_best != best[p] and scores[v_best] < best_s:
                best[p], best_s, changed = v_best, scores[v_best], True
        if not changed:
            break
    print(f"[{tag}] best val MSE={best_s:.5f} ({time.time() - t0:.0f}s)  {best}", flush=True)
    return best


# =============================================================================
# PLOTS
# =============================================================================
C_Q, C_E, C_L = "#E8663A", "#2A78D6", "0.5"


def _curve(ax, xs, pts, color, label, annotate=None):
    m = np.array([p["test_mse_mean"] for p in pts])
    s = np.array([p["test_mse_std"] for p in pts])
    ax.fill_between(xs, m - s, m + s, color=color, alpha=0.15, lw=0)
    ax.errorbar(xs, m, yerr=s, color=color, marker="o", lw=2, capsize=3, label=label)
    if annotate:
        for x, y, t in zip(xs, m, annotate):
            ax.annotate(t, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)


def _style(ax, xlabel):
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_obs_set(R, obs):
    q = [R["quantum"][obs][str(n)] for n in QUBITS if str(n) in R["quantum"][obs]]
    ns = [n for n in QUBITS if str(n) in R["quantum"][obs]]
    e = [R["esn"][str(N)] for N in ESN_SIZES if str(N) in R["esn"]]
    Ns = [N for N in ESN_SIZES if str(N) in R["esn"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    for ax, mode in zip(axes, ("features", "physical")):
        if mode == "features":
            xq, xe, xl = [p["n_features"] for p in q], [p["n_features"] for p in e], "readout features"
        else:
            xq, xe, xl = ns, Ns, "physical size (qubits / neurons)"
        ax.axhline(R["linear"]["test_mse_mean"], color=C_L, ls="--", lw=1.5, label="Linear (raw inputs)")
        _curve(ax, xe, e, C_E, "ESN (N neurons, tuned per N)")
        _curve(ax, xq, q, C_Q, f"Quantum ({OBS_SETS[obs]['label']}, tuned per n)",
               annotate=[f"n={n}" for n in ns])
        _style(ax, xl)
    axes[0].set_ylabel("Test MSE")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(f"Quantum [{OBS_SETS[obs]['label']}] vs ESN, every point tuned  "
                 f"(mean ± std over seeds {SEEDS}, readout={READOUT})")
    lines = [f"n={n}: " + ", ".join(f"{k}={v}" for k, v in p["config"].items()
                                     if k not in ("connectivity",)) for n, p in zip(ns, q)]
    fig.text(0.5, 0.005, "tuned quantum settings  |  " + "   ".join(lines), ha="center", va="bottom",
             fontsize=6.5, color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = OUT_DIR / f"scaling_tuned_{obs}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}", flush=True)


def plot_all(R):
    cmap = plt.get_cmap("Oranges")
    obs_list = [o for o in OBS_SETS if R["quantum"].get(o)]
    e = [R["esn"][str(N)] for N in ESN_SIZES if str(N) in R["esn"]]
    Ns = [N for N in ESN_SIZES if str(N) in R["esn"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    for ax, mode in zip(axes, ("features", "physical")):
        ax.axhline(R["linear"]["test_mse_mean"], color=C_L, ls="--", lw=1.5, label="Linear (raw inputs)")
        _curve(ax, [p["n_features"] for p in e] if mode == "features" else Ns, e, C_E, "ESN (tuned per N)")
        for i, o in enumerate(obs_list):
            ns = [n for n in QUBITS if str(n) in R["quantum"][o]]
            q = [R["quantum"][o][str(n)] for n in ns]
            col = cmap(0.35 + 0.65 * i / max(1, len(obs_list) - 1))
            _curve(ax, [p["n_features"] for p in q] if mode == "features" else ns, q, col,
                   f"Q: {OBS_SETS[o]['label']}")
        _style(ax, "readout features" if mode == "features" else "physical size (qubits / neurons)")
    axes[0].set_ylabel("Test MSE")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"All observable sets, every point tuned (mean ± std over seeds {SEEDS})")
    fig.tight_layout()
    path = OUT_DIR / "scaling_tuned_all_obs.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}", flush=True)


def write_csv(R):
    rows = []
    rows.append({"model": "linear", "obs_set": "", "size": "", **R["linear"]})
    for N, p in R["esn"].items():
        rows.append({"model": "esn", "obs_set": "", "size": N, **p})
    for o, d in R["quantum"].items():
        for n, p in d.items():
            rows.append({"model": "quantum", "obs_set": o, "size": n, **p})
    keys = ["model", "obs_set", "size", "n_features", "test_mse_mean", "test_mse_std", "test_mae_mean",
            "test_mae_std", "val_mse_mean", "val_mse_std", "val_mae_mean", "val_mae_std", "config"]
    with open(OUT_DIR / "tuned_points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "config": json.dumps(r.get("config", {}))})


# =============================================================================
# MAIN
# =============================================================================


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    D = load_data()
    if USE_FAST_SIM and VERIFY_FAST_SIM:
        for n in QUBITS:
            verify_fast_sim(D, n)
    ev = Evaluator(D, OUT_DIR / "eval_cache.json")
    res_path = OUT_DIR / "tuned_points.json"
    R = {"settings": {"seeds": SEEDS, "split_seed": SPLIT_SEED, "readout": READOUT, "epochs": EPOCHS,
                      "rounds": TUNE_ROUNDS, "q_grid": Q_GRID, "e_grid": E_GRID},
         "linear": None, "esn": {}, "quantum": {}}

    def save():
        ev._save()
        res_path.write_text(json.dumps(R, indent=2))

    # 1. linear baseline (only the readout knob is tuned)
    k, vals = READOUT_GRID
    lin_cfg = {k: min(vals, key=lambda v: ev.val_score("linear", 0, {k: v}))}
    R["linear"] = ev.summary("linear", 0, lin_cfg)
    print(f"\nLinear baseline: test MSE {R['linear']['test_mse_mean']:.5f} ± {R['linear']['test_mse_std']:.5f}")

    # 2. ESN, tuned per N (independent of the observable set -> done once)
    cfg = dict(E_BASE)
    for N in ESN_SIZES:
        cfg = tune(ev, "esn", N, cfg, E_GRID, tag=f"ESN N={N}")
        R["esn"][str(N)] = ev.summary("esn", N, cfg)
        save()

    # 3. quantum, tuned per (observable set, n); warm start from n-1
    for obs in OBS_SETS:
        R["quantum"][obs] = {}
        cfg = dict(Q_BASE)
        for n in QUBITS:
            cfg = tune(ev, "quantum", n, cfg, Q_GRID, obs=obs, tag=f"Q {obs} n={n}")
            R["quantum"][obs][str(n)] = ev.summary("quantum", n, cfg, obs)
            p = R["quantum"][obs][str(n)]
            print(f"==> {obs} n={n}: {p['n_features']} features, "
                  f"test MSE {p['test_mse_mean']:.5f} ± {p['test_mse_std']:.5f}", flush=True)
            save()
        plot_obs_set(R, obs)  # each figure appears as soon as its observable set is done

    save()
    write_csv(R)
    plot_all(R)
    print(f"\nDone. Results in {OUT_DIR}")


if __name__ == "__main__":
    run()
