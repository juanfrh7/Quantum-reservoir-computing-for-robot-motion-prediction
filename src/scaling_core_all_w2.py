"""
scaling_core_all_w2.py
======================
Quantum reservoir (ALL weight<=2 Pauli observables) vs ESN, test MSE vs PHYSICAL SIZE
(qubits / neurons), every point tuned. One figure + one JSON with the tuned parameters per qubit number.

Every quantum feature is computed by YOUR code path, exactly as main.py does it:
    QiskitQuantumReservoir (reservoirs.py)  ->  utils.reservoir_features  ->  utils.train_readout (Adam)
There is NO re-implemented / "fast" simulator in this run: scaling_tuned.USE_FAST_SIM is forced to False,
so scaling_tuned.compute_features calls utils.reservoir_features(model, ...) on your QiskitQuantumReservoir.
scaling_tuned.py is only reused for bookkeeping (data loading as in main.py, evaluation cache, coordinate
descent, plot style). Nothing in main.py / utils.py / reservoirs.py / video_data.py / scaling_tuned.py is modified.

Speed-up without touching the model: the seeds of one configuration are simulated in parallel worker
processes (N_WORKERS), each running your QiskitQuantumReservoir.

Limits (from the current QiskitQuantumReservoir, all weight<=2 observables):
    n=8: ~1 h per simulation on a single core; n=9: ~5 h and ~4 GB RAM per model; n=10: ~18 GB RAM just to
    construct the model -> default QUBITS = 1..8.

Outputs (results/scaling_core_all_w2/; previous results are never overwritten):
    scaling_core_all_w2.png   test MSE vs qubits / neurons, mean +- std over seeds, linear baseline
    tuned_params.json         tuned parameters + val/test metrics per qubit number (and ESN sizes)
    eval_cache.json           every evaluation (resumable). On the first run it is initialised with the
                              ESN and linear-baseline entries of results/scaling_tuned/eval_cache.json
                              (computed with your ESNReservoir); quantum entries are NOT reused.

Usage (from src/, next to main.py and scaling_tuned.py):
    python scaling_core_all_w2.py --quick       # smoke test
    python scaling_core_all_w2.py --estimate    # time one simulation per n with your class, then exit
    python scaling_core_all_w2.py               # full run (stop/restart at any time)
    python scaling_core_all_w2.py --plot-only   # redraw the figure from tuned_params.json
"""
import argparse
import json
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

import scaling_tuned as ST          # reused for bookkeeping only; its run() is NOT executed
from utils import make_loader, reservoir_features

ST.USE_FAST_SIM = False             # -> every quantum feature comes from QiskitQuantumReservoir
ST.VERIFY_FAST_SIM = False


def _fast_sim_forbidden(*args, **kwargs):
    raise RuntimeError("FastQuantumSim was called; this script must only use QiskitQuantumReservoir.")


ST.FastQuantumSim = _fast_sim_forbidden   # hard guard (in memory only; scaling_tuned.py is not edited)

M = ST.M

# =============================================================================
# SETTINGS
# =============================================================================
SEEDS = [42, 1, 2]
QUBITS = list(range(1, 9))          # add 9 only if you accept ~5 h per simulation (x seeds / workers)
OBS = "all_w2"                                   # ST.OBS_SETS["all_w2"]: X,Y,Z + all 9 Pauli pairs
MAX_CONSTRUCTOR_GB = 8.0                         # skip n whose QiskitQuantumReservoir needs more RAM
N_WORKERS = int(os.environ.get("N_WORKERS", min(3, os.cpu_count() or 1)))  # parallel processes; 1 = none

# Search per qubit number:
#   "full"  = scaling_tuned.tune (every grid value of every parameter, ROUNDS_FULL passes) - as before
#   "local" = walk to neighbouring grid values while the validation MSE improves (ROUNDS_LOCAL passes)
#   "none"  = no search: evaluate the tuned settings of n-1 (used where one simulation takes ~1 h)
SEARCH_BY_N = {1: "full", 2: "full", 3: "full", 4: "local", 5: "local", 6: "local",
               7: "local", 8: "none", 9: "none"}
ROUNDS_FULL = 2
ROUNDS_LOCAL = 1

# ----- quantum: starting point (n=1) and BROADENED grid ----------------------
# Previous grid (scaling_tuned.py) and where the all_w2 optima sat:
#   input_gain [0.01 .. 1.0]  -> optimum 0.01-0.03 = LOW edge   => extended down to 0.001
#   damping    [0.01 .. 0.3]  -> optimum 0.01 at every n = LOW  => extended down, incl. 0
#   hz         [0.5 .. 4.0]   -> optimum 0.5 mostly = LOW edge  => extended down, incl. 0
#   Jz         [1, 3, 10]     -> optimum 3                       => finer around it
#   time_step  [1, 2]         -> optimum 2 = HIGH edge          => extended to 4
#   lr         [1e-3 .. 1e-2] -> optimum 3e-3 / 1e-2 (HIGH)     => extended to 3e-2, 1e-1
Q_START = dict(input_gain=0.01, damping=0.01, hz=0.5, Jz=3.0, time_step=2,
               interaction=M.INTERACTION, connectivity="all_to_all", lr=3e-3, ridge=0.0)
Q_GRID = OrderedDict([
    ("input_gain", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]),
    ("damping",    [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]),
    ("hz",         [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]),
    ("Jz",         [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]),
    ("time_step",  [1, 2, 3, 4]),
    ("lr",         [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
])

# ----- ESN -------------------------------------------------------------------
# Default: the SAME grid as scaling_tuned.py, so the ESN curve is read back from the copied cache
# (ESN entries were computed with your ESNReservoir) and is identical to your previous figure. BROADEN_ESN = True re-tunes it on a wider grid
# (fairer if the quantum grid is broadened, but costs hours).
ESN_SIZES = list(ST.ESN_SIZES)
BROADEN_ESN = False
E_GRID_BROAD = OrderedDict([
    ("input_gain",      [0.03, 0.1, 0.3, 1.0, 3.0]),
    ("leak_rate",       [0.05, 0.1, 0.3, 0.6, 1.0]),
    ("spectral_radius", [0.3, 0.5, 0.9, 1.2, 1.5]),
    ("lr",              [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
])
LINEAR_LRS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

OUT_DIR = M.RESULTS_DIR / "scaling_core_all_w2"
OLD_CACHE = M.RESULTS_DIR / "scaling_tuned" / "eval_cache.json"
FIG_NAME, JSON_NAME = "scaling_core_all_w2.png", "tuned_params.json"


# =============================================================================
# HELPERS
# =============================================================================
def n_observables(n):
    return 3 * n + 9 * n * (n - 1) // 2


def constructor_gb(n):
    """Peak RAM of QiskitQuantumReservoir.__init__ for all weight<=2 observables
    (list of complex128 matrices + np.stack copy + complex64 tensor)."""
    return n_observables(n) * 4 ** n * (16 + 16 + 8) / 1e9


def grid_for(n):
    g = OrderedDict(Q_GRID)
    if n == 1:
        g.pop("Jz")                    # no qubit pairs -> Jz has no effect
    return g


def local_walk(ev, size, start, grid, rounds, tag):
    """Coordinate search on neighbouring grid values only (cheap version of ST.tune)."""
    best = dict(start)
    best_s = ev.val_score("quantum", size, best, OBS)
    print(f"\n[{tag}] start val MSE={best_s:.5f}  {best}", flush=True)
    for _ in range(rounds):
        changed = False
        for p, values in grid.items():
            if best[p] not in values:
                values = sorted(set(values) | {best[p]})
            i = values.index(best[p])
            for step in (-1, +1):
                j, moved = i + step, False
                while 0 <= j < len(values):
                    cand = {**best, p: values[j]}
                    s = ev.val_score("quantum", size, cand, OBS)
                    print(f"  [{tag}] {p}={values[j]} -> {s:.5f}", flush=True)
                    if s < best_s:
                        best, best_s, moved, changed = cand, s, True, True
                        j += step
                    else:
                        break
                if moved:
                    break
        if not changed:
            break
    print(f"[{tag}] best val MSE={best_s:.5f}  {best}", flush=True)
    return best


def edge_flags(cfg, grid):
    out = []
    for p, values in grid.items():
        if len(values) > 1 and cfg.get(p) == values[0]:
            out.append(f"{p}={cfg[p]} at LOW edge" + (" (cannot go lower)" if values[0] == 0 else ""))
        elif len(values) > 1 and cfg.get(p) == values[-1]:
            out.append(f"{p}={cfg[p]} at HIGH edge")
    return out


def summary(ev, kind, size, cfg, obs=None):
    s = ev.summary(kind, size, cfg, obs)
    s["test_mse_per_seed"] = {str(seed): ev.run(kind, size, cfg, seed, obs)["test"]["loss"] for seed in SEEDS}
    return s


# =============================================================================
# PARALLEL SEEDS (each worker runs your QiskitQuantumReservoir via utils.reservoir_features)
# =============================================================================
_WORKER = {}


def _init_worker(quick, epochs):
    torch.set_num_threads(1)                 # avoid oversubscription: one core per worker
    ST.QUICK, ST.EPOCHS, ST.USE_FAST_SIM = quick, epochs, False
    _WORKER["D"] = ST.load_data()


def _features_job(job):
    kind, size, dyn, seed = job
    assert ST.USE_FAST_SIM is False
    F = ST.compute_features(_WORKER["D"], kind, size, dyn, seed)
    return seed, {k: (a.numpy(), b.numpy()) for k, (a, b) in F.items()}


class ParallelEvaluator(ST.Evaluator):
    """scaling_tuned.Evaluator, but the missing seeds of a quantum configuration are simulated in parallel."""

    def __init__(self, D, path, pool):
        super().__init__(D, path)
        self.pool = pool

    def _prefetch(self, kind, size, cfg, obs):
        if self.pool is None or kind != "quantum":
            return
        dyn = {k: v for k, v in cfg.items() if k not in ST.READOUT_KEYS}
        jobs = []
        for seed in SEEDS:
            key = json.dumps([ST.READOUT, ST.EPOCHS, ST.QUICK, kind, size, obs, seed, sorted(cfg.items())])
            fkey = json.dumps([kind, size, seed, sorted(dyn.items())])
            if key not in self.cache and fkey not in self.feat:
                jobs.append((kind, size, dyn, seed))
        if len(jobs) < 2:
            return
        for seed, F in self.pool.map(_features_job, jobs):
            fkey = json.dumps([kind, size, seed, sorted(dyn.items())])
            self.feat[fkey] = {k: (torch.from_numpy(a), torch.from_numpy(b)) for k, (a, b) in F.items()}
        while len(self.feat) > max(3, len(SEEDS)):
            self.feat.popitem(last=False)

    def val_score(self, kind, size, cfg, obs=None):
        self._prefetch(kind, size, cfg, obs)
        return super().val_score(kind, size, cfg, obs)

    def summary(self, kind, size, cfg, obs=None):
        self._prefetch(kind, size, cfg, obs)
        return super().summary(kind, size, cfg, obs)


# =============================================================================
# PLOT (right-hand panel of scaling_tuned_<obs>.png only)
# =============================================================================
def plot(R):
    q_ns = sorted(int(n) for n in R["quantum_all_w2"])
    q = [R["quantum_all_w2"][str(n)] for n in q_ns]
    e_Ns = sorted(int(N) for N in R["esn"])
    e = [R["esn"][str(N)] for N in e_Ns]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    if R.get("linear"):
        ax.axhline(R["linear"]["test_mse_mean"], color=ST.C_L, ls="--", lw=1.5, label="Linear (raw inputs)")
    if e:
        ST._curve(ax, e_Ns, e, ST.C_E, "ESN (N neurons, tuned per N)")
    if q:
        ST._curve(ax, q_ns, q, ST.C_Q, "Quantum (all weight<=2 Paulis, tuned per n)",
                  annotate=[f"n={n}" for n in q_ns])
    ST._style(ax, "physical size (qubits / neurons)")
    ax.set_ylabel("Test MSE")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"Quantum [all weight<=2 Paulis] vs ESN, every point tuned\n"
                 f"(mean ± std over seeds {R['settings']['seeds']}, readout=adam)", fontsize=11)
    lines = [f"n={n}: " + ", ".join(f"{k}={v}" for k, v in p["config"].items()
                                     if k not in ("connectivity", "interaction", "ridge"))
             + ("" if R["quantum_all_w2"][str(n)].get("search") != "none" else " (not re-tuned)")
             for n, p in zip(q_ns, q)]
    fig.text(0.5, 0.005, "tuned quantum settings  |  " + "   ".join(lines), ha="center", va="bottom",
             fontsize=6.2, color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = OUT_DIR / FIG_NAME
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}", flush=True)


# =============================================================================
# ESTIMATE MODE
# =============================================================================
def estimate(D):
    """Time your QiskitQuantumReservoir (utils.reservoir_features) on 2 training videos per n."""
    Xs, ys = D["data"]["train"]
    total_frames = sum(len(v) for split in D["data"].values() for v in split[0])
    print(f"Timing one simulation per n with QiskitQuantumReservoir (time_step={Q_START['time_step']}, "
          f"{total_frames} frames in total):")
    for n in QUBITS:
        if constructor_gb(n) > MAX_CONSTRUCTOR_GB:
            print(f"  n={n}: skipped, model construction needs ~{constructor_gb(n):.0f} GB of RAM")
            continue
        model = ST.build_quantum(D, n, Q_START, SEEDS[0])
        loader = make_loader(Xs[:2], ys[:2], M.BATCH_SIZE)
        t0 = time.time()
        reservoir_features(model, loader, M.BATCH_SIZE)
        dt = time.time() - t0
        frames = sum(len(v) for v in Xs[:2])
        per_sim = dt / frames * total_frames / 60
        print(f"  n={n}: ~{per_sim:7.1f} min per full simulation -> ~{per_sim * len(SEEDS) / max(1, N_WORKERS):7.1f} "
              f"min per configuration ({len(SEEDS)} seeds on {N_WORKERS} workers)", flush=True)
        del model


# =============================================================================
# MAIN
# =============================================================================
def main():
    global SEEDS, QUBITS, ESN_SIZES, Q_GRID, E_GRID_BROAD, OUT_DIR, LINEAR_LRS, SEARCH_BY_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test on a few videos")
    ap.add_argument("--estimate", action="store_true", help="time one simulation per n and exit")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from tuned_params.json")
    args = ap.parse_args()

    if args.quick:
        ST.QUICK, ST.EPOCHS = True, 5
        SEEDS, QUBITS, ESN_SIZES = [42, 1], [1, 2, 3], [5, 15]
        SEARCH_BY_N = {1: "full", 2: "local", 3: "none"}
        Q_GRID = OrderedDict((k, v[1::3]) for k, v in Q_GRID.items())
        E_GRID_BROAD = OrderedDict((k, v[::2]) for k, v in E_GRID_BROAD.items())
        LINEAR_LRS = [1e-2, 3e-2]
        OUT_DIR = M.RESULTS_DIR / "scaling_core_all_w2_quick"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Configure scaling_tuned's module-level settings that its functions read at call time.
    ST.SEEDS, ST.READOUT, ST.USE_FAST_SIM, ST.TUNE_ROUNDS, ST.OUT_DIR = SEEDS, "adam", False, ROUNDS_FULL, OUT_DIR
    res_path = OUT_DIR / JSON_NAME
    if args.plot_only:
        plot(json.loads(res_path.read_text()))
        return

    D = ST.load_data()
    if args.estimate:
        estimate(D)
        return

    cache_path = OUT_DIR / "eval_cache.json"
    if not cache_path.exists() and OLD_CACHE.exists() and not args.quick:
        old = json.loads(OLD_CACHE.read_text())                    # the old file is only read
        keep = {k: v for k, v in old.items() if json.loads(k)[3] in ("esn", "linear")}
        cache_path.write_text(json.dumps(keep))
        print(f"Initialised cache with the {len(keep)} ESN/linear entries of {OLD_CACHE} "
              f"(quantum entries not reused)", flush=True)
    pool = (ProcessPoolExecutor(N_WORKERS, initializer=_init_worker, initargs=(ST.QUICK, ST.EPOCHS))
            if N_WORKERS > 1 else None)
    ev = ParallelEvaluator(D, cache_path, pool)

    e_grid = E_GRID_BROAD if BROADEN_ESN else ST.E_GRID
    R = json.loads(res_path.read_text()) if res_path.exists() else {}
    R["settings"] = {
        "seeds": SEEDS, "split_seed": ST.SPLIT_SEED, "readout": "adam (utils.train_readout)",
        "epochs": ST.EPOCHS, "observables": "all weight<=2 Paulis: X,Y,Z on every qubit + all 9 Pauli pairs "
        "on every qubit pair", "qubits": QUBITS, "search_by_n": SEARCH_BY_N, "rounds_full": ROUNDS_FULL,
        "rounds_local": ROUNDS_LOCAL, "quantum_grid": Q_GRID, "esn_grid": e_grid, "linear_lrs": LINEAR_LRS,
        "simulation": "QiskitQuantumReservoir (reservoirs.py) via utils.reservoir_features; no fast simulator",
    }
    R.setdefault("esn", {})
    R.setdefault("quantum_all_w2", {})

    def save():
        ev._save()
        res_path.write_text(json.dumps(R, indent=2) + "\n")

    # 1. Linear baseline (only the learning rate is tuned).
    lin_cfg = {"lr": min(LINEAR_LRS, key=lambda v: ev.val_score("linear", 0, {"lr": v}))}
    R["linear"] = summary(ev, "linear", 0, lin_cfg)
    save()
    print(f"\nLinear baseline: test MSE {R['linear']['test_mse_mean']:.5f} ± {R['linear']['test_mse_std']:.5f}",
          flush=True)

    # 2. ESN, tuned per N with scaling_tuned.tune (same algorithm as the previous figure).
    cfg = dict(ST.E_BASE)
    for N in ESN_SIZES:
        cfg = ST.tune(ev, "esn", N, cfg, e_grid, tag=f"ESN N={N}")
        R["esn"][str(N)] = summary(ev, "esn", N, cfg)
        R["esn"][str(N)]["at_grid_edge"] = edge_flags(cfg, e_grid)
        save()
    plot(R)

    # 3. Quantum, all weight<=2 observables, tuned per n (warm start from n-1).
    cfg = dict(Q_START)
    for n in QUBITS:
        if constructor_gb(n) > MAX_CONSTRUCTOR_GB:
            print(f"\nSkipping n={n}: QiskitQuantumReservoir needs ~{constructor_gb(n):.0f} GB of RAM "
                  f"to build {n_observables(n)} dense observables (limit MAX_CONSTRUCTOR_GB={MAX_CONSTRUCTOR_GB}).",
                  flush=True)
            continue
        t0 = time.time()
        grid = grid_for(n)
        mode = SEARCH_BY_N.get(n, "local")
        if mode == "full":
            cfg = ST.tune(ev, "quantum", n, cfg, grid, obs=OBS, tag=f"Q all_w2 n={n}")
        elif mode == "local":
            cfg = local_walk(ev, n, cfg, grid, ROUNDS_LOCAL, tag=f"Q all_w2 n={n}")
        p = summary(ev, "quantum", n, cfg, OBS)
        p.update(search=mode, n_observables=n_observables(n), at_grid_edge=edge_flags(cfg, grid),
                 minutes=round((time.time() - t0) / 60, 1))
        R["quantum_all_w2"][str(n)] = p
        save()
        plot(R)
        print(f"==> n={n}: test MSE {p['test_mse_mean']:.5f} ± {p['test_mse_std']:.5f} | {cfg}"
              + (f" | EDGE: {p['at_grid_edge']}" if p["at_grid_edge"] else ""), flush=True)

    save()
    plot(R)
    if pool is not None:
        pool.shutdown()
    print(f"\nDone. Results in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
