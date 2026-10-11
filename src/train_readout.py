# Train the policy that the Qollab demo (main.py at the top of the repository) runs as circuits.
#
# The reservoir is the 6-qubit quantum reservoir of the size sweep: same settings, couplings and input
# weights as sweep.py. The readout sees the current observation next to the reservoir features, and is a
# ridge fit on features with shot noise added, so that it still works when the features are measured.
#
# NumPy only. Run it in src/:  python train_readout.py   (writes lift_policy.npz at the top of the repository)

import numpy as np
from sweep import (DATASET_PATH, INPUT_STD_EPS, QUANTUM_CONFIG, SEED, SRC_DIR, TRAIN_FRACTION, VAL_FRACTION,
                   quantum_features, quantum_reservoir, ridge_fit, scores)
from video_data import fit_input_normalization, load_dataset, split_video_indices

SAVE_PATH = SRC_DIR.parent / "lift_policy.npz"
NUM_QUBITS = 6
SHOTS = 1000      # shots per measurement basis that the saved readout is trained to tolerate


def train_policy(dataset_path=DATASET_PATH):
    # prints the scores of five readouts and returns the contents of lift_policy.npz
    X, Y = load_dataset(dataset_path)
    train, val, test = split_video_indices(len(X), SEED, TRAIN_FRACTION, VAL_FRACTION)
    mean, std = fit_input_normalization([X[i] for i in train], INPUT_STD_EPS)
    Z = [(video - mean) / std for video in X]                       # standardised observations
    splits = lambda videos: [np.concatenate([videos[i] for i in part]) for part in (train, val, test)]

    # the reservoir of sweep.py, with input weights that act on the standardised observations
    config = {**QUANTUM_CONFIG, "num_qubits": NUM_QUBITS, "input_dim": X[0].shape[1], "output_dim": Y[0].shape[1]}
    gain = config.pop("input_gain")
    # main.py builds exactly this reservoir as a circuit
    assert (config["interaction"], config["connectivity"], config["time_step"], config["reservoir_time"]) == ("XX", "all_to_all", 1, 1.0)
    assert (tuple(config["single_observables"]), tuple(config["pair_observables"])) == (("Z", "X"), ("ZZ", "XX"))
    w_in, _, J, *operators = quantum_reservoir(config)
    w_in.weight = w_in.weight * gain
    F = [quantum_features(z, config, w_in, *operators) for z in Z]

    # measuring a feature with SHOTS shots adds noise of variance (1 - feature^2) / SHOTS
    rng = np.random.default_rng(SEED)
    noisy = [f + rng.normal(size=f.shape) * np.sqrt(np.clip(1 - f**2, 0, 1) / SHOTS) for f in F]

    Y_train, Y_val, Y_test = splits(Y)
    print(f"{len(X)} demonstrations ({len(train)}/{len(val)}/{len(test)}), {sum(map(len, X))} frames\n")
    print(f"{'readout on':<54} {'inputs':>6} {'ridge':>7} {'test loss':>10} {'at ' + str(SHOTS) + ' shots':>14}")
    rows = {"nothing (the average action)": ([z[:, :0] for z in Z], None),
            "current observation": (Z, None),
            "reservoir features": (F, None),
            "observation + reservoir features": ([np.hstack(pair) for pair in zip(Z, F)], None),
            "observation + reservoir features, trained with noise": ([np.hstack(pair) for pair in zip(Z, F)],
                                                                     [np.hstack(pair) for pair in zip(Z, noisy)])}
    for name, (exact, measured) in rows.items():
        A_train, A_val, A_test = splits(exact)
        N_train, N_val, N_test = splits(measured or exact)         # what the readout is fitted and selected on
        W, b, selected = ridge_fit(N_train, Y_train, N_val, Y_val)
        with_shots = f"{scores(N_test, Y_test, W, b)['loss']:>14.4f}" if measured else ""
        print(f"{name:<54} {A_train.shape[1]:>6} {selected['ridge']:>7g} {scores(A_test, Y_test, W, b)['loss']:>10.4f} {with_shots}")

    # the last row is the policy for the demo, with the held-out demonstration of median error to replay
    errors = [scores(np.hstack([Z[i], F[i]]), Y[i], W, b)["loss"] for i in test]
    demo = test[np.argsort(errors)[len(test) // 2]]
    return dict(obs_mean=mean, obs_std=std, w_in=w_in.weight, b_in=w_in.bias, J=J, hz=float(config["hz"]),
                damping=float(config["damping"]), readout_W=W, readout_b=b,
                demo_obs=Z[demo].astype(np.float32), demo_actions=Y[demo].astype(np.float32))


if __name__ == "__main__":
    policy = train_policy()
    np.savez_compressed(SAVE_PATH, **policy)
    print(f"\nSaved {SAVE_PATH.name}: readout weights up to {np.abs(policy['readout_W']).max():.2f}, "
          f"a demonstration of {len(policy['demo_obs'])} frames to replay")
