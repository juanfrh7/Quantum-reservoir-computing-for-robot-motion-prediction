"""Train classical and quantum readouts on a synthetic next-step dataset."""
import torch
from torch import nn
from reservoirs import ESNReservoir, QiskitQuantumReservoir
from utils import (
    evaluate,
    make_loader,
    make_synthetic_dataset,
    reservoir_features,
    train_epoch,
)

# All experiment settings are defined here.
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 42
NUM_SAMPLES = 1000
ESN_SIZE = 8
NUM_QUBITS = 3
LEAK_RATE = 0.3
hz = 1.0
Jz = 10

time_steps = 30
initial_vector_dim = 10 + 3 + 4 + 2
final_vector_dim = initial_vector_dim


def main():
    torch.manual_seed(SEED)

    # 1. Generate histories and next-step targets, then split by sample.
    X, y = make_synthetic_dataset(
        NUM_SAMPLES,
        time_steps,
        initial_vector_dim,
        seed=SEED,
    )
    permutation = torch.randperm(
        NUM_SAMPLES,
        generator=torch.Generator().manual_seed(SEED),
    )
    train_end = int(0.7 * NUM_SAMPLES)
    val_end = int(0.85 * NUM_SAMPLES)
    train_indices = permutation[:train_end]
    val_indices = permutation[train_end:val_end]
    test_indices = permutation[val_end:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    print(
        f"Synthetic next-step data: X={tuple(X.shape)}, y={tuple(y.shape)}; "
        f"splits={len(train_indices)}/{len(val_indices)}/{len(test_indices)}",
        flush=True,
    )

    # 2. Batch the generated sequences.
    train_loader = make_loader(X_train, y_train, BATCH_SIZE)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE)
    test_loader = make_loader(X_test, y_test, BATCH_SIZE)
    input_dim = train_loader.dataset.tensors[0].shape[-1]
    output_dim = train_loader.dataset.tensors[1].shape[-1]

    # 3. Define the two reservoirs. Their inputs and dynamics are fixed.
    models = {
        "Classical ESN": ESNReservoir({
            "input_dim": input_dim,
            "output_dim": output_dim,
            "reservoir_size": ESN_SIZE,
            "leak_rate": LEAK_RATE,
            "spectral_radius": 0.9,
            "time_step": 1,
        }),
        "Quantum": QiskitQuantumReservoir(
            input_dim=input_dim,
            output_dim=output_dim,
            num_qubits=NUM_QUBITS,
            hz=hz,
            Jz=Jz,
            interaction="XX",
            single_observables=("Z", "X"),
            pair_observables=("ZZ", "XX"),
            damping=LEAK_RATE,
            coupling_seed=SEED,
        ),
    }
    loss_fn = nn.MSELoss()
    results = {}

    for name, model in models.items():
        # 4. Only w_out is optimized. Simulate fixed features once.
        model.w_in.requires_grad_(False)
        optimizer = torch.optim.Adam(
            model.w_out.parameters(), lr=LEARNING_RATE
        )
        print(f"\n{name}: computing reservoir features", flush=True)
        train = reservoir_features(model, train_loader, BATCH_SIZE, shuffle=True)
        val = reservoir_features(model, val_loader, BATCH_SIZE)
        test = reservoir_features(model, test_loader, BATCH_SIZE)

        # 5. Train for exactly EPOCHS, using only the training split.
        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train, optimizer, loss_fn)
            print(f"{name} | epoch {epoch}/{EPOCHS} | train loss={loss:.6g}")

        # 6. Evaluate the final readout on validation and test splits.
        results[name] = {
            "validation": evaluate(model, val),
            "test": evaluate(model, test),
        }

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