"""Batching, fixed reservoir features, and readout training. No dataset import."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from video_data import as_videos, batch_indices, frame_metrics, validate_pairs


def make_synthetic_dataset(
    n_samples,
    time_steps,
    input_dim,
    seed,
    noise_std=0.03,
):
    """Return synthetic input videos (N,T,D); supply decision targets separately."""
    # Use a dedicated seeded generator for both initial states and frame noise.
    if min(n_samples, time_steps, input_dim) < 1:
        raise ValueError("Dimensions must be positive.")
    if noise_std < 0:
        raise ValueError("noise_std must be nonnegative.")

    generator = torch.Generator().manual_seed(seed)
    sequence = torch.empty(n_samples, time_steps, input_dim)
    sequence[:, 0] = torch.rand(
        n_samples,
        input_dim,
        generator=generator,
    ) * 1.6 - 0.8
    for step in range(time_steps - 1):
        current = sequence[:, step]
        noise = torch.randn(
            n_samples,
            input_dim,
            generator=generator,
        ) * noise_std
        sequence[:, step + 1] = (
            0.65 * current
            + 0.2 * torch.roll(current, shifts=1, dims=-1)
            + 0.1 * torch.sin(torch.pi * current)
            + noise
        )

    return sequence


def _videos(X):
    """Normalize videos to a list of CPU float32 arrays."""
    # A common dtype keeps loaders independent of the caller's array precision.
    return as_videos(X, dtype=np.float32)


def make_loader(X, y, batch_size, shuffle=False):
    """Batch input videos and same-frame decision targets without shifting."""
    # Match video and frame dimensions while allowing different input/output widths.
    X, y = validate_pairs(X, y, dtype=np.float32)
    if shuffle:
        order = torch.randperm(len(X)).tolist()
        X, y = [X[i] for i in order], [y[i] for i in order]
    dataset = [(torch.from_numpy(video), torch.from_numpy(target))
               for video, target in zip(X, y)]
    return DataLoader(dataset, batch_sampler=list(batch_indices(X, batch_size)))


@torch.no_grad()
def reservoir_features(model, loader, batch_size, shuffle=False):
    """Cache every frame's features/decision targets; reset state for each video."""
    # Flatten videos and frames in the same order for both cached arrays.
    features, targets = [], []
    device = model.w_in.weight.device
    for X, y in loader:
        X = X.to(device)
        if X.shape[-1] != model.input_dim or y.shape[-1] != model.output_dim:
            raise ValueError("Model dimensions must match input features and decision targets.")
        model.initialize(len(X))
        steps = []
        for t in range(X.shape[1]):
            steps.append(torch.cat([
                model.reservoir_step(X[:, t]) for _ in range(model.time_step)
            ], dim=-1))
        features.append(torch.stack(steps, dim=1).flatten(0, 1).cpu())
        targets.append(y.flatten(0, 1))
    model.reservoir_state = None
    return DataLoader(
        TensorDataset(torch.cat(features), torch.cat(targets)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_epoch(model, loader, optimizer, loss_fn):
    """Optimize only the readout on cached reservoir features."""
    # Weight the reported loss by target count so partial batches count correctly.
    model.train()
    device = model.w_out.weight.device
    total, count = 0.0, 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model.w_out(features)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        total += loss.item() * targets.numel()
        count += targets.numel()
    return total / count


@torch.no_grad()
def fit_readout(model, features, targets, ridge=0.0):
    """Fit the optimal linear readout on cached training features in float64.

    Minimize squared error plus ridge * ||weight||²; do not penalize the bias.
    Copy the solution back to the model's original dtype and device.
    """
    # Solve on CPU in float64, then copy coefficients to the original model device.
    features = features.detach().cpu().numpy().astype(np.float64)
    targets = targets.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative.")
    if (features.ndim != 2 or features.shape[1] != model.w_out.in_features
            or targets.shape != (len(features), model.output_dim) or len(features) == 0
            or not np.isfinite(features).all() or not np.isfinite(targets).all()):
        raise ValueError("Invalid training features or decision targets.")
    design = np.column_stack((features, np.ones(len(features))))
    if ridge:
        penalty = np.sqrt(ridge) * np.eye(design.shape[1])[:-1]
        design = np.vstack((design, penalty))
        targets = np.vstack((targets, np.zeros((len(penalty), model.output_dim))))
    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    model.w_out.weight.copy_(torch.as_tensor(
        coefficients[:-1].T, dtype=model.w_out.weight.dtype, device=model.w_out.weight.device
    ))
    model.w_out.bias.copy_(torch.as_tensor(
        coefficients[-1], dtype=model.w_out.bias.dtype, device=model.w_out.bias.device
    ))


def export_numpy_checkpoint(model, path, model_type, model_config, hyperparameters=None):
    """Export weights, model configuration, and optional experiment settings."""
    # Include fixed reservoir parameters as well as the fitted readout.
    if model_type not in ("esn", "quantum"):
        raise ValueError("model_type must be 'esn' or 'quantum'.")
    names = ["w_in.weight", "w_in.bias", "w_out.weight", "w_out.bias",
             "w_res" if model_type == "esn" else "J"]
    state = model.state_dict()
    parameters = {name: state[name].detach().cpu().numpy() for name in names}
    # Save settings as directly readable NumPy scalars and inside the JSON metadata.
    hyperparameters = {} if hyperparameters is None else dict(hyperparameters)
    parameters.update({name: np.array(value) for name, value in hyperparameters.items()})
    metadata = json.dumps({
        "format_version": 1, "model_type": model_type, "model_config": model_config,
        "hyperparameters": hyperparameters,
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        np.savez_compressed(output, metadata=np.array(metadata), **parameters)


@torch.no_grad()
def predict(model, X, batch_size=64):
    """Accept videos (N,T,D), return (N,T,output_dim) decision predictions."""
    # Evaluation needs no gradients, and each full-video call resets its state.
    dense = isinstance(X, (np.ndarray, torch.Tensor))
    videos = _videos(X)
    model.eval()
    device = model.w_out.weight.device
    predictions = [None] * len(videos)
    try:
        for indices in batch_indices(videos, batch_size):
            batch = torch.from_numpy(np.stack([videos[i] for i in indices])).to(device)
            values = model(batch).cpu()
            for row, i in enumerate(indices):
                predictions[i] = values[row]
    finally:
        model.reservoir_state = None
    return torch.stack(predictions) if dense else predictions


@torch.no_grad()
def evaluate(model, X, y, batch_size=64):
    """MSE/MAE against supplied decision targets at each frame."""
    # Validate exact target alignment before aggregating over every coordinate.
    predictions = predict(model, X, batch_size)
    return frame_metrics(predictions, y)
