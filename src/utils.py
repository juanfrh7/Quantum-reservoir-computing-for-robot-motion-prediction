"""Batching, fixed reservoir features, and readout training. No dataset import."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from video_data import as_videos, batch_indices, frame_metrics, validate_pairs



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
def cached_metrics(model, loader):
    """Measure MSE and MAE over actual frames using the current readout weights."""
    # Accumulate in float64 after prediction; partial batches have their true weight.
    model.eval()
    device = model.w_out.weight.device
    squared, absolute, count = 0.0, 0.0, 0
    for features, targets in loader:
        residual = model.w_out(features.to(device)).double() - targets.to(device).double()
        squared += residual.square().sum().item()
        absolute += residual.abs().sum().item()
        count += targets.numel()
    return {"loss": squared / count, "mae": absolute / count}


def train_readout(model, train, validation, epochs=100, learning_rate=1e-3,
                  betas=(0.9, 0.999), eps=1e-8, seed=42, label="Model", verbose=True):
    """Train w_out with Adam, record epoch metrics, and restore the best validation weights."""
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    # Shuffle cached training frames reproducibly; validation never updates parameters.
    model.w_in.requires_grad_(False)
    shuffled = DataLoader(train.dataset, batch_size=train.batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.w_out.parameters(), lr=learning_rate, betas=betas, eps=eps)
    history = {key: [] for key in ("epoch", "train_loss", "val_loss", "train_mae", "val_mae")}
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(epochs + 1):
        if epoch:
            train_epoch(model, shuffled, optimizer, torch.nn.MSELoss())
        # Score the weights after each complete epoch, including the initial baseline.
        training_metrics = cached_metrics(model, train)
        validation_metrics = cached_metrics(model, validation)
        if not all(np.isfinite(value) for value in (*training_metrics.values(), *validation_metrics.values())):
            raise ValueError("Nonfinite training metrics; check data and learning rate.")
        history["epoch"].append(epoch)
        for split, metrics in (("train", training_metrics), ("val", validation_metrics)):
            for metric, value in metrics.items():
                history[f"{split}_{metric}"].append(value)
        if validation_metrics["loss"] < best_loss:
            best_loss, best_epoch = validation_metrics["loss"], epoch
            best_state = deepcopy(model.w_out.state_dict())
        if verbose:
            print(f"{label} | epoch {epoch}/{epochs} | "
                  f"train loss (MSE)={training_metrics['loss']:.6g} | "
                  f"train MAE={training_metrics['mae']:.6g} | "
                  f"val loss (MSE)={validation_metrics['loss']:.6g} | "
                  f"val MAE={validation_metrics['mae']:.6g}", flush=True)
    model.w_out.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch,
            "validation": cached_metrics(model, validation)}


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
