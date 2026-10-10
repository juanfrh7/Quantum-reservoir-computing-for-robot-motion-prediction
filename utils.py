"""Batching, fixed reservoir features, and readout training. No dataset import."""
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_synthetic_dataset(
    n_samples,
    time_steps,
    input_dim,
    seed,
    noise_std=0.03,
):
    """Return histories and their next vectors with shapes (N,T,D) and (N,D)."""
    if min(n_samples, time_steps, input_dim) < 1:
        raise ValueError("n_samples, time_steps, and input_dim must be positive.")
    if noise_std < 0:
        raise ValueError("noise_std must be nonnegative.")

    generator = torch.Generator().manual_seed(seed)
    sequence = torch.empty(n_samples, time_steps + 1, input_dim)
    sequence[:, 0] = torch.rand(
        n_samples,
        input_dim,
        generator=generator,
    ) * 1.6 - 0.8
    for step in range(time_steps):
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

    return sequence[:, :-1], sequence[:, -1]


def make_loader(X, y, batch_size, shuffle=False):
    """Accept X=(N,input_dim) or (N,context,input_dim), y=(N,output_dim)."""
    X = torch.as_tensor(X, dtype=torch.float32).detach().cpu()
    y = torch.as_tensor(y, dtype=torch.float32).detach().cpu()
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    if (
        X.ndim not in (2, 3)
        or y.ndim != 2
        or len(X) != len(y)
        or len(X) == 0
    ):
        raise ValueError("Expected nonempty matching X and y with the documented shapes.")
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def reservoir_features(model, loader, batch_size, shuffle=False):
    """Compute fixed features once. Each sample/history starts at zero."""
    features, targets = [], []
    device = model.w_in.weight.device
    for X, y in loader:
        X = X.to(device)
        if X.ndim == 2:
            X = X.unsqueeze(1)
        model.initialize(len(X))
        for t in range(X.shape[1]):
            last = torch.cat([
                model.reservoir_step(X[:, t]) for _ in range(model.time_step)
            ], dim=-1)
        features.append(last.cpu())
        targets.append(y)
    model.reservoir_state = None
    return DataLoader(
        TensorDataset(torch.cat(features), torch.cat(targets)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_epoch(model, loader, optimizer, loss_fn):
    """Optimize only the readout on cached reservoir features."""
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
def evaluate(model, loader):
    """MSE loss and MAE averaged over samples and target coordinates."""
    model.eval()
    device = model.w_out.weight.device
    squared, absolute, count = 0.0, 0.0, 0
    for features, targets in loader:
        error = model.w_out(features.to(device)) - targets.to(device)
        squared += error.square().sum().item()
        absolute += error.abs().sum().item()
        count += error.numel()
    return {"loss": squared / count, "mae": absolute / count}
