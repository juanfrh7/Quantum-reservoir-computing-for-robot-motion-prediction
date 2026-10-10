"""NumPy-only feature extraction, optimal linear readout fitting, and metrics."""
import numpy as np

from video_data import as_videos, batch_indices, frame_metrics, validate_pairs


def _histories(model, X, batch_size):
    """Validate dense or variable-length videos against the model's input width."""
    # Reject empty or nonfinite videos before advancing any reservoir state.
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be positive.")
    return as_videos(X, model.input_dim, "inputs")


def reservoir_features(model, X, batch_size=64):
    """Return dense features or a list of (T_i,F) arrays matching the input form."""
    # Batch only the video axis so each video retains its chronological frame order.
    dense = isinstance(X, np.ndarray)
    videos = _histories(model, X, batch_size)
    features = [None] * len(videos)
    try:
        for indices in batch_indices(videos, batch_size):
            batch = np.stack([videos[i] for i in indices])
            model.initialize(len(batch))
            steps = [model.features(batch[:, t]) for t in range(batch.shape[1])]
            values = np.stack(steps, axis=1)
            for row, i in enumerate(indices):
                features[i] = values[row]
    finally:
        model.reservoir_state = None
    return np.stack(features) if dense else features


def _targets(y, n_samples, output_dim):
    """Normalize scalar targets and validate a flat decision-target matrix."""
    # A one-dimensional target array represents one output coordinate per sample.
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape != (n_samples, output_dim) or not np.isfinite(y).all():
        raise ValueError(f"Expected finite y with shape ({n_samples}, {output_dim}).")
    return y


def fit_readout(model, features, y, ridge=0.0):
    """Minimize ||F W.T + b - y||² + ridge * ||W||²; leave bias unpenalized.

    ridge=0 gives an optimal least-squares readout (minimum-norm if singular).
    Uses an augmented least-squares solve for stability, without matrix inversion.
    """
    # Flatten video and frame axes together, preserving feature/target alignment.
    if isinstance(features, (list, tuple)):
        features, y = validate_pairs(
            features, y, model.w_out.weight.shape[1], model.output_dim
        )
        features, y = np.concatenate(features), np.concatenate(y)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 3:
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (*features.shape[:2], model.output_dim):
            raise ValueError("Video targets must match (N,T,output_dim) features.")
        features = features.reshape(-1, features.shape[-1])
        y = y.reshape(-1, model.output_dim)
    if (features.ndim != 2 or len(features) == 0
            or features.shape[1] != model.w_out.weight.shape[1]
            or not np.isfinite(features).all()):
        raise ValueError("Invalid reservoir feature matrix.")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative.")
    y = _targets(y, len(features), model.output_dim)
    design = np.column_stack((features, np.ones(len(features))))
    if ridge:
        penalty = np.sqrt(ridge) * np.eye(design.shape[1])[:-1]
        design = np.vstack((design, penalty))
        y = np.vstack((y, np.zeros((len(penalty), model.output_dim))))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    model.w_out.weight = coefficients[:-1].T.copy()
    model.w_out.bias = coefficients[-1].copy()
    return model


def predict(model, X, batch_size=64):
    """Return aligned decisions as a dense array or a list of (T_i,output_dim) arrays."""
    # Reservoir features carry information from the current and preceding frames.
    features = reservoir_features(model, X, batch_size)
    if isinstance(features, list):
        return [model.readout(video) for video in features]
    return model.readout(features)


def evaluate(model, X, y, batch_size=64):
    """Score predictions at t against supplied decision targets y[:,t,:]."""
    # Require an exact target shape to prevent accidental NumPy broadcasting.
    validate_pairs(X, y, model.input_dim, model.output_dim)
    predictions = predict(model, X, batch_size)
    return frame_metrics(predictions, y)


def make_synthetic_dataset(n_samples, time_steps, input_dim, seed=42, noise_std=0.03):
    """Return synthetic input videos (N,T,D); supply decision targets separately."""
    # A local random generator keeps synthetic data independent of global RNG state.
    if min(n_samples, time_steps, input_dim) < 1 or noise_std < 0:
        raise ValueError("Dimensions must be positive and noise_std nonnegative.")
    rng = np.random.default_rng(seed)
    sequence = np.empty((n_samples, time_steps, input_dim))
    sequence[:, 0] = rng.uniform(-0.8, 0.8, (n_samples, input_dim))
    for t in range(time_steps - 1):
        current = sequence[:, t]
        sequence[:, t + 1] = (
            0.65 * current + 0.2 * np.roll(current, 1, axis=-1)
            + 0.1 * np.sin(np.pi * current)
            + rng.normal(0, noise_std, current.shape)
        )
    return sequence
