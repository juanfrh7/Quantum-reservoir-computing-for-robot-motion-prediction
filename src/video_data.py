"""NumPy-only validation, storage, batching, and metrics for video collections."""
from pathlib import Path
import pickle
import numpy as np


def as_videos(data, width=None, name="videos", dtype=np.float64):
    """Normalize a dense (N,T,D) array or list of (T_i,D) arrays to a list."""
    # Keep frames within each video intact and require a shared vector width.
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        if data.ndim != 3 or data.dtype == object:
            raise ValueError(f"{name} must be a numeric (N,T,D) array or list of (T_i,D) arrays.")
    elif not isinstance(data, (list, tuple)):
        raise ValueError(f"{name} must be a (N,T,D) array or list of (T_i,D) arrays.")
    if len(data) == 0:
        raise ValueError(f"{name} must contain at least one video.")
    videos = []
    for i, item in enumerate(data):
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        video = np.asarray(item, dtype=dtype)
        if video.ndim != 2 or min(video.shape) < 1 or not np.isfinite(video).all():
            raise ValueError(f"{name}[{i}] must be a nonempty finite (frames, vector_dim) array.")
        if width is None:
            width = video.shape[1]
        if video.shape[1] != width:
            raise ValueError(f"{name}[{i}] has vector width {video.shape[1]}; expected {width}.")
        videos.append(video)
    return videos


def validate_pairs(X, y, input_dim=None, output_dim=None, dtype=np.float64):
    """Validate video inputs and same-frame targets, including each frame count."""
    # Matching total frame counts alone does not establish per-video alignment.
    videos = as_videos(X, input_dim, "inputs", dtype)
    targets = as_videos(y, output_dim, "targets", dtype)
    if len(videos) != len(targets):
        raise ValueError("Inputs and targets must contain the same number of videos.")
    for i, (video, target) in enumerate(zip(videos, targets)):
        if len(video) != len(target):
            raise ValueError(f"Video {i}: input and target frame counts must match.")
    return videos, targets


def batch_indices(videos, batch_size):
    """Yield batches of video indices grouped by equal frame count."""
    # Equal-length groups preserve vectorized updates without padding any frames.
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    groups = {}
    for i, video in enumerate(videos):
        groups.setdefault(len(video), []).append(i)
    for indices in groups.values():
        for start in range(0, len(indices), batch_size):
            yield indices[start:start + batch_size]


def fit_input_normalization(X_train, min_std=1e-8):
    """Estimate per-feature mean and standard deviation from real training frames."""
    # Include each actual frame once, without validation/test data or padding.
    if not np.isfinite(min_std) or min_std <= 0:
        raise ValueError("min_std must be finite and positive.")
    frames = np.concatenate(as_videos(X_train))
    mean, std = frames.mean(axis=0), frames.std(axis=0)
    # Constant features become zero after centering; avoid division by a tiny std.
    return mean, np.where(std < min_std, 1.0, std)


def load_dataset(path, dtype=np.float64):
    """Load the notebook's pickle and return aligned (T_i,D) input/target arrays."""
    # Time-step lists describe alignment; only the value matrices enter the reservoirs.
    with Path(path).open("rb") as source:
        dataset = pickle.load(source)
    if not isinstance(dataset, dict) or not {"inputs", "targets"} <= dataset.keys():
        raise ValueError("Dataset must contain 'inputs' and 'targets'.")
    inputs, targets = dataset["inputs"], dataset["targets"]
    if (not isinstance(inputs, (list, tuple)) or not isinstance(targets, (list, tuple))
            or len(inputs) != len(targets)):
        raise ValueError("Inputs and targets must be lists with the same number of videos.")
    X, y = [], []
    for i, (video, target) in enumerate(zip(inputs, targets)):
        if (not isinstance(video, (list, tuple)) or len(video) != 2
                or not isinstance(target, (list, tuple)) or len(target) != 2):
            raise ValueError(f"Video {i}: expected [time_steps, values] for inputs and targets.")
        times, target_times = np.asarray(video[0]), np.asarray(target[0])
        if (times.ndim != 1 or not np.array_equal(times, target_times)
                or len(times) != len(video[1]) or len(times) != len(target[1])):
            raise ValueError(f"Video {i}: input and target time steps must match their frames.")
        X.append(video[1])
        y.append(target[1])
    input_dim = len(dataset["input_keys"]) if "input_keys" in dataset else None
    output_dim = len(dataset["target_keys"]) if "target_keys" in dataset else None
    return validate_pairs(X, y, input_dim, output_dim, dtype)


def split_video_indices(n_videos, seed=42, train_fraction=0.7, val_fraction=0.15):
    """Return reproducible video splits using NumPy; defaults are 70/15/15."""
    # Training and standalone evaluation recreate the same held-out videos.
    if n_videos < 10:
        raise ValueError("At least 10 videos are required for train/validation/test splits.")
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 - train_fraction):
        raise ValueError("Train and validation fractions must leave a positive test fraction.")
    indices = np.random.default_rng(seed).permutation(n_videos)
    train_end = int(train_fraction * n_videos)
    val_end = int((train_fraction + val_fraction) * n_videos)
    if not 0 < train_end < val_end < n_videos:
        raise ValueError("Split fractions must give at least one video per split.")
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def save_videos(path, videos):
    """Save one numeric (T_i,D) array per video in a pickle-free .npz file."""
    # Zero-padded numeric keys preserve video identity independently of length.
    videos = as_videos(videos, dtype=None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        np.savez_compressed(output, **{
            f"video_{i:06d}": video for i, video in enumerate(videos)
        })


def load_videos(path):
    """Load per-video .npz arrays or a legacy dense (N,T,D) .npy dataset."""
    # Never require object arrays or pickle to represent variable frame counts.
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        as_videos(loaded)
        return loaded
    with loaded as archive:
        keys = [f"video_{i:06d}" for i in range(len(archive.files))]
        if not keys or set(keys) != set(archive.files):
            raise ValueError("Video archives must contain consecutive video_000000, video_000001, ... arrays.")
        videos = [archive[key].copy() for key in keys]
    as_videos(videos)
    return videos


def frame_metrics(predictions, targets):
    """Return MSE/MAE weighted by actual frame coordinates across all videos."""
    # Sum errors before dividing so short and long videos contribute per frame.
    predictions, targets = validate_pairs(predictions, targets)
    squared, absolute, count = 0.0, 0.0, 0
    for prediction, target in zip(predictions, targets):
        if prediction.shape != target.shape:
            raise ValueError("Decision target width must match the trained model's output_dim.")
        error = prediction - target
        squared += float(np.sum(error ** 2))
        absolute += float(np.sum(np.abs(error)))
        count += error.size
    return {"loss": squared / count, "mae": absolute / count}
