"""Stateful ESN and density-matrix quantum reservoirs using only NumPy.

Linear weights use (output_dim, input_dim), as in the original reservoirs.
Pass videos (N,T,D) to predict (N,T,output_dim) decision vectors, or pass (N,D)
to update a stream.
No training framework or quantum SDK is required.
"""
import json
from pathlib import Path
import numpy as np

from video_data import as_videos


class Linear:
    def __init__(self, input_dim, output_dim, rng):
        """Create a linear layer with weights shaped (output_dim, input_dim)."""
        # Scale initial weights and biases by the input width.
        bound = 1 / np.sqrt(input_dim)
        self.weight = rng.uniform(-bound, bound, (output_dim, input_dim))
        self.bias = rng.uniform(-bound, bound, output_dim)

    def __call__(self, inputs):
        """Apply the affine readout along the last input dimension."""
        # Weights store outputs by row, so batched inputs multiply their transpose.
        return np.asarray(inputs) @ self.weight.T + self.bias


class _Reservoir:
    def _inputs(self, inputs):
        """Validate a frame batch and prepare its persistent reservoir state."""
        # A new stream starts at zero; changing batch size requires an explicit reset.
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(f"Expected inputs with shape (batch_size, {self.input_dim}).")
        if len(inputs) == 0 or not np.isfinite(inputs).all():
            raise ValueError("Inputs must be nonempty and finite.")
        if self.reservoir_state is None:
            self.initialize(len(inputs))
        if len(self.reservoir_state) != len(inputs):
            raise ValueError("Batch size changed; call initialize(new_batch_size).")
        return inputs

    def features(self, inputs):
        """Concatenate reservoir features from all substeps of one input frame."""
        # Repeated substeps use the same frame and preserve their temporal order.
        inputs = self._inputs(inputs)
        return np.concatenate(
            [self.reservoir_step(inputs) for _ in range(self.time_step)], axis=-1
        )

    def readout(self, features):
        """Map cached reservoir features to decision vectors."""
        # The same linear layer applies to frame batches and full video feature arrays.
        return self.w_out(features)

    def forward(self, inputs):
        """Predict dense/ragged videos from fresh states or advance a frame stream."""
        # Full videos reset state, while two-dimensional inputs continue a stream.
        if isinstance(inputs, (list, tuple)):
            return [self.forward(video[None])[0]
                    for video in as_videos(inputs, self.input_dim, "inputs")]
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim == 3:
            if (min(inputs.shape) < 1
                    or inputs.shape[-1] != self.input_dim
                    or not np.isfinite(inputs).all()):
                raise ValueError("Expected finite videos (N,T,input_dim), N/T >= 1.")
            self.initialize(len(inputs))
            try:
                return np.stack([
                    self.readout(self.features(inputs[:, t]))
                    for t in range(inputs.shape[1])
                ], axis=1)
            finally:
                self.reservoir_state = None
        return self.readout(self.features(inputs))

    __call__ = forward

    def get_reservoir_state(self):
        """Return the current state array, or None when no stream is active."""
        # Expose the live state without copying it.
        return self.reservoir_state

    def save(self, path):
        """Save configuration and all fixed/learned parameters; exclude runtime state."""
        parameters = {
            "w_in.weight": self.w_in.weight,
            "w_in.bias": self.w_in.bias,
            "w_out.weight": self.w_out.weight,
            "w_out.bias": self.w_out.bias,
        }
        if self.model_type == "esn":
            parameters["w_res"] = self.w_res
        else:
            parameters["J"] = self.J
        metadata = json.dumps({
            "format_version": 1,
            "model_type": self.model_type,
            "model_config": self.config,
        })
        # A file handle prevents np.savez from silently appending an extension.
        with Path(path).open("wb") as output:
            np.savez_compressed(output, metadata=np.array(metadata), **parameters)


class ESNReservoir(_Reservoir):
    model_type = "esn"

    def __init__(self, config):
        """Build a seeded ESN with fixed input/recurrent weights and a linear readout."""
        # Normalize recurrent weights to the configured spectral radius after masking.
        self.config = dict(config)
        self.input_dim = int(config["input_dim"])
        self.output_dim = int(config["output_dim"])
        self.reservoir_dim = int(config.get("reservoir_size", 50))
        self.time_step = int(config.get("time_step", 1))
        self.leak_rate = float(config.get("leak_rate", 0.5))
        if min(self.input_dim, self.output_dim, self.reservoir_dim, self.time_step) < 1:
            raise ValueError("Dimensions and time_step must be positive.")
        connectivity = float(config.get("reservoir_connectivity", 1.0))
        radius = float(config.get("spectral_radius", 1.0))
        if not 0 <= self.leak_rate <= 1 or not 0 <= connectivity <= 1 or radius < 0:
            raise ValueError("Invalid leak rate, connectivity, or spectral radius.")
        rng = np.random.default_rng(config.get("seed", 42))
        self.w_in = Linear(self.input_dim, self.reservoir_dim, rng)
        self.w_out = Linear(self.reservoir_dim * self.time_step, self.output_dim, rng)
        self.w_res = rng.standard_normal((self.reservoir_dim, self.reservoir_dim))
        if connectivity < 1:
            self.w_res *= rng.random(self.w_res.shape) < connectivity
        current_radius = np.abs(np.linalg.eigvals(self.w_res)).max()
        if current_radius > 0:
            self.w_res *= radius / current_radius
        elif radius > 0:
            raise ValueError("Recurrent matrix has zero spectral radius; increase connectivity.")
        self.reservoir_state = None

    def initialize(self, batch_size):
        """Reset each video to a zero state of shape (batch_size, reservoir_dim)."""
        # Each row is an independent reservoir state for one video.
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.reservoir_state = np.zeros((batch_size, self.reservoir_dim))

    def reservoir_step(self, inputs, t=0):
        """Advance the leaky tanh reservoir by one input frame substep."""
        # Blend the previous state with the candidate state using leak_rate.
        inputs = self._inputs(inputs)
        next_state = np.tanh(self.reservoir_state @ self.w_res.T + self.w_in(inputs))
        self.reservoir_state = (
            (1 - self.leak_rate) * self.reservoir_state + self.leak_rate * next_state
        )
        return self.reservoir_state


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}


def _tensor(matrices):
    """Return the ordered Kronecker product of the supplied matrices."""
    # The first matrix becomes the leftmost tensor factor.
    result = np.ones((1, 1), dtype=complex)
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


class QuantumReservoir(_Reservoir):
    """Match the original circuit order with dense U @ rho @ U.conj().T.

    Logical wire 0 is the leftmost tensor factor. Density matrices require
    O(4**num_qubits) memory per sample; use small reservoirs and bounded batches.
    """
    model_type = "quantum"

    def __init__(self, input_dim, output_dim, num_qubits, hz=1.5, Jz=3.0,
                 reservoir_time=1.0, interaction="XX", connectivity="all_to_all",
                 single_observables=("Z", "X"), pair_observables=("ZZ", "XX"),
                 damping=0.0, time_step=1, coupling_seed=42, seed=42):
        """Build fixed quantum dynamics, observables, and input/output projections."""
        self.config = dict(
            input_dim=input_dim, output_dim=output_dim, num_qubits=num_qubits,
            hz=hz, Jz=Jz, reservoir_time=reservoir_time, interaction=interaction,
            connectivity=connectivity, single_observables=list(single_observables),
            pair_observables=list(pair_observables), damping=damping,
            time_step=time_step, coupling_seed=coupling_seed, seed=seed,
        )
        self.input_dim, self.output_dim = int(input_dim), int(output_dim)
        self.num_qubits, self.time_step = int(num_qubits), int(time_step)
        if min(self.input_dim, self.output_dim, self.num_qubits, self.time_step) < 1:
            raise ValueError("Dimensions and time_step must be positive.")
        self.reservoir_dim = 2 ** self.num_qubits
        self.hz, self.reservoir_time = float(hz), float(reservoir_time)
        self.interaction, self.damping = interaction, float(damping)
        if not 0 <= self.damping <= 1:
            raise ValueError("damping must be in [0, 1].")
        if len(interaction) != 2 or any(op not in _PAULI for op in interaction):
            raise ValueError("interaction must be a two-letter Pauli string.")
        if connectivity == "all_to_all":
            self.pairs = [(i, j) for i in range(self.num_qubits)
                          for j in range(i + 1, self.num_qubits)]
        elif connectivity == "nearest_neighbour":
            self.pairs = [(i, i + 1) for start in (0, 1)
                          for i in range(start, self.num_qubits - 1, 2)]
        else:
            raise ValueError(f"Unknown connectivity: {connectivity!r}")
        rng = np.random.default_rng(coupling_seed)
        # Preserve the original coupling precision and sampling order.
        self.J = np.zeros((self.num_qubits, self.num_qubits), dtype=np.float32)
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                self.J[i, j] = self.J[j, i] = rng.uniform(-float(Jz) / 2, float(Jz) / 2)
        self.observable_labels = []
        for op in single_observables:
            if len(op) != 1 or op not in _PAULI:
                raise ValueError(f"Invalid single observable: {op!r}")
            for wire in range(self.num_qubits):
                label = ["I"] * self.num_qubits
                label[wire] = op
                self.observable_labels.append("".join(label))
        for op in pair_observables:
            if len(op) != 2 or any(p not in _PAULI for p in op):
                raise ValueError(f"Invalid pair observable: {op!r}")
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    label = ["I"] * self.num_qubits
                    label[i], label[j] = op
                    self.observable_labels.append("".join(label))
        if not self.observable_labels:
            raise ValueError("At least one observable is required.")
        self.observables = np.stack([
            _tensor([_PAULI[p] for p in label]) for label in self.observable_labels
        ])
        self.num_quantum_features = self.time_step * len(self.observables)
        rng = np.random.default_rng(seed)
        self.w_in = Linear(self.input_dim, self.num_qubits, rng)
        self.w_out = Linear(self.num_quantum_features, self.output_dim, rng)
        self._build_operators()
        self.reservoir_state = None

    def _local(self, matrix, wire):
        """Embed a single-qubit operator on one logical wire."""
        # All other wires receive identity operators in left-to-right wire order.
        return _tensor([matrix if i == wire else _PAULI["I"]
                        for i in range(self.num_qubits)])

    def _build_operators(self):
        """Cache Ising evolution and local amplitude-damping operators."""
        # Left-multiply pair rotations to preserve the original circuit execution order.
        angle = self.hz * self.reservoir_time
        rz = np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
        evolution = _tensor([rz] * self.num_qubits)
        identity = np.eye(self.reservoir_dim)
        for i, j in self.pairs:
            label = ["I"] * self.num_qubits
            label[i], label[j] = self.interaction
            pauli = _tensor([_PAULI[p] for p in label])
            theta = float(self.J[i, j]) * self.reservoir_time
            gate = np.cos(theta / 2) * identity - 1j * np.sin(theta / 2) * pauli
            evolution = gate @ evolution
        self._evolution = evolution
        self._damping_operators = []
        if self.damping:
            k0 = np.diag([1.0, np.sqrt(1 - self.damping)])
            k1 = np.array([[0, np.sqrt(self.damping)], [0, 0]])
            self._damping_operators = [
                (self._local(k0, wire), self._local(k1, wire))
                for wire in range(self.num_qubits)
            ]

    def initialize(self, batch_size):
        """Reset each video to the all-zero pure-state density matrix."""
        # Only the top-left population is occupied in the initial density matrix.
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.reservoir_state = np.zeros(
            (batch_size, self.reservoir_dim, self.reservoir_dim), dtype=complex
        )
        self.reservoir_state[:, 0, 0] = 1

    def get_reservoir_features(self):
        """Return real observable expectations for every current density matrix."""
        # Each feature is the trace of the density matrix times an observable.
        if self.reservoir_state is None:
            raise RuntimeError("Initialize the reservoir first.")
        return np.einsum("bij,kji->bk", self.reservoir_state, self.observables).real

    def reservoir_step(self, inputs):
        """Encode a frame, evolve the density matrices, and extract quantum features."""
        inputs = self._inputs(inputs)
        rho = self.reservoir_state
        angles = self.w_in(inputs)
        for wire in range(self.num_qubits):
            c, s = np.cos(angles[:, wire] / 2), np.sin(angles[:, wire] / 2)
            # RY(theta) = cos(theta/2) I - i sin(theta/2) Y.
            unitary = (c[:, None, None] * np.eye(self.reservoir_dim)
                       - 1j * s[:, None, None] * self._local(_PAULI["Y"], wire))
            rho = unitary @ rho @ unitary.conj().swapaxes(-1, -2)
        rho = self._evolution @ rho @ self._evolution.conj().T
        for k0, k1 in self._damping_operators:
            rho = k0 @ rho @ k0.conj().T + k1 @ rho @ k1.conj().T
        self.reservoir_state = rho
        return self.get_reservoir_features()


def load_model(path):
    """Load a NumPy checkpoint without pickle or optional dependencies."""
    # Restore the actual fixed weights and couplings rather than relying on seeds.
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported NumPy checkpoint version.")
        kind, config = metadata["model_type"], metadata["model_config"]
        if kind == "esn":
            model = ESNReservoir(config)
        elif kind == "quantum":
            model = QuantumReservoir(**config)
        else:
            raise ValueError(f"Unknown model type: {kind!r}")
        for layer_name in ("w_in", "w_out"):
            layer = getattr(model, layer_name)
            for parameter_name in ("weight", "bias"):
                value = archive[f"{layer_name}.{parameter_name}"]
                if value.shape != getattr(layer, parameter_name).shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {layer_name}.{parameter_name}.")
                setattr(layer, parameter_name, value.astype(np.float64))
        name = "w_res" if kind == "esn" else "J"
        value = archive[name]
        if value.shape != getattr(model, name).shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name}.")
        setattr(model, name, value.copy())
        if kind == "quantum":
            model._build_operators()
    return model
