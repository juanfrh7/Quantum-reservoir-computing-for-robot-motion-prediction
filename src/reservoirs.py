import torch
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Kraus, Pauli, SuperOp

from video_data import as_videos


def _predict_video(model, inputs):
    """Predict a decision vector after observing frames 0..t of each video."""
    # Reset at video boundaries and retain reservoir memory across frames.
    if isinstance(inputs, (list, tuple)):
        return [_predict_video(model, torch.as_tensor(
            video, dtype=model.w_in.weight.dtype, device=model.w_in.weight.device
        )[None])[0] for video in as_videos(inputs, model.input_dim, "inputs")]
    if (min(inputs.shape) < 1
            or inputs.shape[-1] != model.input_dim
            or not torch.isfinite(inputs).all()):
        raise ValueError("Expected finite videos (N,T,input_dim), N/T >= 1.")
    model.initialize(len(inputs))
    try:
        steps = []
        for t in range(inputs.shape[1]):
            features = torch.cat([
                model.reservoir_step(inputs[:, t]) for _ in range(model.time_step)
            ], dim=-1)
            steps.append(model.w_out(features))
        return torch.stack(steps, dim=1)
    finally:
        model.reservoir_state = None


class ESNReservoir(torch.nn.Module):
    def __init__(self, config):
        """Build a fixed ESN reservoir with a trainable linear decision readout."""
        # Only the readout is optimized; input and recurrent weights remain fixed.
        super().__init__()

        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.reservoir_dim = config.get("reservoir_size", 50)
        self.leak_rate = config.get("leak_rate", 0.5)
        self.time_step = config.get("time_step", 1)

        self.w_in = torch.nn.Linear(self.input_dim, self.reservoir_dim,
                                    bias=True)
        self.w_in.requires_grad_(False)

        self.w_out = torch.nn.Linear(self.reservoir_dim * config.get("time_step", 1),
                                     self.output_dim,
                                     bias=True)

        self.register_buffer(
            "w_res",
            self.make_recurrent_weights(config),)

        self.reservoir_state = None

    def initialize_reservoir_state(self, batch_size, type="zero", device=None):
        """Create zero ESN states when type is zero, optionally on a specified device."""
        # Each batch row holds one independent video state.
        if type == "zero":
            self.reservoir_state = torch.zeros(
                batch_size,
                self.reservoir_dim,
                device=device,)

    def initialize(self, batch_size):
        """Reset the ESN on the same device as its input projection."""
        # Match the state device to the fixed input weights before processing frames.
        self.initialize_reservoir_state(
            batch_size,
            device=self.w_in.weight.device,
        )

    def reservoir_step(self, inputs, t=0):
        """Update the leaky tanh state for one frame substep."""
        # The optional t argument is retained for compatibility; state advances per call.
        input_drive = self.w_in(inputs)
        recurrent_drive = self.reservoir_state @ self.w_res.T

        next_state = torch.tanh(recurrent_drive + input_drive)

        self.reservoir_state = (
            (1 - self.leak_rate) * self.reservoir_state
            + self.leak_rate * next_state)

        return self.reservoir_state

    def readout(self, reservoir_states):
        """Convert reservoir features into decision vectors."""
        # The affine layer operates on the last feature dimension.
        return self.w_out(reservoir_states)

    def forward(self, inputs):
        """Predict full videos or advance a persistent single-frame stream."""
        # Three-dimensional inputs start independent videos; frame batches retain state.
        if isinstance(inputs, (list, tuple)) or inputs.ndim == 3:
            return _predict_video(self, inputs)
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim or len(inputs) == 0:
            raise ValueError("Expected inputs (N,D) or videos (N,T,D).")
        if self.reservoir_state is None:
            self.initialize(len(inputs))
        if len(self.reservoir_state) != len(inputs):
            raise ValueError("Batch size changed; call initialize(new_batch_size).")
        return self.readout(torch.cat([
            self.reservoir_step(inputs) for _ in range(self.time_step)
        ], dim=-1))

    def make_recurrent_weights(self, config):
        """Sample sparse recurrent weights and set their spectral radius."""
        # Apply the connectivity mask before measuring and rescaling the eigenvalues.
        dim = self.reservoir_dim
        spectral_radius = config.get("spectral_radius", 1.0)
        connectivity = config.get("reservoir_connectivity", 1.0)

        weight = torch.randn(dim, dim)

        if connectivity < 1.0:
            mask = torch.rand(dim, dim) < connectivity
            weight = weight * mask

        eigenvalues = torch.linalg.eigvals(weight)
        current_radius = eigenvalues.abs().max()

        weight = weight / current_radius
        weight = weight * spectral_radius

        return weight

class QiskitQuantumReservoir(torch.nn.Module):
    """
    Qiskit density-matrix reservoir with a trainable linear readout.

    Videos: (N,T,D) -> (N,T,output_dim) decision predictions, with a fresh state.
    Streaming: (batch_size,input_dim) -> (batch_size,output_dim).

    Only w_out is trainable. The input projection and circuit
    parameters are fixed.

    The quantum state persists between streaming (N,D) calls.
    Full-video (N,T,D) calls reset state for independent videos automatically.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_qubits,
        hz=1.5,
        Jz=3.0,
        reservoir_time=1.0,
        interaction="XX",
        connectivity="all_to_all",
        single_observables=("Z", "X"),
        pair_observables=("ZZ", "XX"),
        damping=0.0,
        time_step=1,
        coupling_seed=42,
    ):
        """Build fixed quantum dynamics and observables with a trainable readout."""
        super().__init__()

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_qubits = int(num_qubits)
        self.time_step = int(time_step)

        if min(
            self.input_dim,
            self.output_dim,
            self.num_qubits,
            self.time_step,
        ) < 1:
            raise ValueError("Dimensions and time_step must be positive.")

        self.reservoir_dim = 2 ** self.num_qubits
        self.hz = float(hz)
        self.reservoir_time = float(reservoir_time)
        self.interaction = interaction
        self.damping = float(damping)

        if not 0.0 <= self.damping <= 1.0:
            raise ValueError("damping must be in [0, 1].")

        if (
            len(self.interaction) != 2
            or any(op not in "IXYZ" for op in self.interaction)
        ):
            raise ValueError(
                "interaction must be a two-letter Pauli string."
            )

        if connectivity == "all_to_all":
            self.pairs = [
                (i, j)
                for i in range(self.num_qubits)
                for j in range(i + 1, self.num_qubits)
            ]
        elif connectivity in ("nearest_neighbour"):
            # Preserve the interaction order of the Torch reservoir.
            self.pairs = [
                (i, i + 1)
                for start in (0, 1)
                for i in range(start, self.num_qubits - 1, 2)
            ]
        else:
            raise ValueError(f"Unknown connectivity: {connectivity!r}")

        # Fixed random Ising couplings.
        rng = np.random.default_rng(coupling_seed)
        Jz = float(Jz)

        couplings = np.zeros(
            (self.num_qubits, self.num_qubits),
            dtype=np.float32,
        )

        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                couplings[i, j] = couplings[j, i] = rng.uniform(
                    -Jz / 2,
                    Jz / 2,
                )

        self.register_buffer("J", torch.from_numpy(couplings))

        # Logical wire 0 is the leftmost tensor factor.
        # Single observables are evaluated on every qubit.
        # Pair observables are evaluated on every pair i < j.
        self.observable_labels = []

        for op in single_observables:
            if len(op) != 1 or op not in "IXYZ":
                raise ValueError(f"Invalid single observable: {op!r}")

            for wire in range(self.num_qubits):
                label = ["I"] * self.num_qubits
                label[wire] = op
                self.observable_labels.append("".join(label))

        for op in pair_observables:
            if len(op) != 2 or any(p not in "IXYZ" for p in op):
                raise ValueError(f"Invalid pair observable: {op!r}")

            for wire_i in range(self.num_qubits):
                for wire_j in range(wire_i + 1, self.num_qubits):
                    label = ["I"] * self.num_qubits
                    label[wire_i], label[wire_j] = op
                    self.observable_labels.append("".join(label))

        if not self.observable_labels:
            raise ValueError("At least one observable is required.")

        self.register_buffer(
            "observables",
            torch.tensor(
                np.stack([
                    Pauli(label).to_matrix()
                    for label in self.observable_labels
                ]),
                dtype=torch.complex64,
            ),
        )

        self.num_quantum_features = (
            self.time_step * len(self.observable_labels)
        )

        # Fixed input projection.
        self.w_in = torch.nn.Linear(self.input_dim, self.num_qubits)
        self.w_in.requires_grad_(False)

        # Trainable readout.
        self.w_out = torch.nn.Linear(
            self.num_quantum_features,
            self.output_dim,
        )

        # Runtime state moves with .to(), but is not saved in state_dict.
        self.register_buffer(
            "reservoir_state",
            None,
            persistent=False,
        )

    def initialize(self, batch_size):
        """Reset every sample to |0...0><0...0|."""
        # The initial state has unit population only in the all-zero basis state.
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        rho = torch.zeros(
            batch_size,
            self.reservoir_dim,
            self.reservoir_dim,
            dtype=self.observables.dtype,
            device=self.w_in.weight.device,
        )
        rho[:, 0, 0] = 1.0

        self.reservoir_state = rho

    def encode_data(self, circuit, angles):
        """Apply one RY encoding gate per qubit."""
        if len(angles) != self.num_qubits:
            raise ValueError("Provide one angle per qubit.")

        for wire, angle in enumerate(angles):
            # Reverse indices to match the Torch tensor ordering.
            qubit = self.num_qubits - 1 - wire
            circuit.ry(float(angle), qubit)

        return circuit

    def apply_ising_evolution(self, circuit):
        """Apply Z fields followed by ordered Ising pair rotations."""
        for qubit in range(self.num_qubits):
            circuit.rz(
                self.hz * self.reservoir_time,
                qubit,
            )

        pauli_pair = Pauli(self.interaction).to_matrix()
        couplings = self.J.detach().cpu().numpy()

        for wire_i, wire_j in self.pairs:
            theta = (
                float(couplings[wire_i, wire_j])
                * self.reservoir_time
            )

            # exp(-i * theta * P_i P_j / 2).
            gate = (
                np.cos(theta / 2) * np.eye(4)
                - 1j * np.sin(theta / 2) * pauli_pair
            )

            # First qarg is the local least-significant qubit.
            circuit.unitary(
                gate,
                [
                    self.num_qubits - 1 - wire_j,
                    self.num_qubits - 1 - wire_i,
                ],
                label=self.interaction,
            )

        return circuit

    def apply_amplitude_damping(self, circuit):
        """Append amplitude damping to every qubit."""
        # The two Kraus operators model population decay from the excited state.
        gamma = self.damping

        if gamma == 0.0:
            return circuit

        k0 = np.array(
            [
                [1.0, 0.0],
                [0.0, np.sqrt(1.0 - gamma)],
            ],
            dtype=complex,
        )
        k1 = np.array(
            [
                [0.0, np.sqrt(gamma)],
                [0.0, 0.0],
            ],
            dtype=complex,
        )

        instruction = Kraus([k0, k1]).to_instruction()

        for qubit in range(self.num_qubits):
            circuit.append(instruction, [qubit])

        return circuit

    def build_circuit(self, angles):
        """Build one encoding, evolution, and damping step."""
        # Damping acts after the encoded state undergoes fixed Ising evolution.
        circuit = QuantumCircuit(self.num_qubits)

        self.encode_data(circuit, angles)
        self.apply_ising_evolution(circuit)
        self.apply_amplitude_damping(circuit)

        return circuit

    @torch.no_grad()
    def _evolve(self, rho, angles):
        """Evolve a batch of density matrices using Qiskit."""
        # Simulate samples individually and restore their original tensor dtype and device.
        evolved_states = []

        for state, sample_angles in zip(rho, angles):
            state_numpy = state.detach().cpu().numpy()
            angles_numpy = sample_angles.detach().cpu().numpy()

            circuit = self.build_circuit(angles_numpy)
            evolved = DensityMatrix(state_numpy).evolve(circuit)

            evolved_states.append(
                torch.as_tensor(
                    evolved.data.copy(),
                    dtype=rho.dtype,
                    device=rho.device,
                )
            )

        return torch.stack(evolved_states)

    @torch.no_grad()
    def get_reservoir_features(self):
        """Return observable expectations for the current state."""
        # Contract matrix indices to compute Tr(rho O) for every sample and observable.
        if self.reservoir_state is None:
            raise RuntimeError("Initialize the reservoir first.")

        features = torch.einsum(
            "bij,kji->bk",
            self.reservoir_state,
            self.observables,
        ).real

        return features.to(dtype=self.w_out.weight.dtype)

    @torch.no_grad()
    def reservoir_step(self, inputs):
        """Encode inputs, update the state, and extract features."""
        # The fixed input projection supplies one encoding angle per qubit.
        if self.reservoir_state is None:
            raise RuntimeError("Initialize the reservoir first.")

        if self.reservoir_state.shape[0] != inputs.shape[0]:
            raise ValueError(
                "Batch size changed; call initialize(new_batch_size)."
            )

        angles = self.w_in(inputs)
        self.reservoir_state = self._evolve(
            self.reservoir_state,
            angles,
        )

        return self.get_reservoir_features()

    def forward(self, inputs):
        """Predict full videos or advance a persistent quantum frame stream."""
        if isinstance(inputs, (list, tuple)) or inputs.ndim == 3:
            return _predict_video(self, inputs)
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected inputs with shape (batch_size, {self.input_dim})."
            )

        if inputs.shape[0] == 0:
            raise ValueError("Input batch must not be empty.")

        if self.reservoir_state is None:
            self.initialize(inputs.shape[0])

        features = torch.cat(
            [
                self.reservoir_step(inputs)
                for _ in range(self.time_step)
            ],
            dim=-1,
        )

        # Keep the readout outside torch.no_grad().
        return self.w_out(features)

    def get_reservoir_state(self):
        """Return the current density-matrix batch, or None after a video reset."""
        # Expose the live runtime state rather than a saved checkpoint parameter.
        return self.reservoir_state
