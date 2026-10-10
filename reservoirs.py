import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Kraus, Pauli

class ESNReservoir(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.reservoir_dim = config.get("reservoir_size", 50)
        self.leak_rate = config.get("leak_rate", 0.5)
        self.time_step = config.get("time_step", 1)

        self.w_in = nn.Linear(self.input_dim, self.reservoir_dim, bias=True)
        self.w_in.requires_grad_(False)
        self.w_out = nn.Linear(
            self.reservoir_dim * self.time_step,
            self.output_dim,
            bias=True,
        )

        self.register_buffer(
            "w_res",
            self.make_recurrent_weights(config),
        )
        self.reservoir_state = None

    def initialize(self, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.reservoir_state = self.w_in.weight.new_zeros(
            batch_size,
            self.reservoir_dim,
        )

    def reservoir_step(self, inputs):
        if self.reservoir_state is None:
            raise RuntimeError("Initialize the reservoir first.")
        if self.reservoir_state.shape[0] != inputs.shape[0]:
            raise ValueError(
                "Batch size changed; call initialize(new_batch_size)."
            )
        input_drive = self.w_in(inputs)
        recurrent_drive = self.reservoir_state @ self.w_res.T

        next_state = torch.tanh(recurrent_drive + input_drive)

        self.reservoir_state = (
            (1 - self.leak_rate) * self.reservoir_state
            + self.leak_rate * next_state
        )
        return self.reservoir_state

    def readout(self, reservoir_states):
        return self.w_out(reservoir_states)

    def make_recurrent_weights(self, config):
        dim = self.reservoir_dim
        spectral_radius = config.get("spectral_radius", 1.0)
        connectivity = config.get("reservoir_connectivity", 1.0)

        weight = torch.randn(dim, dim)

        if connectivity < 1.0:
            mask = torch.rand(dim, dim) < connectivity
            weight = weight * mask

        eigenvalues = torch.linalg.eigvals(weight)
        current_radius = eigenvalues.abs().max()
        weight = weight / current_radius.clamp_min(torch.finfo(weight.dtype).eps)
        weight = weight * spectral_radius

        return weight

class QiskitQuantumReservoir(torch.nn.Module):
    """
    Qiskit density-matrix reservoir with a trainable linear readout.

    Input:  (batch_size, input_dim)
    Output: (batch_size, output_dim)

    Only w_out is trainable. The input projection and circuit
    parameters are fixed.

    The quantum state persists between forward calls.
    Call initialize(batch_size) before an independent batch/sequence.
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
        self.w_in = nn.Linear(self.input_dim, self.num_qubits)
        self.w_in.requires_grad_(False)

        # Trainable readout.
        self.w_out = nn.Linear(
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
        circuit = QuantumCircuit(self.num_qubits)

        self.encode_data(circuit, angles)
        self.apply_ising_evolution(circuit)
        self.apply_amplitude_damping(circuit)

        return circuit

    @torch.no_grad()
    def _evolve(self, rho, angles):
        """Evolve a batch of density matrices using Qiskit."""
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
        return self.reservoir_state