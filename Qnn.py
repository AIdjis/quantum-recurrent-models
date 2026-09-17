import torch
import torch.nn as nn
import pennylane as qml


class QLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits=5,
        n_qlayers=1,
        batch_first=True,
        return_sequences=False,
        return_state=False,
        entanglement=True,
        backend="default.qubit"
    ):

        super(QLSTM, self).__init__()

        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = input_size + hidden_size

        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend
        self.entanglement = entanglement

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.wires = list(range(n_qubits))

        self.dev = qml.device(self.backend,wires=self.wires)

        def _circuit(inputs, weights):

            # Angle encoding
            qml.AngleEmbedding(inputs, wires=self.wires, rotation="Y")

            # Variational layers
            for l in range(self.n_qlayers):

                # Trainable single-qubit rotations
                for q in range(self.n_qubits):

                    qml.Rot(
                        weights[l, q, 0],
                        weights[l, q, 1],
                        weights[l, q, 2],
                        wires=q
                    )

                # CNOT ring / strongly-entangling pattern
                if self.entanglement:

                    r = (l % (self.n_qubits - 1)) + 1

                    for q in range(self.n_qubits):

                        qml.CNOT(wires=[q, (q + r) % self.n_qubits])

            # Measurement
            return [ qml.expval(qml.PauliZ(w)) for w in self.wires]

        weight_shapes = {"weights": (n_qlayers, n_qubits, 3)}

        self.qnode = qml.QNode(
            _circuit,
            self.dev,
            interface="torch",
            diff_method="backprop",
            cache=True
        )

        self.VQC = nn.ModuleDict({

            "forget_gate":
                qml.qnn.TorchLayer(
                    self.qnode,
                    weight_shapes
                ),

            "input_gate":
                qml.qnn.TorchLayer(
                    self.qnode,
                    weight_shapes
                ),

            "candidate_gate":
                qml.qnn.TorchLayer(
                    self.qnode,
                    weight_shapes
                ),

            "output_gate": qml.qnn.TorchLayer(self.qnode, weight_shapes)
        })

        self.clayer_in = nn.Linear(self.concat_size, n_qubits)

        self.clayer_out = nn.Linear(n_qubits, hidden_size)

    def forward(self, x, init_states=None):

        if self.batch_first:

            batch_size, seq_length, _ = x.size()

        else:

            seq_length, batch_size, _ = x.size()

        hidden_seq = []

        # Initialize hidden and cell states
        if init_states is None:

            h_t = torch.zeros(
                batch_size,
                self.hidden_size,
                device=x.device
            )

            c_t = torch.zeros(
                batch_size,
                self.hidden_size,
                device=x.device
            )

        else:

            h_t, c_t = init_states

            h_t = h_t[0]
            c_t = c_t[0]

        # Process sequence
        for t in range(seq_length):

            x_t = x[:, t, :]

            # Concatenate hidden state and input
            v_t = torch.cat((h_t, x_t) ,dim=1)

            # Classical projection
            y_t = self.clayer_in(v_t)

            y_t = torch.clamp(
                y_t,
                -3.0,
                3.0
            )

            # Forget gate
            f_t = torch.sigmoid(
                self.clayer_out(
                    self.VQC["forget_gate"](y_t)
                )
            )

            # Input gate
            i_t = torch.sigmoid(
                self.clayer_out(
                    self.VQC["input_gate"](y_t)
                )
            )

            # Candidate/update gate
            g_t = torch.tanh(
                self.clayer_out(
                    self.VQC["candidate_gate"](y_t)
                )
            )

            # Output gate
            o_t = torch.sigmoid(
                self.clayer_out(
                    self.VQC["output_gate"](y_t)
                )
            )


            c_t = (f_t * c_t +i_t * g_t)

            h_t = (o_t * torch.tanh(c_t))

            hidden_seq.append(h_t.unsqueeze(0))

        # Output sequence
        hidden_seq = torch.cat(
            hidden_seq,
            dim=0
        )

        hidden_seq = hidden_seq.transpose(0,1).contiguous()

        return hidden_seq, (h_t, c_t)


class QGRU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits=5,
        n_qlayers=1,
        batch_first=True,
        entanglement=True,
        backend="default.qubit"
    ):

        super(QGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concat_size = input_size + hidden_size

        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.batch_first = batch_first
        self.entanglement = entanglement

        self.wires = list(range(n_qubits))

        self.dev = qml.device(
            backend,
            wires=self.wires
        )

        # Quantum circuit
        def _circuit(inputs, weights):

            # Angle encoding
            qml.AngleEmbedding(
                inputs,
                wires=self.wires,
                rotation="Y"
            )

            # Variational layers
            for l in range(self.n_qlayers):

                # Trainable rotations
                for q in range(self.n_qubits):

                    qml.Rot(
                        weights[l, q, 0],
                        weights[l, q, 1],
                        weights[l, q, 2],
                        wires=q
                    )

                # CNOT ring
                if self.entanglement:

                    r = (l % (self.n_qubits - 1)) + 1

                    for q in range(self.n_qubits):

                        qml.CNOT(
                            wires=[
                                q,
                                (q + r) % self.n_qubits
                            ]
                        )

            # Measurement
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        weight_shapes = {"weights": (n_qlayers,n_qubits, 3)}

        self.qnode = qml.QNode(
            _circuit,
            self.dev,
            interface="torch",
            cache=True
        )


        self.VQC = nn.ModuleDict({

            "update_gate":
                qml.qnn.TorchLayer(
                    self.qnode,
                    weight_shapes
                ),

            "reset_gate":
                qml.qnn.TorchLayer(
                    self.qnode,
                    weight_shapes
                ),

            "candidate_gate":
                qml.qnn.TorchLayer(
                    self.qnode,
                    weight_shapes
                )
        })

        self.clayer_in = nn.Linear(self.concat_size, n_qubits)

        self.clayer_out = nn.Linear(n_qubits, hidden_size)

    def forward(self, x, init_state=None):

        if self.batch_first:

            batch_size, seq_length, _ = x.size()

        else:

            seq_length, batch_size, _ = x.size()

        # Initialize hidden state
        if init_state is None:

            h_t = torch.zeros(
                batch_size,
                self.hidden_size,
                device=x.device
            )

        else:

            h_t = init_state

        hidden_seq = []

        # Process sequence
        for t in range(seq_length):

            x_t = x[:, t, :]

            # Concatenate hidden state and input
            v_t = torch.cat(
                (h_t, x_t),
                dim=1
            )

            # Classical projection
            y_t = self.clayer_in(v_t)

            y_t = torch.clamp(y_t, -3.0 , 3.0)

            u_t = torch.sigmoid(
                self.clayer_out(
                    self.VQC["update_gate"](y_t)
                )
            )

            r_t = torch.sigmoid(
                self.clayer_out(
                    self.VQC["reset_gate"](y_t)
                )
            )

            v_reset = torch.cat(
                (
                    r_t * h_t,
                    x_t
                ),
                dim=1
            )

            y_t = self.clayer_in(v_reset)

            y_t = torch.clamp(y_t, -3.0, 3.0)

            candidate = torch.tanh(
                self.clayer_out(
                    self.VQC["candidate_gate"](y_t)
                )
            )

            h_t = ((1 - u_t) * h_t + u_t * candidate)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq,dim=0)

        hidden_seq = hidden_seq.transpose(0,1).contiguous()

        return hidden_seq, h_t