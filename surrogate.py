import torch
import torch.nn as nn


class ClassicalCosineLayer(nn.Module):

    def __init__(self, n_features):

        super(ClassicalCosineLayer, self).__init__()

        # Trainable scale
        self.scale = nn.Parameter(
            torch.ones(n_features)
        )

        # Trainable shift
        self.shift = nn.Parameter(
            torch.zeros(n_features)
        )

    def forward(self, x):

        return self.scale * torch.cos(x + self.shift)


class SurrogateLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits=5,
        batch_first=True,
        return_sequences=False,
        return_state=False
    ):

        super(SurrogateLSTM, self).__init__()

        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = input_size + hidden_size
        self.n_qubits = n_qubits

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        # Classical layers surrounding the surrogate
        self.clayer_in = nn.Linear(
            self.concat_size,
            n_qubits
        )

        self.clayer_out = nn.Linear(
            n_qubits,
            hidden_size
        )

        # Classical cosine surrogates
        # One independent transformation per LSTM gate
        self.VQC = nn.ModuleDict({

            "forget_gate":
                ClassicalCosineLayer(n_qubits),

            "input_gate":
                ClassicalCosineLayer(n_qubits),

            "candidate_gate":
                ClassicalCosineLayer(n_qubits),

            "output_gate":
                ClassicalCosineLayer(n_qubits)
        })


    def forward(self, x, init_states=None):

        if self.batch_first:

            batch_size, seq_length, _ = x.size()

        else:

            seq_length, batch_size, _ = x.size()

        hidden_seq = []

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
            v_t = torch.cat((h_t, x_t), dim=1)

            # Classical projection
            y_t = self.clayer_in(v_t)

            y_t = torch.clamp(y_t,-3.0,3.0)

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

            # LSTM state update
            c_t = (f_t * c_t+i_t * g_t)

            h_t = (o_t *torch.tanh(c_t) )

            hidden_seq.append(
                h_t.unsqueeze(0)
            )

        # Output sequence
        hidden_seq = torch.cat(hidden_seq,dim=0)

        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)


class SurrogateGRU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits=5,
        batch_first=True
    ):

        super(SurrogateGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concat_size = input_size + hidden_size

        self.n_qubits = n_qubits
        self.batch_first = batch_first


        self.clayer_in = nn.Linear(
            self.concat_size,
            n_qubits
        )

        self.clayer_out = nn.Linear(
            n_qubits,
            hidden_size
        )


        self.VQC = nn.ModuleDict({

            "update_gate":
                ClassicalCosineLayer(n_qubits),

            "reset_gate":
                ClassicalCosineLayer(n_qubits),

            "candidate_gate":
                ClassicalCosineLayer(n_qubits)
        })


    def forward(self, x, init_state=None):

        if self.batch_first:

            batch_size, seq_length, _ = x.size()

        else:

            seq_length, batch_size, _ = x.size()

        if init_state is None:

            h_t = torch.zeros(
                batch_size,
                self.hidden_size,
                device=x.device
            )

        else:

            h_t = init_state

        hidden_seq = []

        for t in range(seq_length):

            x_t = x[:, t, :]

            v_t = torch.cat((h_t, x_t), dim=1)

            y_t = self.clayer_in(v_t)

            y_t = torch.clamp(y_t, -3.0, 3.0)

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

            y_t = torch.clamp(y_t,-3.0,3.0)

            candidate = torch.tanh(
                self.clayer_out(
                    self.VQC["candidate_gate"](y_t)
                )
            )

            h_t = ((1 - u_t) * h_t+u_t * candidate)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq,dim=0)

        hidden_seq = hidden_seq.transpose(0,1).contiguous()

        return hidden_seq, h_t