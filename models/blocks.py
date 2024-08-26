import torch
import torch.nn as nn


class AttentionNetGated(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, dropout: bool = True, n_classes: int = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            input_dim (int): input feature dimension
            hidden_dim (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(AttentionNetGated, self).__init__()
        self.attention_a = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        ]

        self.attention_b = [
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        ]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        # N x n_classes
        A = self.attention_c(A)
        return A, x


class PreGatedAttention(nn.Module):
    def __init__(self, dim1: int = 256, dim2: int = 256, dk: int = 256):
        super(PreGatedAttention, self).__init__()
        self.dk = dk
        self.scale = 1 / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
        self.fc_Q = nn.Linear(dim2, self.dk)
        self.fc_K = nn.Linear(dim1, self.dk)
        self.fc_V = nn.Linear(dim1, self.dk)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        Q = self.fc_Q(x2)
        K = self.fc_K(x1)
        V = self.fc_V(x1)

        QK = torch.matmul(Q, K.transpose(-2, -1))
        P = (torch.matmul(torch.tanh(Q), torch.tanh(K.transpose(-2, -1))) + 1) / 2
        scores = (QK * P) / self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        Q_hat = torch.matmul(attention_weights, V)

        return Q, Q_hat, attention_weights


class ContextualAttentionGate(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 128):
        super(ContextualAttentionGate, self).__init__()
        # FC Layer for Q
        self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU())
        # First FC Layer for Q_hat
        self.fc2 = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU())
        # Second FC Layer for Q_hat
        self.fc3 = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU())

        self.G = nn.Sequential(nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.E = nn.Sequential(nn.ReLU(), nn.LayerNorm(hidden_dim))

        self.fc_c = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, Q: torch.Tensor, Q_hat: torch.Tensor):
        G = self.G(self.fc1(Q) + self.fc2(Q_hat))
        E = self.E(self.fc3(Q_hat))
        C = G * E
        C = self.fc_c(C)

        return C


class PreGatingContextualAttentionGate(nn.Module):
    def __init__(self, dim1: int = 256, dim2: int = 256, dk: int = 256, output_dim: int = 128):
        r"""
        Pre-gating and Contextual Attention Gate (PCAG)

        args:
            dim1 (int): dimension of the first input
            dim2 (int): dimension of the second input
            dk (int): hidden layer dimension
            output_dim (int): dimension of the output
        """
        super(PreGatingContextualAttentionGate, self).__init__()

        self.dk = dk
        self.output_dim = output_dim

        self.pg_coattn = PreGatedAttention(dim1=dim1, dim2=dim2, dk=self.dk)

        self.CAG = ContextualAttentionGate(dim=self.dk, hidden_dim=self.output_dim)

        self.final_fc = nn.Sequential(nn.Linear(self.dk, self.output_dim), nn.ReLU())

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        Q, Q_hat, attention_weights = self.pg_coattn(x1, x2)

        C = self.CAG(Q, Q_hat)
        Q = self.final_fc(Q)

        return Q + C, attention_weights


def test_pga():
    print('Testing PreGatedAttention...')

    x1 = torch.randn((4, 4))
    x2 = torch.randn((2, 4))

    block = PreGatedAttention(dim1=4, dim2=4, dk=4)

    Q, Q_hat, attention_weights = block(x1, x2)

    assert Q.shape[0] == Q_hat.shape[0] == attention_weights.shape[0] == 2
    assert Q_hat.shape[1] == Q_hat.shape[1] == attention_weights.shape[1] == 4

    print('Forward successful')


def test_cag():
    print('Testing ContextualAttentionGate...')

    x1 = torch.randn((8, 256))
    x2 = torch.randn((8, 256))

    block = ContextualAttentionGate(hidden_dim=256)

    C = block(x1, x2)

    assert C.shape[0] == 8
    assert C.shape[1] == 256

    block = ContextualAttentionGate(hidden_dim=128)

    C = block(x1, x2)

    assert C.shape[0] == 8
    assert C.shape[1] == 256

    print('Forward successful')


def test_pcag():
    print('Testing PreGatingContextualAttentionGate...')

    slide = torch.randn((3000, 1024))
    omics = torch.randn((6, 256))

    block = PreGatingContextualAttentionGate(dim1=1024, dim2=256, dk=256, output_dim=128)
    output, attention_weights = block(slide, omics)
    assert output.shape[0] == attention_weights.shape[0] == 6
    assert output.shape[1] == 128
    assert attention_weights.shape[1] == 3000

    print('Forward successful')
