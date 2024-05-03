import torch.nn as nn


class AttentionNetGated(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, dropout=True, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(AttentionNetGated, self).__init__()
        self.attention_a = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()]

        self.attention_b = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
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
