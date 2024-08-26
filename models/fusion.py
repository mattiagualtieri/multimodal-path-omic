import torch
import torch.nn as nn

from models.utils import init_max_weights


class ConcatFusion(nn.Module):
    def __init__(self, dims: list, hidden_size: int = 256, output_size: int = 256):
        super(ConcatFusion, self).__init__()
        self.fusion_layer = nn.Sequential(*[
            nn.Linear(sum(dims), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        ])

    def forward(self, *x):
        concat = torch.cat(x, dim=0)
        return self.fusion_layer(concat)


class GatedConcatFusion(nn.Module):
    def __init__(self, dims: list, hidden_size: int = 256, output_size: int = 256):
        super(GatedConcatFusion, self).__init__()
        self.gates = []
        for dim in dims:
            self.gates.append(nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid()))
        self.fusion_layer = nn.Sequential(*[
            nn.Linear(sum(dims), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        ])

    def forward(self, *x):
        items = []
        for gate, item in zip(self.gates, x):
            g = gate(item)
            items.append(item * g)
        concat = torch.cat(items, dim=0)
        return self.fusion_layer(concat)


class BilinearFusion(nn.Module):
    def __init__(self, dim1: int = 256, dim2: int = 256, hidden_size: int = 32, output_size: int = 64,
                 mm_hidden_size: int = 64,
                 use_skip_connection=True,
                 use_bilinear=True,
                 use_gates=True,
                 dropout=0.25):

        super(BilinearFusion, self).__init__()
        self.use_skip_connection = use_skip_connection
        self.use_bilinear = use_bilinear
        self.use_gates = use_gates

        self.linear_h1 = nn.Sequential(nn.Linear(dim1, hidden_size), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1, dim2, hidden_size) if use_bilinear else nn.Linear(dim1 + dim2, hidden_size)
        self.linear_o1 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(p=dropout))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2, hidden_size), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim2, dim1, hidden_size) if use_bilinear else nn.Linear(dim2 + dim1, hidden_size)
        self.linear_o2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(p=dropout))

        self.post_fusion_dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Sequential(*[
            nn.Linear((hidden_size + 1) * (hidden_size + 1), mm_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        ])

        self.fc2 = nn.Sequential(*[
            nn.Linear(mm_hidden_size + (hidden_size * 2) + 2, output_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        ])

        init_max_weights(self)

    def forward(self, *x):
        if len(x) != 2:
            raise RuntimeError('Bilinear fusion is possible only on 2 inputs')
        x1 = x[0]
        x2 = x[1]
        if self.use_gates:
            h1 = self.linear_h1(x1)
            z1 = self.linear_z1(x1, x2) if self.use_bilinear else self.linear_z1(torch.cat((x1, x2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_o1(x1)

        if self.use_gates:
            h2 = self.linear_h2(x2)
            z2 = self.linear_z2(x2, x1) if self.use_bilinear else self.linear_z2(torch.cat((x2, x1), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_o2(x2)

        # Fusion - Kronecker Product
        o1 = o1.unsqueeze(0)
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=o1.device)), 1)
        o2 = o2.unsqueeze(0)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=o2.device)), 1)
        out = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)

        out = self.post_fusion_dropout(out)

        out = self.fc1(out)
        if self.use_skip_connection:
            out = torch.cat((out, o1, o2), 1)
        out = self.fc2(out)
        return out.squeeze(0)


def test_concat_fusion():
    print('Testing ConcatFusion...')

    x1 = torch.randn((256,))
    x2 = torch.randn((256,))
    fusion = ConcatFusion(dims=[len(x1), len(x2)])
    out = fusion(x1, x2)
    assert len(out) == 256

    x1 = torch.randn((256,))
    x2 = torch.randn((256,))
    x3 = torch.randn((256,))
    fusion = ConcatFusion(dims=[len(x1), len(x2), len(x3)], hidden_size=128, output_size=32)
    out = fusion(x1, x2, x3)
    assert len(out) == 32

    print('Test successful')


def test_gated_concat_fusion():
    print('Testing GatedConcatFusion...')

    x1 = torch.randn((256,))
    x2 = torch.randn((256,))
    fusion = GatedConcatFusion(dims=[len(x1), len(x2)])
    out = fusion(x1, x2)
    assert len(out) == 256

    x1 = torch.randn((256,))
    x2 = torch.randn((256,))
    x3 = torch.randn((256,))
    fusion = GatedConcatFusion(dims=[len(x1), len(x2), len(x3)], hidden_size=128, output_size=32)
    out = fusion(x1, x2, x3)
    assert len(out) == 32

    print('Test successful')


def test_bilinear_fusion():
    print('Testing BilinearFusion...')

    x1 = torch.randn((256,))
    x2 = torch.randn((256,))
    fusion = BilinearFusion(dim1=len(x1), dim2=len(x2))
    out = fusion(x1, x2)
    assert len(out) == 64

    x1 = torch.randn((256,))
    x2 = torch.randn((256,))
    fusion = BilinearFusion(dim1=len(x1), dim2=len(x2), output_size=256)
    out = fusion(x1, x2)
    assert len(out) == 256

    print('Test successful')
