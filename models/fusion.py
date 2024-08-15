import torch
import torch.nn as nn

from models.utils import init_max_weights


class ConcatFusion(nn.Module):
    def __init__(self, dim1=256, dim2=256):
        super(ConcatFusion, self).__init__()
        self.fusion_layer = nn.Sequential(*[
            nn.Linear(dim1 + dim2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        ])

    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=0)
        return self.fusion_layer(concat)


class GatedConcatFusion(nn.Module):
    def __init__(self, dim1=256, dim2=256):
        super(GatedConcatFusion, self).__init__()
        self.g1 = nn.Sequential(nn.Linear(dim1, 1), nn.Sigmoid())
        self.g2 = nn.Sequential(nn.Linear(dim2, 1), nn.Sigmoid())
        self.fusion_layer = nn.Sequential(*[
            nn.Linear(dim1 + dim2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        ])

    def forward(self, x1, x2):
        g1 = self.g1(x1)
        g2 = self.g2(x2)
        x1 = g1 * x1
        x2 = g2 * x2
        concat = torch.cat([x1, x2], dim=0)
        return self.fusion_layer(concat)


class BilinearFusion(nn.Module):
    def __init__(self, skip=True, use_bilinear=True, gate1=1, gate2=1, dim1=32, dim2=32, scale_dim1=1, scale_dim2=1, mmhid=64, dropout=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1 // scale_dim1, dim2 // scale_dim2
        skip_dim = dim1 + dim2 + 2 if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout))

        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.encoder1 = nn.Sequential(nn.Linear((dim1 + 1) * (dim2 + 1), mmhid), nn.ReLU(), nn.Dropout(p=dropout))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout))
        init_max_weights(self)

    def forward(self, vec1, vec2):
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        # Fusion
        o1 = o1.unsqueeze(0)
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=o1.device)), 1)
        o2 = o2.unsqueeze(0)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=o2.device)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, o1, o2), 1)
        out = self.encoder2(out)
        return out.squeeze(0)
