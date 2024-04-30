import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/mahmoodlab/MCAT/blob/master/Model%20Computation%20%2B%20Complexity%20Overview.ipynb


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


class MultimodalCoAttentionTransformer(nn.Module):
    def __init__(self, omic_sizes: [], n_classes: int = 4, dropout: float = 0.25):
        super(MultimodalCoAttentionTransformer, self).__init__()
        self.n_classes = n_classes
        self.d_k = 256

        # H
        fc = nn.Sequential(
            nn.Linear(1024, self.d_k),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.H = fc

        # G
        omic_encoders = []
        for omic_size in omic_sizes:
            fc = nn.Sequential(
                nn.Sequential(
                    nn.Linear(omic_size, 256),
                    nn.ELU(),
                    nn.AlphaDropout(p=dropout, inplace=False)),
                nn.Sequential(
                    nn.Linear(256, self.d_k),
                    nn.ELU(),
                    nn.AlphaDropout(p=dropout, inplace=False))
            )
            omic_encoders.append(fc)
        self.G = nn.ModuleList(omic_encoders)

        # Genomic-Guided Co-Attention
        self.co_attention = nn.MultiheadAttention(embed_dim=self.d_k, num_heads=1)

        # Path Transformer (T_H)
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)

        # WSI Global Attention Pooling (rho_H)
        self.path_attention_head = AttentionNetGated(n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout)])

        # Omic Transformer (T_G)
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)

        # Genomic Global Attention Pooling (rho_G)
        self.omic_attention_head = AttentionNetGated(n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout)])

        # Fusion Layer
        self.fusion_layer = nn.Sequential(*[
            nn.Linear(256 * 2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        ])

        # Classifier
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, wsi, omics):
        # WSI Fully connected layer
        # H_bag: (Mxd_k)
        H_bag = self.H(wsi).squeeze(0)

        # N Omics Fully connected layers
        G_omic = [self.G[index].forward(omic.type(torch.float32)) for index, omic in enumerate(omics)]
        # G_bag: (Nxd_k)
        G_bag = torch.stack(G_omic).squeeze(1)

        # Co-Attention results
        # H_coattn: Genomic-Guided WSI-level Embeddings (Nxd_k)
        # A_coattn: Co-Attention Matrix (NxM)
        H_coattn, A_coattn = self.co_attention(query=G_bag, key=H_bag, value=H_bag)

        # Set-Based MIL Transformers
        # Attention is permutation-equivariant, so dimensions are the same (Nxd_k)
        path_trans = self.path_transformer(H_coattn)
        omic_trans = self.omic_transformer(G_bag)

        # Global Attention Pooling
        A_path, h_path = self.path_attention_head(path_trans.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        # h_path: final WSI embeddings (dk)
        h_path = self.path_rho(h_path).squeeze()

        A_omic, h_omic = self.omic_attention_head(omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
        # h_omic: final omics embeddings (dk)
        h_omic = self.omic_rho(h_omic).squeeze()

        # Fusion Layer
        concat = torch.cat([h_path, h_omic], dim=0)
        # h: final representation (dk)
        h = self.fusion_layer(concat)

        # Survival Layer

        # logits: classifier output
        # size   --> (1, 4)
        # domain --> R
        logits = self.classifier(h).unsqueeze(0)
        # hazards: probability of patient death in interval j
        # size   --> (1, 4)
        # domain --> [0, 1]
        hazards = torch.sigmoid(logits)
        # survs: probability of patient survival after time t
        # size   --> (1, 4)
        # domain --> [0, 1]
        survs = torch.cumprod(1 - hazards, dim=1)
        # Y: predicted probability distribution
        # size   --> (1, 4)
        # domain --> [0, 1] (probability distribution)
        Y = F.softmax(logits, dim=1)

        attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}

        return hazards, survs, Y, attention_scores


def test():
    print('Testing MultimodalCoAttentionTransformer...')

    wsi = torch.randn((15231, 1024))
    omics = [torch.randn(dim) for dim in [100, 200, 300, 400, 500, 600]]
    omic_sizes = [omic.size()[0] for omic in omics]
    model = MultimodalCoAttentionTransformer(omic_sizes=omic_sizes)
    # hazards, S, Y_hat, attention_scores = model(wsi, omics)
    Y = model(wsi, omics)

    print('Forward successful')


if __name__ == '__main__':
    test()
