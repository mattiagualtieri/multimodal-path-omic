import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import AttentionNetGated
from models.reducers import (PoolingSequenceReducer,
                             ConvolutionalSequenceReducer,
                             AttentionGatedSequenceReducer,
                             PerceiverSequenceReducer)


class ThreeStreamMultimodalCoAttentionTransformer(nn.Module):
    def __init__(self, omic_sizes: [], n_classes: int = 4, dropout: float = 0.25, seq_reducer='perceiver',
                 reduced_size=10):
        super(ThreeStreamMultimodalCoAttentionTransformer, self).__init__()
        self.n_classes = n_classes
        self.d_k = 256
        self.reduced_size = reduced_size

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

        # Patches reducer
        self.seq_reducer = seq_reducer
        if self.seq_reducer == 'pooling':
            self.reducer = PoolingSequenceReducer(output_size=reduced_size)
        elif self.seq_reducer == 'conv':
            self.reducer = ConvolutionalSequenceReducer(output_size=reduced_size)
        elif self.seq_reducer == 'attention':
            self.reducer = AttentionGatedSequenceReducer(output_size=reduced_size)
        elif self.seq_reducer == 'perceiver':
            self.reducer = PerceiverSequenceReducer()
        else:
            raise NotImplementedError(f'Sequence reducer {self.seq_reducer} not implemented')

        # Genomic-Guided Co-Attention
        self.co_attention = nn.MultiheadAttention(embed_dim=self.d_k, num_heads=1)

        # Path Transformer (T_GG)
        genomic_g_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout,
                                                             activation='relu')
        self.genomic_g_transformer = nn.TransformerEncoder(genomic_g_encoder_layer, num_layers=2)

        # WSI Global Attention Pooling (rho_GG)
        self.genomic_g_attention_head = AttentionNetGated(n_classes=1)
        self.genomic_g_rho = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout)])

        # Omic Transformer (T_G)
        genomic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout,
                                                           activation='relu')
        self.genomic_transformer = nn.TransformerEncoder(genomic_encoder_layer, num_layers=2)

        # Genomic Global Attention Pooling (rho_G)
        self.genomic_attention_head = AttentionNetGated(n_classes=1)
        self.genomic_rho = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout)])

        # Histo Transformer (T_H)
        histo_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout,
                                                         activation='relu')
        self.histo_transformer = nn.TransformerEncoder(histo_encoder_layer, num_layers=2)

        # Histo Global Attention Pooling (rho_H)
        self.histo_attention_head = AttentionNetGated(n_classes=1)
        self.histo_rho = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout)])

        # Fusion Layer
        self.fusion_layer = nn.Sequential(*[
            nn.Linear(256 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
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

        # This is the same as H_bag, but with lower dimensionality
        # H_bag_reduced (Rxd_k)
        if isinstance(self.reducer, PerceiverSequenceReducer):
            queries = torch.zeros(self.reduced_size, self.d_k)
            H_bag_reduced = self.reducer(H_bag, queries=queries)
            H_bag_reduced = H_bag_reduced.squeeze(0)
        else:
            H_bag_reduced = self.reducer(H_bag)

        # Co-Attention results

        # genomic_g_coattn: Genomic-Guided WSI-level Embeddings (Nxd_k)
        # A_genomic_g_coattn: Co-Attention Matrix (NxM)
        genomic_g_coattn, A_genomic_g_coattn = self.co_attention(query=G_bag, key=H_bag, value=H_bag)

        # Set-Based MIL Transformers
        # Attention is permutation-equivariant, so dimensions are the same (Nxd_k) and (Rxd_k)
        genomic_g_trans = self.genomic_g_transformer(genomic_g_coattn)
        genomic_trans = self.genomic_transformer(G_bag)
        histo_trans = self.histo_transformer(H_bag_reduced)

        # Global Attention Pooling
        A_genomic_g, h_genomic_g = self.genomic_g_attention_head(genomic_g_trans.squeeze(1))
        A_genomic_g = torch.transpose(A_genomic_g, 1, 0)
        A_genomic_g = torch.mm(F.softmax(A_genomic_g, dim=1), h_genomic_g)
        # h_genomic_g: final Genomic-Guided embedding (dk)
        h_genomic_g = self.genomic_g_rho(A_genomic_g).squeeze()

        A_genomic, h_genomic = self.genomic_attention_head(genomic_trans.squeeze(1))
        A_genomic = torch.transpose(A_genomic, 1, 0)
        h_genomic = torch.mm(F.softmax(A_genomic, dim=1), h_genomic)
        # h_genomic: final Genomics embedding (dk)
        h_genomic = self.genomic_rho(h_genomic).squeeze()

        A_histo, h_histo = self.histo_attention_head(histo_trans.squeeze(1))
        A_histo = torch.transpose(A_histo, 1, 0)
        h_histo = torch.mm(F.softmax(A_histo, dim=1), h_histo)
        # h_histo: final Patches embedding (dk)
        h_histo = self.histo_rho(h_histo).squeeze()

        # Fusion Layer
        concat = torch.cat([h_genomic_g, h_genomic, h_histo], dim=0)
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

        # attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}
        attention_scores = {}

        return hazards, survs, Y, attention_scores


def test_mdcat():
    print('Testing ThreeStreamMultimodalCoAttentionTransformer...')

    wsi = torch.randn((3000, 1024))
    omics = [torch.randn(dim) for dim in [100, 200, 300, 400, 500, 600]]
    omic_sizes = [omic.size()[0] for omic in omics]
    model = ThreeStreamMultimodalCoAttentionTransformer(omic_sizes=omic_sizes, seq_reducer='perceiver')
    hazards, S, Y_hat, attention_scores = model(wsi, omics)
    assert hazards.shape[0] == S.shape[0] == Y_hat.shape[0] == 1
    assert hazards.shape[1] == S.shape[1] == Y_hat.shape[1] == 4
    # assert attention_scores['coattn'].shape[0] == len(omic_sizes)
    # assert attention_scores['coattn'].shape[1] == 3000
    # assert attention_scores['path'].shape[0] == attention_scores['omic'].shape[0] == 1
    # assert attention_scores['path'].shape[1] == attention_scores['omic'].shape[1] == len(omic_sizes)

    print('Forward successful')
