import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import AttentionNetGated, PreGatingContextualAttention
from models.fusion import BilinearFusion, ConcatFusion, GatedConcatFusion


class GeneExprNarrowContextualAttentionGateTransformer(nn.Module):
    def __init__(self, model_size: str = 'medium', n_classes: int = 3, dropout: float = 0.25):
        super(GeneExprNarrowContextualAttentionGateTransformer, self).__init__()
        if model_size == 'small':
            self.model_sizes = [128, 128]
        elif model_size == 'medium':
            self.model_sizes = [256, 256]
        elif model_size == 'big':
            self.model_sizes = [512, 512]

        # H
        fc = nn.Sequential(
            nn.Linear(1024, self.model_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.H = fc

        self.self_attention = nn.MultiheadAttention(embed_dim=self.model_sizes[1], num_heads=1)

        # Path Transformer (T_H)
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_sizes[1], nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)

        # WSI Global Attention Pooling (rho_H)
        self.path_attention_head = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.path_rho = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])

        # Classifier
        self.classifier = nn.Linear(self.model_sizes[1], n_classes)

    def forward(self, wsi):
        # WSI Fully connected layer
        # H_bag: (Mxd_k)
        H_bag = self.H(wsi).squeeze(0)

        # Self-Attention results
        # H_coattn: Genomic-Guided WSI-level Embeddings (Nxd_k)
        # A_coattn: Co-Attention Matrix (NxM)
        H_coattn, A_coattn = self.self_attention(query=H_bag, key=H_bag, value=H_bag)

        # Set-Based MIL Transformers
        # Attention is permutation-equivariant, so dimensions are the same (Nxd_k)
        path_trans = self.path_transformer(H_coattn)

        # Global Attention Pooling
        A_path, h_path = self.path_attention_head(path_trans.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        # h_path: final WSI embeddings (dk)
        h_path = self.path_rho(h_path).squeeze()

        # Survival Layer

        # logits: classifier output
        # size   --> (1, 3)
        # domain --> R
        logits = self.classifier(h_path)
        Y = F.softmax(logits)

        attention_scores = {'attn': A_coattn, 'path': A_path}

        return Y, attention_scores

    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_ge_nacagat():
    print('Testing GeneExprNarrowGatingContextualAttentionGateTransformer...')

    wsi = torch.randn((3000, 1024))

    model_sizes = ['small', 'medium', 'big']

    for model_size in model_sizes:
        print(f'Size {model_size}')
        model = GeneExprNarrowContextualAttentionGateTransformer(model_size=model_size)
        Y, attention_scores = model(wsi)
        assert Y.shape[0] == 3
        assert attention_scores['attn'].shape[0] == 3000
        assert attention_scores['attn'].shape[1] == 3000
        assert attention_scores['path'].shape[0] == 1
        assert attention_scores['path'].shape[1] == 3000
        print('Forward successful')
