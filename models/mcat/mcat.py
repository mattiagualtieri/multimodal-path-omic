import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import AttentionNetGated
from models.fusion import BilinearFusion, ConcatFusion, GatedConcatFusion


# https://github.com/mahmoodlab/MCAT/blob/master/Model%20Computation%20%2B%20Complexity%20Overview.ipynb


class MultimodalCoAttentionTransformer(nn.Module):
    def __init__(self, omic_sizes: [], model_size: str = 'medium', n_classes: int = 4, dropout: float = 0.25, fusion: str = 'concat', device: str = 'cpu', inference: bool = False):
        super(MultimodalCoAttentionTransformer, self).__init__()
        self.inference = inference
        self.n_classes = n_classes
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

        # G
        omic_encoders = []
        for omic_size in omic_sizes:
            fc = nn.Sequential(
                nn.Sequential(
                    nn.Linear(omic_size, self.model_sizes[0]),
                    nn.ELU(),
                    nn.AlphaDropout(p=dropout, inplace=False)),
                nn.Sequential(
                    nn.Linear(self.model_sizes[0], self.model_sizes[1]),
                    nn.ELU(),
                    nn.AlphaDropout(p=dropout, inplace=False))
            )
            omic_encoders.append(fc)
        self.G = nn.ModuleList(omic_encoders)

        # Genomic-Guided Co-Attention
        self.co_attention = nn.MultiheadAttention(embed_dim=self.model_sizes[1], num_heads=1)

        # Path Transformer (T_H)
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_sizes[1], nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)

        # WSI Global Attention Pooling (rho_H)
        self.path_attention_head = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.path_rho = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])

        # Omic Transformer (T_G)
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_sizes[1], nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)

        # Genomic Global Attention Pooling (rho_G)
        self.omic_attention_head = AttentionNetGated(n_classes=1, input_dim=self.model_sizes[1], hidden_dim=self.model_sizes[1])
        self.omic_rho = nn.Sequential(*[nn.Linear(self.model_sizes[1], self.model_sizes[1]), nn.ReLU(), nn.Dropout(dropout)])

        # Fusion Layer
        self.fusion = fusion
        if self.fusion == 'concat':
            self.fusion_layer = ConcatFusion(dims=[self.model_sizes[1], self.model_sizes[1]],
                                             hidden_size=self.model_sizes[1], output_size=self.model_sizes[1]).to(device=device)
        elif self.fusion == 'bilinear':
            self.fusion_layer = BilinearFusion(dim1=self.model_sizes[1], dim2=self.model_sizes[1], output_size=self.model_sizes[1])
        elif self.fusion == 'gated_concat':
            self.fusion_layer = GatedConcatFusion(dims=[self.model_sizes[1], self.model_sizes[1]],
                                                  hidden_size=self.model_sizes[1], output_size=self.model_sizes[1]).to(device=device)
        else:
            raise RuntimeError(f'Fusion mechanism {self.fusion} not implemented')

        # Classifier
        self.classifier = nn.Linear(self.model_sizes[1], n_classes)

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
        H_coattn, A_coattn = self.co_attention(query=G_bag, key=H_bag, value=H_bag, need_weights=self.inference)

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
        # h: final representation (dk)
        h = self.fusion_layer(h_path, h_omic)

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

    def get_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_mcat():
    print('Testing MultimodalCoAttentionTransformer...')

    wsi = torch.randn((3000, 1024))
    omics = [torch.randn(dim) for dim in [100, 200, 300, 400, 500, 600]]
    omic_sizes = [omic.size()[0] for omic in omics]

    model_sizes = ['small', 'medium', 'big']

    for model_size in model_sizes:
        print(f'Size {model_size}')
        model = MultimodalCoAttentionTransformer(omic_sizes=omic_sizes, model_size=model_size, inference=True)
        hazards, S, Y_hat, attention_scores = model(wsi, omics)
        assert hazards.shape[0] == S.shape[0] == Y_hat.shape[0] == 1
        assert hazards.shape[1] == S.shape[1] == Y_hat.shape[1] == 4
        assert attention_scores['coattn'].shape[0] == len(omic_sizes)
        assert attention_scores['coattn'].shape[1] == 3000
        assert attention_scores['path'].shape[0] == attention_scores['omic'].shape[0] == 1
        assert attention_scores['path'].shape[1] == attention_scores['omic'].shape[1] == len(omic_sizes)
        print('Forward successful')

    print('Testing non-inference')
    model = MultimodalCoAttentionTransformer(omic_sizes=omic_sizes, model_size=model_sizes[0], inference=False)
    hazards, S, Y_hat, attention_scores = model(wsi, omics)
    assert hazards.shape[0] == S.shape[0] == Y_hat.shape[0] == 1
    assert hazards.shape[1] == S.shape[1] == Y_hat.shape[1] == 4
    assert attention_scores['coattn'] is None
    assert attention_scores['path'].shape[0] == attention_scores['omic'].shape[0] == 1
    assert attention_scores['path'].shape[1] == attention_scores['omic'].shape[1] == len(omic_sizes)
    print('Forward successful')
