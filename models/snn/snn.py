import math
import torch
import torch.nn as nn


class SelfNormalizingBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.25):
        super(SelfNormalizingBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

    def forward(self, x):
        return self.block(x)


class SelfNormalizingNetwork(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 4, dropout: float = 0.25, model_size: str = 'small'):
        super(SelfNormalizingNetwork, self).__init__()
        self.n_classes = n_classes
        self.sizes = {
            'small': [256, 256, 256, 256],
            'large': [1024, 1024, 1024, 256]
        }

        hidden = self.sizes[model_size]
        fc = [SelfNormalizingBlock(input_dim, hidden[0], dropout=dropout)]
        for i, _ in enumerate(hidden[1:]):
            fc.append(SelfNormalizingBlock(hidden[i], hidden[i + 1], dropout=dropout))
        self.fc = nn.Sequential(*fc)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        self.init_weights()

    def forward(self, x):
        features = self.fc(x)
        logits = self.classifier(features)
        Y = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        survs = torch.cumprod(1 - hazards, dim=1)
        return hazards, survs, Y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                standard_deviation = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.normal_(0, standard_deviation)
                m.bias.data.zero_()

    def relocate(self, device):
        device = torch.device(device=device)
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.fc = nn.DataParallel(self.fc, device_ids=device_ids).to('cuda:0')
        else:
            self.fc = self.fc.to(device)
        self.classifier = self.classifier.to(device)


def test_snn_small():
    print('Testing SelfNormalizingNetwork (small)...')

    batch_size = 64
    genomic_size = 60000
    x = torch.randn((batch_size, genomic_size))
    model = SelfNormalizingNetwork(input_dim=genomic_size)
    model.relocate('cpu')
    hazards, survs, Y = model(x)
    assert hazards.shape[0] == survs.shape[0] == Y.shape[0] == batch_size
    assert hazards.shape[1] == survs.shape[1] == 4
    assert Y.shape[1] == 1

    print('Forward successful')


def test_snn_large():
    print('Testing SelfNormalizingNetwork (large)...')

    batch_size = 64
    genomic_size = 60000
    x = torch.randn((batch_size, genomic_size))
    model = SelfNormalizingNetwork(input_dim=genomic_size, model_size='large')
    model.relocate('cpu')
    hazards, survs, Y = model(x)
    assert hazards.shape[0] == survs.shape[0] == Y.shape[0] == batch_size
    assert hazards.shape[1] == survs.shape[1] == 4
    assert Y.shape[1] == 1

    print('Forward successful')
