import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import AttentionNetGated
from models.perceiver import Perceiver


class PoolingSequenceReducer(nn.Module):
    def __init__(self, output_size=10):
        super(PoolingSequenceReducer, self).__init__()
        self.pooling = nn.AdaptiveMaxPool1d(output_size=output_size)

    def forward(self, x):
        x = x.permute(1, 0)
        return self.pooling(x).permute(1, 0)


class ConvolutionalSequenceReducer(nn.Module):
    def __init__(self, output_size=10, embed_dim=256):
        super(ConvolutionalSequenceReducer, self).__init__()
        self.output_size = output_size
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc_input_size = 128
        self.fc1 = nn.Linear(self.fc_input_size, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.output_size * self.embed_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.mean(x, dim=(2, 0))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(self.output_size, self.embed_dim)
        return x


class AttentionGatedSequenceReducer(nn.Module):
    def __init__(self, output_size=10, embed_dim=256):
        super(AttentionGatedSequenceReducer, self).__init__()
        self.embed_dim = embed_dim
        self.attn_net_gated = AttentionNetGated(input_dim=self.embed_dim, n_classes=output_size, dropout=False)

    def forward(self, x):
        A_x, x = self.attn_net_gated(x)
        reduced = torch.transpose(A_x, 1, 0)
        return torch.mm(F.softmax(reduced, dim=1), x)


class PerceiverSequenceReducer(nn.Module):
    def __init__(self, embed_dim=256):
        super(PerceiverSequenceReducer, self).__init__()
        self.embed_dim = embed_dim
        self.perceiver = Perceiver(dim=self.embed_dim, queries_dim=self.embed_dim, logits_dim=self.embed_dim)

    def forward(self, x, queries):
        x = x.unsqueeze(0)
        return self.perceiver(x, queries=queries)


def test_pooling_reducer():
    print('Testing PoolingSequenceReducer...')

    output_seq_length = 10
    reducer = PoolingSequenceReducer(output_size=output_seq_length)
    x = torch.randn((2000, 256))
    out = reducer(x)
    assert out.shape[0] == output_seq_length
    assert out.shape[1] == 256
    x = torch.randn((3000, 256))
    out = reducer(x)
    assert out.shape[0] == output_seq_length
    assert out.shape[1] == 256

    print('Test successful')


def test_conv_reducer():
    print('Testing ConvolutionalSequenceReducer...')

    output_seq_length = 10
    reducer = ConvolutionalSequenceReducer(output_size=output_seq_length)
    x = torch.randn((2000, 256))
    out = reducer(x)
    assert out.shape[0] == output_seq_length
    assert out.shape[1] == 256
    x = torch.randn((3000, 256))
    out = reducer(x)
    assert out.shape[0] == output_seq_length
    assert out.shape[1] == 256

    print('Test successful')


def test_attention_reducer():
    print('Testing AttentionGatedSequenceReducer...')

    output_seq_length = 10
    reducer = AttentionGatedSequenceReducer(output_size=output_seq_length)
    x = torch.randn((2000, 256))
    out = reducer(x)
    assert out.shape[0] == output_seq_length
    assert out.shape[1] == 256
    x = torch.randn((3000, 256))
    out = reducer(x)
    assert out.shape[0] == output_seq_length
    assert out.shape[1] == 256

    print('Test successful')


def test_perceiver_reducer():
    print('Testing PerceiverSequenceReducer...')

    reducer = PerceiverSequenceReducer()
    x = torch.randn((2000, 256))
    queries = torch.zeros(10, 256)
    out = reducer(x, queries=queries)
    out = out.squeeze(0)
    assert out.shape[0] == 10
    assert out.shape[1] == 256
    x = torch.randn((3000, 256))
    queries = torch.zeros(10, 256)
    out = reducer(x, queries=queries)
    out = out.squeeze(0)
    assert out.shape[0] == 10
    assert out.shape[1] == 256

    print('Test successful')
