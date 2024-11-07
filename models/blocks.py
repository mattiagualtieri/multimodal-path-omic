import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import softmax, dropout
from torch.nn.functional import _in_projection_packed, linear

from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_


class AttentionNetGated(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, dropout_p: bool = True, n_classes: int = 1):
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
        if dropout_p:
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


class PreGatingContextualAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device=None, dtype=None, dropout_p: float = 0.25) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout_p
        self.batch_first = False
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        self.bias_k = self.bias_v = None

        self.add_zero_attn = False

        self.CAG = ContextualAttentionGate(dim=self.embed_dim, hidden_dim=self.embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        q_hat, attn_output, attn_output_weights = multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal)
        c = self.CAG(query, q_hat.squeeze(0))
        return attn_output + c, attn_output_weights


def multi_head_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[torch.Tensor],
    in_proj_bias: Optional[torch.Tensor],
    bias_k: Optional[torch.Tensor],
    bias_v: Optional[torch.Tensor],
    dropout_p: float,
    out_proj_weight: torch.Tensor,
    out_proj_bias: Optional[torch.Tensor],
    training: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

    query = query.unsqueeze(1)
    key = key.unsqueeze(1)
    value = value.unsqueeze(1)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads

    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    assert bias_k is None
    assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath and swears) calculate attention and out projection
    #

    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)

    assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"

    attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    P = torch.matmul(torch.tanh(q), torch.tanh(k.transpose(-2, -1))) + 1
    P = P / 2
    attn_output_weights = attn_output_weights * P
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    if dropout_p > 0.0:
        attn_output_weights = dropout(attn_output_weights, p=dropout_p, train=training)

    attn_output = torch.bmm(attn_output_weights, v)

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    # optionally average attention weights over heads
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    if average_attn_weights:
        attn_output_weights = attn_output_weights.mean(dim=1)

    # squeeze the output if input was unbatched
    attn_output = attn_output.squeeze(1)
    attn_output_weights = attn_output_weights.squeeze(0)
    return q, attn_output, attn_output_weights


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
        scores = (QK) / self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        Q_hat = torch.matmul(attention_weights, V)

        return Q, Q_hat, attention_weights


class ContextualAttentionGate(nn.Module):
    def __init__(self, dim: int = 256, hidden_dim: int = 128):
        super(ContextualAttentionGate, self).__init__()
        # FC Layer for Q
        self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ELU())
        # First FC Layer for Q_hat
        self.fc2 = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ELU())
        # Second FC Layer for Q_hat
        self.fc3 = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ELU())

        self.G = nn.Sequential(nn.ELU(), nn.LayerNorm(hidden_dim))
        self.E = nn.Sequential(nn.ELU(), nn.LayerNorm(hidden_dim))

        self.fc_c = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU())

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

        # return Q + C, attention_weights
        return Q, attention_weights


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
