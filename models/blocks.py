import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import softmax, dropout
from torch.nn.functional import _in_projection_packed, _in_projection, pad, linear

from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
# from torch.nn.modules.activation import _is_make_fx_tracing, _check_arg_device, _arg_requires_grad


class AttentionNetGated(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, dropout: bool = True, n_classes: int = 1):
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


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim

        self.num_heads = num_heads
        self.dropout = 0.0
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
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        why_not_fast_path = ''
        if ((attn_mask is not None and torch.is_floating_point(attn_mask))
           or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = None

        attn_mask = None

        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            # elif _is_make_fx_tracing():
            #     why_not_fast_path = "we are running make_fx tracing"
            # elif not all(_check_arg_device(x) for x in tensor_args):
            #     why_not_fast_path = ("some Tensor argument's device is neither one of "
            #                          f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}")
            # elif torch.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
            #     why_not_fast_path = ("grad is enabled and at least one of query or the "
            #                          "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # THIS ONE IS CALLED
        q_hat, attn_output, attn_output_weights = multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
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
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: torch.Tensor,
    out_proj_bias: Optional[torch.Tensor],
    training: bool = True,
    key_padding_mask: Optional[torch.Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[torch.Tensor] = None,
    k_proj_weight: Optional[torch.Tensor] = None,
    v_proj_weight: Optional[torch.Tensor] = None,
    static_k: Optional[torch.Tensor] = None,
    static_v: Optional[torch.Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

    query = query.unsqueeze(1)
    key = key.unsqueeze(1)
    value = value.unsqueeze(1)
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = None

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = None

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    if need_weights:
        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)

        assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        P = (torch.matmul(torch.tanh(q), torch.tanh(k.transpose(-2, -1))) + 1) / 2
        attn_output_weights = attn_output_weights + P
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = dropout(attn_output_weights, p=dropout_p)

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
