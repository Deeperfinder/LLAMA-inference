import torch
import torch.nn as nn

import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    intermediate_size: int = 11008

    vocab_size: int = 151936
    dec_voc_size: int = 151936
    drop_prob: int = 0.1

class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super(MultiHeadAttention, self).__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.w_q = nn.Linear(self.dim, self.dim, bias=False)
        self.w_k = nn.Linear(self.dim, self.dim, bias=False)
        self.w_v = nn.Linear(self.dim, self.dim, bias=False)
        self.w_o = nn.Linear(self.dim, self.dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        enc : Optional[torch.Tensor],
        start_pos: Optional[int],
        mask: Optional[torch.Tensor],
        # mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module

        Args:
            x: (torch.Tensor) : Input tensor.
            start_pos (int) : Starting position for caching
            freqs_cis (torch.Tensor) : Precomputed frequency tensor.
            mask (torch.Tensor) : Mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention
        
        """
        bsz, seq_len, _ = x.shape
        if enc is not None:
            q = self.w_q(x)
            k,v = self.w_k(enc), self.w_v(enc)
        else:
            q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis= freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            # mask = torch.tril(torch.ones(seq_len, seq_len, dtype=bool))
            # mask为0的地方，值为true，指示哪些位置是有效的，哪些位置是无效的。
            # masked_fill在conditoin为true的位置上，将对应的score值替换为"-inf"
            score = score.masked_fill(mask == 0, float("-inf"))
        score = self.softmax(score.float()).type_as(q) 
        output = torch.matmul(score, v)
        output = output.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.dim)

        output = self.w_o(output)
        return output

class TokenEmbedding(nn.Embedding):
    def __init__(self, args: ModelArgs):
        super(TokenEmbedding, self).__init__(args.vocab_size, args.dim, padding_idx=1)

class PositionalEncoding(nn.Module):
    """
    Positional encoding module.
    dimension = (batch, seq_len, d_model)
    """
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zero(max_len, d_model).to(device)
        self.encoding.required_grad = False

        pos = torch.arange(0, max_len, device)
        pos = pos.float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2, device)
        
        self.encoding[:, 0::2] = torch.sin(pos / (10000 **(_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 **(_2i / d_model)))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]

class RMSNorm(nn.Module):
    """
    RMSNorm module.
    为了保留模型的表达能力,RMSNorm引入了可学习的参数w和平移参数b,(通常只用w)
    y = w * x + b
    """
    def __init__ (self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normed = x * rms
        return self.weight * x_normed
    
# torch.nn.modules.TransformerDecoder
# x = self.linear2(self.dropout(self.activation(self.linear1(x))))
class PositionwiseFeedForward(nn.Module):
    """
    posirionwise feedforward module.
    """
    def __init__(self, d_model, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc2(self.dropout(nn.GELU(self.fc1(x))))
        return self.dropout(x)
    
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor (end, dim // 2): Precomputed frequency tensor with complex exponentials.

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 时间步长 (0 -> end)
    # (end, dim // 2)
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key.
    Args:
        xq: query tensor, shape (batch, seq_len, n_heads, head_dim)
        xk: key tensor, shape (batch, seq_len, n_heads, head_dim)
        freqs_cis: cos/sin frequencies, shape (d_model//2, end)
    Returns:
        xq: query tensor, shape (batch, seq_len, d_model)
        xk: key tensor, shape (batch, seq_len, d_model)
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class EncoderLayer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(args)
        self.norm1 = RMSNorm(args.dim)
        self.drop1 = nn.Dropout(args.drop_prob)

        self.ffn = PositionwiseFeedForward(args.dim, args.intermediate_size, args.drop_prob)
        self.norm2 = RMSNorm(args.dim)
        self.drop2 = nn.Dropout(args.drop_prob)

    def forward(self, x, mask):
        _x = x
        x = self.attention(x, mask=mask)
        
        x = self.drop1(x)
        x = self.norm1(x + _x)
        
        _x = x
        x = self.ffn(x)
        x = self.norm2(x + _x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(args)
        self.norm1 = RMSNorm(args.dim)
        self.drop1 = nn.Dropout(args.drop_prob)


        self.cross_attention = MultiHeadAttention(args)
        self.drop2 = nn.Dropout(args.drop_prob)
        self.norm2 = RMSNorm(args.dim)

        self.ffn = PositionwiseFeedForward(args.dim, args.intermediate_size, args.drop_prob)
        self.norm3 = RMSNorm(args.dim)
        self.drop3 = nn.Dropout(args.drop_prob)

    def forward(self, dec, enc, padding_mask, mask):
        _x = dec
        x = self.attention(x, mask=mask) # 下三角矩阵
        x = self.drop1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            x = self.cross_attention(x, enc, enc) # 对位置的掩码
            x = self.drop2(x)
            x = self.norm2(x + _x)
        _x = x
        x = self.ffn(x)
        x = self.drop3(x)
        x = self.norm3(x + _x)
        
if __name__ == "__main__":
    # test MultiHeadAttention
    X = torch.randn(128, 16, 4096)
    args_ = ModelArgs()
    freq_cis = precompute_freqs_cis(args_.dim // args_.n_heads, args_.max_seq_len*2)
    freq_cis = freq_cis[0 : 16]
    attention = MultiHeadAttention(ModelArgs())
    output = attention(X, freq_cis, 0)
    print(output, output.shape)