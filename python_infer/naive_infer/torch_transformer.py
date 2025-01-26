import torch
import torch.nn as nn

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
    def __init__ (self, d_model, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
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
        x = self.fc2(self.dropout(F.gelu(self.fc1(x))))
        return self.dropout(x)
    
    