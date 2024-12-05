import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Encoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0, window_size=4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
            self.norm_layers_1.append(nn.LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(nn.LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        return x * x_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0, window_size=None, heads_share=True):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if not self.heads_share:
            nn.init.xavier_uniform_(self.conv_o.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        
        x = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, d_k, t]
        b, d, t_s = key.size()
        t_t = query.size(2)
        
        query = rearrange(query, 'b (h k) t -> b h k t', h=self.n_heads)
        key = rearrange(key, 'b (h k) t -> b h k t', h=self.n_heads)
        value = rearrange(value, 'b (h k) t -> b h k t', h=self.n_heads)

        scores = torch.matmul(query.transpose(-2, -1), key) / math.sqrt(self.k_channels)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value.transpose(-2, -1))
        output = rearrange(output, 'b h t k -> b (h k) t')
        return output


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_vocab = config.model.n_vocab
        hidden_channels = config.model.hidden_channels
        filter_channels = config.model.filter_channels
        filter_kernel_size = config.model.filter_kernel_size
        n_heads = config.model.n_heads
        n_layers = config.model.n_layers
        kernel_size = config.model.kernel_size
        p_dropout = config.model.p_dropout
        
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            filter_kernel_size,
            n_layers,
            n_heads,
            kernel_size,
            p_dropout)
        
    def forward(self, x, x_lengths):
        # Create padding mask
        x_mask = torch.zeros_like(x).to(x.device).float()
        for i in range(x.size(0)):
            x_mask[i, :x_lengths[i]] = 1.0
        x_mask = x_mask.unsqueeze(1)  # [B, 1, T]
        
        # Embedding
        x = self.emb(x) * math.sqrt(self.emb.embedding_dim)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        
        # Apply encoder
        x = self.encoder(x * x_mask, x_mask)
        
        return x, x_mask

def sequence_mask(lengths, maxlen):
    """Create sequence mask for padding."""
    batch_size = lengths.numel()
    max_len = maxlen or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1))) 