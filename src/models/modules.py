import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, spectral_norm
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
import logging
from typing import Optional, Tuple, List
from functools import partial

import src.utils.commons as commons

logger = logging.getLogger(__name__)

LRELU_SLOPE = 0.1

def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01):
    """Initialize weights with improved initialization."""
    if hasattr(m, 'weight'):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        else:
            m.weight.data.normal_(mean, std)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

@torch.jit.script
def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Compute padding size with JIT optimization."""
    return int((kernel_size * dilation - dilation) / 2)

class TextEncoder(nn.Module):
    """Text encoder with improved architecture."""
    
    def __init__(self, n_vocab: int, out_channels: int, hidden_channels: int,
                 filter_channels: int, n_heads: int, n_layers: int,
                 kernel_size: int, p_dropout: float):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        # Improved embedding initialization
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.kaiming_normal_(self.emb.weight, mode='fan_out', nonlinearity='linear')
        
        # Create encoder
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
            
        # Output projection with proper initialization
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        nn.init.zeros_(self.proj.bias)
        
        # Enable gradient checkpointing for memory efficiency
        self.use_gradient_checkpointing = True

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with improved efficiency."""
        # Compute embedding with proper scaling
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = torch.transpose(x, 1, -1)
        
        # Create mask
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        
        # Apply encoder with gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.encoder,
                x * x_mask, x_mask
            )
        else:
            x = self.encoder(x * x_mask, x_mask)
        
        # Project to output dimensions
        stats = self.proj(x) * x_mask
        
        # Split into mean and log variance
        m, logs = torch.split(stats, self.out_channels, dim=1)
        
        return x, m, logs, x_mask

class Encoder(nn.Module):
    """Transformer encoder with improved efficiency."""
    
    def __init__(self, hidden_channels: int, filter_channels: int,
                 n_heads: int, n_layers: int, kernel_size: int = 1,
                 p_dropout: float = 0.0, window_size: int = 4, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        # Improved dropout
        self.drop = nn.Dropout(p_dropout)
        
        # Create attention and FFN layers
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    activation='gelu'  # Use GELU for better performance
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))
        
        # Initialize weights
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with improved efficiency."""
        # Create attention mask
        attn_mask = None
        if x_mask is not None:
            attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
            x = x * x_mask
        
        # Apply layers with residual connections
        for i in range(self.n_layers):
            # Self-attention
            residual = x
            x = self.norm_layers_1[i](x)
            x = self.attn_layers[i](x, x, attn_mask)
            x = self.drop(x)
            x = residual + x
            
            # Feed-forward
            residual = x
            x = self.norm_layers_2[i](x)
            x = self.ffn_layers[i](x, x_mask)
            x = self.drop(x)
            x = residual + x
        
        # Apply final mask if needed
        if x_mask is not None:
            x = x * x_mask
            
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head attention with improved efficiency."""
    
    def __init__(self, channels: int, out_channels: int, n_heads: int,
                 p_dropout: float = 0., window_size: Optional[int] = None,
                 heads_share: bool = True, block_length: Optional[int] = None,
                 proximal_bias: bool = False, proximal_init: bool = False):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        
        # Create projections with improved initialization
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        nn.init.zeros_(self.conv_o.weight)
        
        # Improved dropout
        self.drop = nn.Dropout(p_dropout)

        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with improved efficiency."""
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        
        return x

    def attention(self, query: torch.Tensor, key: torch.Tensor,
                  value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention with improved efficiency."""
        # Reshape tensors
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        # Compute attention scores with improved stability
        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))

        # Add proximal bias if needed
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
        
        # Apply masking
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                scores = scores.masked_fill(block_mask == 0, -1e4)
        
        # Compute attention weights with improved numerical stability
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        
        # Compute output
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        
        return output, p_attn

    def _attention_bias_proximal(self, length: int) -> torch.Tensor:
        """Compute attention bias for proximal attention."""
        r = torch.arange(length, device=self.conv_q.weight.device)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)

class FFN(nn.Module):
    """Feed-forward network with improved architecture."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 filter_channels: int, kernel_size: int,
                 p_dropout: float = 0.0, activation: Optional[str] = None,
                 causal: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        # Create convolutions with improved initialization
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        nn.init.kaiming_normal_(self.conv_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv_2.weight)
        
        # Improved dropout
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with improved activation functions."""
        x = self.conv_1(self.padding(x))
        
        # Apply activation
        if self.activation == "gelu":
            x = F.gelu(x)
        else:
            x = F.relu(x)
            
        x = self.drop(x)
        x = self.conv_2(self.padding(x))
        
        return x if x_mask is None else x * x_mask

    def _causal_padding(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal padding."""
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x: torch.Tensor) -> torch.Tensor:
        """Apply same padding."""
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

class LayerNorm(nn.Module):
    """Layer normalization with improved stability."""
    
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        # Initialize parameters
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with improved memory efficiency."""
        # Compute statistics
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        
        # Normalize and scale
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
        
        return x

class DurationPredictor(nn.Module):
    """Duration predictor with improved architecture."""
    
    def __init__(self, in_channels: int, hidden_channels: int,
                 kernel_size: int, p_dropout: float,
                 gin_channels: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        # Improved dropout
        self.drop = nn.Dropout(p_dropout)
        
        # Create convolutions with proper initialization
        self.conv_1 = nn.Conv1d(in_channels, hidden_channels, kernel_size,
                               padding=kernel_size//2)
        self.norm_1 = LayerNorm(hidden_channels)
        self.conv_2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                               padding=kernel_size//2)
        self.norm_2 = LayerNorm(hidden_channels)
        self.proj = nn.Conv1d(hidden_channels, 1, 1)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv_1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.proj.weight)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)
            nn.init.zeros_(self.cond.weight)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor,
                g: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with improved activation functions."""
        # Apply conditioning if provided
        if g is not None:
            g = self.cond(g)
            x = x + g
        
        # First conv block
        x = self.conv_1(x * x_mask)
        x = F.gelu(x)  # Use GELU for better performance
        x = self.norm_1(x)
        x = self.drop(x)
        
        # Second conv block
        x = self.conv_2(x * x_mask)
        x = F.gelu(x)  # Use GELU for better performance
        x = self.norm_2(x)
        x = self.drop(x)
        
        # Project to output
        x = self.proj(x * x_mask)
        
        return x * x_mask

class Generator(nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
        # Initialize weights
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        identity = x
        
        x = F.leaky_relu(x, 0.2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = self.norm2(x)
        
        return x + identity

class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        
        # Input normalization
        self.norm = nn.LayerNorm([80])
        
        # Convolutional layers
        channels = [1, 32, 64, 128, 256]
        kernel_size = (3, 1)
        stride = (2, 1)
        padding = (1, 0)
        
        # Initial convolution
        self.pre = nn.Sequential(
            nn.Conv2d(1, channels[1], kernel_size, stride, padding),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1)
        )
        
        # Main convolutions
        self.convs = nn.ModuleList()
        for i in range(1, len(channels) - 1):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size, stride, padding),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(0.1)
            ))
        
        # Final layer
        self.post = nn.Sequential(
            nn.Conv2d(channels[-1], 1, kernel_size, 1, padding),
            nn.Dropout(0.1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x: input mel-spectrogram [B, n_mel, T]
        Returns:
            logits: discriminator output [B, 1]
            fmap: feature maps
        """
        # Create a new tensor for processing
        x = x.detach().clone()
        
        # Normalize input
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        
        # Add channel dimension [B, 1, n_mel, T]
        x = x.unsqueeze(1)
        
        # Handle period-based processing
        b, c, h, t = x.shape
        
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            last_t = x[..., -1:]
            x = torch.cat([x, last_t.repeat(1, 1, 1, n_pad)], dim=-1)
            t = t + n_pad
        
        # Reshape for period processing
        x = x.view(b, c, h, t // self.period, self.period)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(b, c, t // self.period, h * self.period)
        
        # Feature maps
        fmap = []
        
        # Initial convolution
        x = self.pre(x)
        fmap.append(x.clone())
        
        # Main convolutions
        for conv in self.convs:
            x = conv(x)
            fmap.append(x.clone())
        
        # Final layer
        x = self.post(x)
        fmap.append(x.clone())
        
        # Global average pooling
        x = torch.mean(x, dim=[2, 3])
        
        # Clamp output values
        x = torch.clamp(x, min=-100, max=100)
        
        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Fewer discriminators for better stability
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(2),
            PeriodDiscriminator(3),
            PeriodDiscriminator(5)
        ])

    def forward(self, y, y_hat):
        """
        Args:
            y: real mel-spectrogram [B, n_mel, T]
            y_hat: generated mel-spectrogram [B, n_mel, T]
        Returns:
            y_d_hat_r: discriminator outputs for real [B, n_discriminators]
            y_d_hat_g: discriminator outputs for generated [B, n_discriminators]
            fmap_r: feature maps for real
            fmap_g: feature maps for generated
        """
        y_d_hat_r = []
        y_d_hat_g = []
        fmap_r = []
        fmap_g = []

        for d in self.discriminators:
            y_d_r, fmap_r_i = d(y)
            y_d_g, fmap_g_i = d(y_hat)
            y_d_hat_r.append(y_d_r)
            y_d_hat_g.append(y_d_g)
            fmap_r.extend(fmap_r_i)
            fmap_g.extend(fmap_g_i)

        y_d_hat_r = torch.cat(y_d_hat_r, dim=1)  # [B, n_discriminators]
        y_d_hat_g = torch.cat(y_d_hat_g, dim=1)  # [B, n_discriminators]

        return y_d_hat_r, y_d_hat_g, fmap_r, fmap_g