import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator with improved stability and performance."""
    
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        periods = [2, 3, 5, 7, 11]  # Prime numbers for better coverage
        
        # Create discriminators with different periods
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period, use_spectral_norm)
            for period in periods
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights with improved initialization."""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @torch.jit.ignore
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with improved efficiency.
        Args:
            y: real mel-spectrogram [B, n_mel, T]
            y_hat: generated mel-spectrogram [B, n_mel, T]
        Returns:
            Tuple containing discriminator outputs and feature maps
        """
        if not torch.is_tensor(y) or not torch.is_tensor(y_hat):
            raise ValueError("Inputs must be torch tensors")
            
        if y.shape != y_hat.shape:
            raise ValueError(f"Shape mismatch: {y.shape} vs {y_hat.shape}")
        
        y_d_hat_r, y_d_hat_g = [], []
        fmap_r, fmap_g = [], []

        # Process in parallel if possible
        if torch.cuda.device_count() > 1:
            results = nn.parallel.parallel_apply(
                self.discriminators,
                [(y,), (y_hat,)] * len(self.discriminators)
            )
            for i in range(0, len(results), 2):
                y_d_r, fmap_r_i = results[i]
                y_d_g, fmap_g_i = results[i + 1]
                y_d_hat_r.append(y_d_r)
                y_d_hat_g.append(y_d_g)
                fmap_r.extend(fmap_r_i)
                fmap_g.extend(fmap_g_i)
        else:
            for d in self.discriminators:
                y_d_r, fmap_r_i = d(y)
                y_d_g, fmap_g_i = d(y_hat)
                y_d_hat_r.append(y_d_r)
                y_d_hat_g.append(y_d_g)
                fmap_r.extend(fmap_r_i)
                fmap_g.extend(fmap_g_i)

        # Concatenate results efficiently
        y_d_hat_r = torch.cat(y_d_hat_r, dim=1)
        y_d_hat_g = torch.cat(y_d_hat_g, dim=1)

        return y_d_hat_r, y_d_hat_g, fmap_r, fmap_g

class PeriodDiscriminator(nn.Module):
    """Period-based discriminator with improved architecture."""
    
    def __init__(self, period: int, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        # Improved architecture with better scaling
        channels = [1, 32, 128, 512, 1024, 1024]
        kernel_size = (5, 1)
        stride = (3, 1)
        padding = (2, 0)
        
        # Create convolutional layers with improved initialization
        self.convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.convs.append(
                norm_f(nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size,
                    stride,
                    padding,
                    bias=True
                ))
            )
        
        # Post-processing conv with proper initialization
        self.conv_post = norm_f(nn.Conv2d(channels[-1], 1, (3, 1), 1, (1, 0)))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Layer normalization for stability
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm([ch, 1]) for ch in channels[1:]
        ])

    @torch.jit.ignore
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with improved stability and efficiency.
        Args:
            x: input mel-spectrogram [B, n_mel, T]
        Returns:
            Tuple of discriminator output and feature maps
        """
        try:
            # Add channel dimension and handle padding
            x = x.unsqueeze(1)
            b, c, h, t = x.shape
            
            # Efficient padding
            if t % self.period != 0:
                n_pad = self.period - (t % self.period)
                x = F.pad(x, (0, n_pad), mode='reflect')
                t = t + n_pad
            
            # Reshape for period processing
            x = x.view(b, c, h, t // self.period, self.period)
            x = x.permute(0, 1, 3, 2, 4).contiguous()
            x = x.view(b, c, t // self.period, h * self.period)
            
            # Feature maps with improved processing
            fmap = []
            
            # Apply convolutional layers with residual connections
            for i, (conv, norm) in enumerate(zip(self.convs, self.norm_layers)):
                # Apply convolution
                x_conv = conv(x)
                
                # Apply normalization and activation
                x_conv = norm(x_conv.transpose(1, 2)).transpose(1, 2)
                x_conv = F.leaky_relu(x_conv, 0.1)
                
                # Apply dropout
                x_conv = self.dropout(x_conv)
                
                # Store feature map
                fmap.append(x_conv)
                
                # Update main tensor
                x = x_conv
            
            # Post-processing
            x = self.conv_post(x)
            fmap.append(x)
            
            # Efficient global pooling
            x = torch.flatten(x, 1, -1)
            x = torch.mean(x, dim=-1, keepdim=True)
            
            return x, fmap
            
        except Exception as e:
            logger.error(f"Error in PeriodDiscriminator forward pass: {e}")
            raise

class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator with improved architecture."""
    
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        # Create discriminators at different scales
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=use_spectral_norm)
            for _ in range(3)
        ])
        
        # Improved pooling layers
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @torch.jit.ignore
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """Forward pass with improved efficiency."""
        y_d_rs, y_d_gs = [], []
        fmap_rs, fmap_gs = [], []
        
        # Process each scale
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(nn.Module):
    """Scale-specific discriminator with improved architecture."""
    
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        # Improved architecture
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))
        
        # Add batch normalization for stability
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(ch) for ch in [16, 64, 256, 1024, 1024, 1024]
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    @torch.jit.ignore
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with improved stability."""
        fmap = []
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x)
            x = bn(x)
            x = F.leaky_relu(x, 0.1)
            x = self.dropout(x)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap

def feature_loss(fmap_r: List[torch.Tensor], fmap_g: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
    """Compute feature matching loss with optional feature weighting."""
    if weights is None:
        weights = [1.0] * len(fmap_r)
    
    loss = 0
    for dr, dg, w in zip(fmap_r, fmap_g, weights):
        for rl, gl in zip(dr, dg):
            loss += w * F.l1_loss(rl, gl)
    
    return loss * 2

def discriminator_loss(disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[float], List[float]]:
    """Compute discriminator loss with improved stability."""
    loss = 0
    r_losses, g_losses = [], []
    
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        # Use label smoothing for better stability
        real_labels = torch.ones_like(dr) * 0.9  # Smooth real labels to 0.9
        fake_labels = torch.zeros_like(dg) + 0.1  # Smooth fake labels to 0.1
        
        r_loss = F.mse_loss(dr, real_labels)
        g_loss = F.mse_loss(dg, fake_labels)
        
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    
    return loss, r_losses, g_losses

def generator_loss(disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Compute generator loss with improved stability."""
    loss = 0
    gen_losses = []
    
    for dg in disc_outputs:
        # Use label smoothing
        real_labels = torch.ones_like(dg) * 0.9
        l = F.mse_loss(dg, real_labels)
        gen_losses.append(l)
        loss += l
    
    return loss, gen_losses

LRELU_SLOPE = 0.1

@torch.jit.script
def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Compute padding size with JIT optimization."""
    return int((kernel_size * dilation - dilation) / 2) 