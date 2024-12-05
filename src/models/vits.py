import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.utils.weight_norm as weight_norm

import src.utils.commons as commons
import src.utils.utils as utils
from src.models.modules import Encoder, MultiPeriodDiscriminator, Generator, DurationPredictor, TextEncoder

class VITS(nn.Module):
    """
    VITS: Conditional Variational Autoencoder with Adversarial Learning
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.segment_size = config.model.segment_size
        
        # Initialize components with improved initialization
        self.enc_p = TextEncoder(
            n_vocab=config.model.n_vocab,
            out_channels=config.model.hidden_channels,
            hidden_channels=config.model.hidden_channels,
            filter_channels=config.model.filter_channels,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            kernel_size=config.model.kernel_size,
            p_dropout=config.model.p_dropout)
            
        self.dec = Generator(
            initial_channel=config.model.hidden_channels,
            resblock=config.model.resblock, 
            resblock_kernel_sizes=config.model.resblock_kernel_sizes,
            resblock_dilation_sizes=config.model.resblock_dilation_sizes,
            upsample_rates=config.model.upsample_rates,
            upsample_initial_channel=config.model.upsample_initial_channel,
            upsample_kernel_sizes=config.model.upsample_kernel_sizes,
            gin_channels=config.model.gin_channels)
            
        self.enc_q = nn.Sequential(
            weight_norm(nn.Conv1d(config.data.n_mel_channels, config.model.hidden_channels, 1)),
            Encoder(
                hidden_channels=config.model.hidden_channels,
                filter_channels=config.model.filter_channels,
                n_heads=config.model.n_heads,
                n_layers=config.model.n_layers,
                kernel_size=config.model.kernel_size,
                p_dropout=config.model.p_dropout,
                window_size=4)
        )
            
        self.flow = ResidualCouplingBlock(
            channels=config.model.hidden_channels,
            hidden_channels=config.model.hidden_channels,
            kernel_size=config.model.kernel_size,
            dilation_rate=1,
            n_layers=config.model.n_layers,
            gin_channels=config.model.gin_channels)
            
        self.dp = DurationPredictor(
            in_channels=config.model.hidden_channels,
            hidden_channels=config.model.hidden_channels,
            kernel_size=config.model.kernel_size,
            p_dropout=config.model.p_dropout,
            gin_channels=config.model.gin_channels)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Enable gradient checkpointing for memory efficiency
        self.use_gradient_checkpointing = True
    
    def _init_weights(self, module):
        """Initialize network weights with improved initialization."""
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, text_padded, input_lengths, mel_padded, output_lengths):
        """Forward pass for training with improved memory efficiency."""
        # Create masks
        x_mask = torch.unsqueeze(commons.sequence_mask(input_lengths, text_padded.size(1)), 1).to(text_padded.dtype)
        y_mask = torch.unsqueeze(commons.sequence_mask(output_lengths, mel_padded.size(2)), 1).to(mel_padded.dtype)
        
        # Text encoder with gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            x, m_p, logs_p, _ = torch.utils.checkpoint.checkpoint(
                self.enc_p, text_padded, input_lengths)
        else:
            x, m_p, logs_p, _ = self.enc_p(text_padded, input_lengths)
        
        # Clamp values for numerical stability
        m_p = torch.clamp(m_p, min=-100, max=100)
        logs_p = torch.clamp(logs_p, min=-100, max=100)
        
        # Posterior encoder with gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            z = torch.utils.checkpoint.checkpoint(
                self.enc_q, mel_padded)
        else:
            z = self.enc_q(mel_padded)
        
        # Flow
        z_p = self.flow(z, y_mask)
        
        # Duration predictor
        w = self.dp(x, x_mask)
        w = torch.clamp(w, min=-100, max=100) * x_mask
        
        # Expand prior
        m_p = torch.matmul(w, m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(w, logs_p.transpose(1, 2)).transpose(1, 2)
        
        # Sample with improved handling
        z_slice, ids_slice = commons.rand_slice_segments(z, output_lengths, self.segment_size)
        
        # Generator with gradient checkpointing
        if self.use_gradient_checkpointing and self.training:
            o = torch.utils.checkpoint.checkpoint(
                self.dec, z_slice)
        else:
            o = self.dec(z_slice)
        
        # Ensure output has correct dimensions
        if o.size(1) != mel_padded.size(1):
            o = o.repeat(1, mel_padded.size(1), 1)
        
        # Pad or trim output to match target
        if o.size(2) > mel_padded.size(2):
            o = o[:, :, :mel_padded.size(2)]
        elif o.size(2) < mel_padded.size(2):
            o = F.pad(o, (0, mel_padded.size(2) - o.size(2)))
        
        # Clamp output values
        o = torch.clamp(o, min=-100, max=100)
        
        return {
            'z': z,
            'z_p': z_p,
            'z_slice': z_slice,
            'mel_output': o,
            'm_p': m_p,
            'logs_p': logs_p,
            'x_mask': x_mask,
            'y_mask': y_mask,
            'w': w
        }

    @torch.inference_mode()
    def infer(self, text_padded, input_lengths, noise_scale=0.667, length_scale=1.0):
        """Forward pass for inference with improved efficiency."""
        # Create mask
        x_mask = torch.unsqueeze(commons.sequence_mask(input_lengths, text_padded.size(1)), 1).to(text_padded.dtype)
        
        # Text encoder
        x, m_p, logs_p, _ = self.enc_p(text_padded, input_lengths)
        
        # Duration predictor
        w = self.dp(x, x_mask)
        w = w * x_mask * length_scale
        
        # Expand prior
        m_p = torch.matmul(w, m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(w, logs_p.transpose(1, 2)).transpose(1, 2)
        
        # Sample with noise scaling
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, None, reverse=True)
        
        # Generate audio
        o = self.dec(z)
        
        # Apply tanh and normalize
        o = torch.tanh(o)
        o = utils.normalize_audio(o.cpu().numpy(), target_level=-27)
        o = torch.from_numpy(o).to(z.device)
        
        return o, x_mask, z

class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        
        # Create flow layers with improved initialization
        self.flows = nn.ModuleList()
        for i in range(4):
            self.flows.append(
                ResidualCouplingLayer(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    mean_only=True
                )
            )
            self.flows.append(Flip())
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, x_mask=None, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.mean_only = mean_only
        
        # Create layers with weight normalization
        self.pre = weight_norm(nn.Conv1d(channels // 2, hidden_channels, 1))
        self.enc = WN(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            n_layers=n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels
        )
        
        # Initialize post conv with zeros for stability
        self.post = weight_norm(nn.Conv1d(hidden_channels, channels if not mean_only else channels // 2, 1))
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, x, x_mask=None, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.channels // 2] * 2, 1)
        h = self.pre(x0) * x_mask if x_mask is not None else self.pre(x0)
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask if x_mask is not None else self.post(h)
        
        if not self.mean_only:
            m, logs = torch.split(stats, [self.channels // 2] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) if not self.mean_only else m + x1
            x = torch.cat([x0, x1], 1)
        else:
            x1 = (x1 - m) * torch.exp(-logs) if not self.mean_only else x1 - m
            x = torch.cat([x0, x1], 1)
        return x

class Flip(nn.Module):
    def forward(self, x, *args, **kwargs):
        return torch.flip(x, [1])

class WN(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super().__init__()
        assert(kernel_size % 2 == 1)
        assert(hidden_channels % 2 == 0)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
            self.cond_layer = weight_norm(cond_layer)

        # Create layers with improved initialization
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            
            in_layer = weight_norm(nn.Conv1d(
                hidden_channels, 
                2*hidden_channels, 
                kernel_size,
                dilation=dilation, 
                padding=padding
            ))
            self.in_layers.append(in_layer)

            # Last layer has different number of channels
            if i < n_layers - 1:
                res_skip_channels = 2*hidden_channels
            else:
                res_skip_channels = hidden_channels
                
            res_skip_layer = weight_norm(nn.Conv1d(hidden_channels, res_skip_channels, 1))
            self.res_skip_layers.append(res_skip_layer)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, x_mask=None, g=None):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset+2*self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(
                x_in,
                g_l,
                n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:,:self.hidden_channels,:]
                x = (x + res_acts) * x_mask if x_mask is not None else (x + res_acts)
                output = output + res_skip_acts[:,self.hidden_channels:,:]
            else:
                output = output + res_skip_acts
        
        return output * x_mask if x_mask is not None else output

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts