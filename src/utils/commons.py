import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q) with improved numerical stability."""
    # Clamp the log variances to avoid numerical issues
    logs_p = torch.clamp(logs_p, min=-10, max=10)
    logs_q = torch.clamp(logs_q, min=-10, max=10)
    
    # Compute terms separately for better stability
    var_p = torch.exp(2. * logs_p)
    var_q = torch.exp(2. * logs_q)
    var_ratio = var_p / (var_q + 1e-7)
    
    # Mean difference term
    mean_diff_sq = (m_p - m_q) ** 2
    mean_diff_term = mean_diff_sq / (var_q + 1e-7)
    
    # Log variance difference
    log_var_diff = 2. * (logs_q - logs_p)
    
    # Combine terms
    kl = 0.5 * (var_ratio + mean_diff_term + log_var_diff - 1.0)
    
    # Ensure the result is finite
    kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
    return kl

def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = torch.rand(shape)
    return -torch.log(-torch.log(uniform_samples + 1e-20) + 1e-20)

def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g

def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """Random slice segments from a batch of sequences."""
    b, c, t = x.size()
    
    if x_lengths is None:
        x_lengths = torch.full((b,), t, device=x.device, dtype=torch.long)
    else:
        x_lengths = x_lengths.to(dtype=torch.long)
    
    # Make sure segment_size is not larger than the shortest sequence
    segment_size = min(segment_size, torch.min(x_lengths).item())
    
    # Calculate maximum starting position for each sequence
    ids_str_max = x_lengths - segment_size + 1
    ids_str_max = torch.clamp(ids_str_max, min=1)  # Ensure at least 1
    
    # Generate random starting positions
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    
    # Create segments
    segments = torch.zeros(b, c, segment_size, device=x.device)
    for i in range(b):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        # Handle case where sequence is shorter than segment_size
        if idx_end > x_lengths[i]:
            idx_str = x_lengths[i] - segment_size
            idx_end = x_lengths[i]
        segments[i] = x[i, :, idx_str:idx_end]
    
    return segments, ids_str

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1))
    inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment)
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)

def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)

def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """Fused add, tanh, sigmoid multiply."""
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts

def convert_pad_shape(pad_shape):
    """Convert padding shape to PyTorch format."""
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def sequence_mask(length, max_length=None):
    """Create sequence mask."""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    device = duration.device
    
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)
    
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2,3) * mask
    return path 