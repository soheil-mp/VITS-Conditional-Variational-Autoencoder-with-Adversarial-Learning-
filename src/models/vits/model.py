import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from .text_encoder import TextEncoder
from .posterior_encoder import PosteriorEncoder
from .flow_decoder import ResidualCouplingBlock
from .discriminator import MultiPeriodDiscriminator

class VITS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Text encoder
        self.text_encoder = TextEncoder(config)
        
        # Posterior encoder
        self.posterior_encoder = PosteriorEncoder(config)
        
        # Flow decoder
        self.flow = ResidualCouplingBlock(config)
        
        # Discriminator
        self.discriminator = MultiPeriodDiscriminator()

    def forward(self, text_inputs, text_lengths, mel_inputs, mel_lengths):
        """
        Forward pass
        Args:
            text_inputs: [B, T_text]
            text_lengths: [B]
            mel_inputs: [B, n_mel, T_mel]
            mel_lengths: [B]
        Returns:
            Dictionary containing model outputs
        """
        # Text encoder
        text_enc, text_mask = self.text_encoder(text_inputs, text_lengths)
        
        # Posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(mel_inputs, mel_lengths)
        
        # Flow decoder
        z_p = self.flow(z, y_mask, g=None)
        
        # Discriminator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(mel_inputs, z_p)
        
        outputs = {
            'z': z,
            'z_p': z_p,
            'm_q': m_q,
            'logs_q': logs_q,
            'y_mask': y_mask,
            'y_d_hat_r': y_d_hat_r,
            'y_d_hat_g': y_d_hat_g,
            'fmap_r': fmap_r,
            'fmap_g': fmap_g,
            'text_enc': text_enc,
            'text_mask': text_mask
        }
        
        return outputs

    def infer(self, text_inputs, text_lengths):
        """
        Inference
        Args:
            text_inputs: [B, T_text]
            text_lengths: [B]
        Returns:
            z: generated audio
            mask: text mask
            None: placeholder for compatibility
        """
        # Text encoder
        text_enc, text_mask = self.text_encoder(text_inputs, text_lengths)
        
        # Generate random noise
        z_p = torch.randn(text_inputs.size(0), self.config.model.hidden_channels, text_mask.size(-1)).to(text_inputs.device)
        
        # Flow decoder (reverse)
        z = self.flow(z_p, text_mask, g=None, reverse=True)
        
        return z, text_mask, None

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        """
        Voice conversion
        Args:
            y: Source mel-spectrogram [B, n_mel, T]
            y_lengths: Source mel-spectrogram lengths [B]
            sid_src: Source speaker ID
            sid_tgt: Target speaker ID
        """
        assert self.n_speakers > 0, "Model must be trained with multiple speakers for voice conversion"
        
        # Encode source mel-spectrogram
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths)
        
        # Convert speaker embedding
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        
        # Apply flow
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        
        # Decode to audio
        o_hat = self.decoder(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)

def sequence_mask(lengths, maxlen=None):
    """Generate sequence mask"""
    if maxlen is None:
        maxlen = lengths.max()
    x = torch.arange(maxlen, dtype=lengths.dtype, device=lengths.device)
    return x.unsqueeze(0) < lengths.unsqueeze(1) 