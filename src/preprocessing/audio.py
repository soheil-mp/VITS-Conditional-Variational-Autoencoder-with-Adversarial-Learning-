import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional
import warnings
import logging
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio processing class with optimized mel-spectrogram conversion."""
    
    def __init__(self, config):
        self.config = config
        
        # Mel-spectrogram parameters
        self.sr = config.data.sampling_rate
        self.n_fft = config.data.filter_length
        self.hop_length = config.data.hop_length
        self.win_length = config.data.win_length
        self.n_mels = config.data.n_mel_channels
        self.fmin = config.data.mel_fmin
        self.fmax = config.data.mel_fmax
        self.segment_size = config.model.segment_size
        
        # Dynamic range compression
        self.clip_val = getattr(config.data, 'clip_val', 1e-5)
        self.top_db = getattr(config.data, 'top_db', 80.0)
        
        # Create mel filter bank
        self.mel_basis = self._create_mel_basis()
        
        # Create and cache window function
        self.window = torch.hann_window(self.win_length).float()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mel_basis = self.mel_basis.to(self.device)
        self.window = self.window.to(self.device)
        
        # Enable faster audio resampling
        if hasattr(torchaudio.transforms, 'Resample'):
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=48000,  # Default audio sample rate
                new_freq=self.sr
            ).to(self.device)
    
    def _create_mel_basis(self) -> torch.Tensor:
        """Create mel filter bank with improved initialization."""
        try:
            # Create mel basis with librosa
            mel_basis = librosa.filters.mel(
                sr=self.sr,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                htk=True  # Use HTK formula for better accuracy
            )
            
            # Convert to torch tensor
            mel_basis = torch.from_numpy(mel_basis).float()
            
            # Normalize mel basis
            mel_basis = torch.nn.functional.normalize(mel_basis, p=1, dim=1)
            
            return mel_basis
            
        except Exception as e:
            logger.error(f"Error creating mel basis: {e}")
            return torch.ones(self.n_mels, self.n_fft // 2 + 1)
    
    @staticmethod
    def normalize_audio(wav: torch.Tensor, target_level: float = -27.0) -> torch.Tensor:
        """Normalize audio with improved peak normalization."""
        if wav.size(0) == 0:
            return wav
            
        try:
            # Calculate RMS energy
            rms = torch.sqrt(torch.mean(wav ** 2))
            scalar = (10 ** (target_level / 20)) / (rms + 1e-10)
            wav = wav * scalar
            
            # Apply soft clipping for better audio quality
            wav = torch.tanh(wav)
            
            return wav
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return wav
    
    def load_wav(self, path: str, normalize: bool = True) -> Tuple[torch.Tensor, int]:
        """Load and process audio file with improved handling."""
        try:
            # Check if file exists
            if not Path(path).exists():
                raise FileNotFoundError(f"Audio file not found: {path}")
            
            # Load audio with torchaudio for faster loading
            wav, sr = torchaudio.load(path)
            wav = wav.squeeze(0).to(self.device)
            
            # Resample if necessary
            if sr != self.sr and hasattr(self, 'resampler'):
                self.resampler.orig_freq = sr
                wav = self.resampler(wav)
                sr = self.sr
            elif sr != self.sr:
                wav = librosa.resample(wav.cpu().numpy(), orig_sr=sr, target_sr=self.sr)
                wav = torch.from_numpy(wav).float().to(self.device)
                sr = self.sr
            
            # Normalize audio
            if normalize:
                wav = self.normalize_audio(wav)
            
            return wav, sr
            
        except Exception as e:
            logger.error(f"Error loading audio file {path}: {e}")
            return torch.zeros(1, device=self.device), self.sr
    
    @torch.no_grad()
    def get_mel(self, wav: torch.Tensor, center: bool = True) -> torch.Tensor:
        """Convert waveform to mel-spectrogram with improved efficiency."""
        if wav.size(0) == 0:
            return torch.zeros(self.n_mels, 1, device=self.device)
            
        try:
            # Move input to device
            wav = wav.to(self.device)
            
            # Ensure audio length is multiple of hop_length
            if wav.size(0) % self.hop_length != 0:
                pad_len = self.hop_length - (wav.size(0) % self.hop_length)
                wav = torch.nn.functional.pad(wav, (0, pad_len), mode='constant', value=0)
            
            # Compute STFT with optimized parameters
            stft = torch.stft(
                wav,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=center,
                pad_mode='constant',
                normalized=False,
                onesided=True,
                return_complex=True
            )
            
            # Convert to power spectrogram
            magnitudes = torch.abs(stft).pow(2)
            
            # Convert to mel-scale with parallel computation
            mel = torch.matmul(self.mel_basis, magnitudes)
            
            # Dynamic range compression
            mel = torch.log(torch.clamp(mel, min=float(self.clip_val)))
            
            # Normalize mel spectrogram
            mel = torch.max(mel, mel.max() - float(self.top_db))
            mel = (mel + float(self.top_db)) / float(self.top_db)
            
            # Ensure the mel spectrogram length is multiple of segment_size
            frames_per_seg = self.segment_size // self.hop_length
            n_frames = mel.size(1)
            if n_frames % frames_per_seg != 0:
                pad_len = frames_per_seg - (n_frames % frames_per_seg)
                mel = torch.nn.functional.pad(mel, (0, pad_len), mode='constant', value=0)
            
            return mel
            
        except Exception as e:
            logger.error(f"Error computing mel spectrogram: {e}")
            return torch.zeros(self.n_mels, 1, device=self.device)
    
    @torch.no_grad()
    def get_wav(self, mel: torch.Tensor, n_iters: int = 32) -> torch.Tensor:
        """Convert mel-spectrogram to waveform using improved Griffin-Lim."""
        try:
            # Move input to device
            mel = mel.to(self.device)
            
            # Convert from log scale with dynamic range expansion
            mel = torch.exp(mel * self.top_db) - self.clip_val
            
            # Convert from mel-scale to linear
            magnitudes = torch.matmul(self.mel_basis.T, mel)
            
            # Initialize phase with better random initialization
            angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.shape)))
            angles = torch.from_numpy(angles).float().to(self.device)
            
            # Improved Griffin-Lim with momentum
            momentum = 0.99
            prev_angles = None
            
            for i in range(n_iters):
                # Build complex spectrogram
                stft = magnitudes * torch.exp(1j * angles)
                
                # Inverse STFT
                wav = torch.istft(
                    stft,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=self.window,
                    center=True
                )
                
                # Recompute spectrogram
                stft = torch.stft(
                    wav,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=self.window,
                    center=True,
                    return_complex=True
                )
                
                # Update angles with momentum
                new_angles = torch.angle(stft)
                if prev_angles is not None:
                    angles = new_angles + momentum * (angles - prev_angles)
                else:
                    angles = new_angles
                prev_angles = new_angles
            
            # Normalize output
            wav = self.normalize_audio(wav)
            
            return wav
            
        except Exception as e:
            logger.error(f"Error converting mel to audio: {e}")
            return torch.zeros(1, device=self.device)
    
    def save_wav(self, wav: torch.Tensor, path: str, normalize: bool = True) -> bool:
        """Save audio file with proper format and normalization."""
        try:
            # Create output directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Normalize if requested
            if normalize:
                wav = self.normalize_audio(wav)
            
            # Convert to numpy and save
            wav = wav.cpu().numpy()
            librosa.output.write_wav(path, wav, self.sr)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving audio file {path}: {e}")
            return False