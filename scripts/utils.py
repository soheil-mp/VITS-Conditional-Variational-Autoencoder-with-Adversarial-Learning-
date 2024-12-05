import os
import glob
import torch
import numpy as np
from pathlib import Path

def save_checkpoint(
    model,
    discriminator,
    optimizer,
    optim_d,
    scaler,
    scaler_d,
    epoch,
    step,
    config,
    is_epoch_end=False,
    is_best=False,
    best_loss=float('inf')
):
    """Save model checkpoint with improved handling."""
    # Create checkpoint directory if it doesn't exist
    Path(config.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model': model.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optim_d': optim_d.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'scaler_d': scaler_d.state_dict() if scaler_d is not None else None,
        'epoch': epoch,
        'step': step,
        'config': config,
        'best_loss': best_loss
    }
    
    if is_best:
        path = os.path.join(config.train.checkpoint_dir, 'best_model.pt')
    elif is_epoch_end:
        path = os.path.join(config.train.checkpoint_dir, f'epoch_{epoch:04d}.pt')
    else:
        path = os.path.join(config.train.checkpoint_dir, f'step_{step:07d}.pt')
    
    # Save checkpoint atomically
    temp_path = path + '.tmp'
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)
    print(f"Saved checkpoint to {path}")
    
    # Keep only the last N checkpoints to save space
    if is_epoch_end:
        cleanup_old_checkpoints(config.train.checkpoint_dir, keep_last_n=5)

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=5, pattern='epoch_*.pt'):
    """Clean up old checkpoints keeping only the last N."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
    if len(checkpoints) <= keep_last_n:
        return
        
    checkpoints.sort()
    for checkpoint in checkpoints[:-keep_last_n]:
        try:
            os.remove(checkpoint)
        except OSError as e:
            print(f"Error removing checkpoint {checkpoint}: {e}")

def load_checkpoint(checkpoint_path, device, model=None, discriminator=None, 
                   optimizer=None, optim_d=None, scaler=None, scaler_d=None):
    """Load checkpoint with improved error handling."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if model is not None and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            
        if discriminator is not None and 'discriminator' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator'])
            
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            
        if optim_d is not None and 'optim_d' in checkpoint:
            optim_d.load_state_dict(checkpoint['optim_d'])
            
        if scaler is not None and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
            scaler.load_state_dict(checkpoint['scaler'])
            
        if scaler_d is not None and 'scaler_d' in checkpoint and checkpoint['scaler_d'] is not None:
            scaler_d.load_state_dict(checkpoint['scaler_d'])
        
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint from {checkpoint_path}: {e}")
        return None

def find_latest_checkpoint(checkpoint_dir, include_step=True):
    """Find the latest checkpoint with improved handling."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Check for best model first
    best_model = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model):
        return best_model
        
    # Look for epoch checkpoints
    epoch_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'epoch_*.pt'))
    if epoch_checkpoints:
        return max(epoch_checkpoints, key=os.path.getctime)
    
    # Look for step checkpoints if allowed
    if include_step:
        step_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'step_*.pt'))
        if step_checkpoints:
            return max(step_checkpoints, key=os.path.getctime)
    
    return None

class HParams:
    """Hyperparameters class with improved functionality."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v
    
    def keys(self):
        return self.__dict__.keys()
    
    def items(self):
        return self.__dict__.items()
    
    def values(self):
        return self.__dict__.values()
    
    def __len__(self):
        return len(self.__dict__)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        return setattr(self, key, value)
    
    def __contains__(self, key):
        return key in self.__dict__
    
    def __repr__(self):
        return self.__dict__.__repr__()

def normalize_audio(audio, target_level=-27):
    """Normalize audio to a target dB level."""
    rms = np.sqrt(np.mean(audio ** 2))
    scalar = (10 ** (target_level / 20)) / (rms + 1e-10)
    audio = audio * scalar
    return np.clip(audio, -1.0, 1.0)

def compute_mel_spectrogram(audio, config):
    """Compute mel spectrogram with librosa."""
    try:
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=config.data.sampling_rate,
            n_fft=config.data.filter_length,
            hop_length=config.data.hop_length,
            win_length=config.data.win_length,
            n_mels=config.data.n_mel_channels,
            fmin=config.data.mel_fmin,
            fmax=config.data.mel_fmax
        )
        mel_spec = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))
        return mel_spec
    except ImportError:
        print("librosa not found. Please install librosa for mel spectrogram computation.")
        return None

def setup_logger(log_dir):
    """Set up logging with improved formatting."""
    import logging
    from datetime import datetime
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up logging format
    log_format = '%(asctime)s %(levelname)s %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create file handler
    log_file = os.path.join(log_dir, f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Set up logger
    logger = logging.getLogger('VITS')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    