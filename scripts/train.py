import os
import yaml
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
import torch.cuda.amp as amp
from tqdm import tqdm, trange
import soundfile as sf
import numpy as np
from collections import defaultdict
import glob

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import commons
from src.utils import utils
from src.data.dataset import TextMelDataset, TextMelCollate
from src.models.vits import VITS, MultiPeriodDiscriminator
from src.text.text_processing import text_to_sequence

def generate_sample(model, text, config, device):
    """Generate a sample prediction."""
    model.eval()
    with torch.no_grad():
        # Convert text to sequence
        sequence = text_to_sequence(text, config.data.text_cleaners)
        sequence = torch.LongTensor(sequence).unsqueeze(0).to(device)
        sequence_length = torch.LongTensor([sequence.size(1)]).to(device)
        
        # Generate audio
        audio, _, _ = model.infer(sequence, sequence_length)
        audio = audio[0].cpu().float().numpy()
        
        # Normalize audio to [-1, 1] range
        audio = audio / max(abs(audio.min()), abs(audio.max()))
        
    model.train()
    return audio

def save_checkpoint(model, optimizer, discriminator, optimizer_d, scaler, scaler_d, epoch, step, config, losses=None, is_epoch_end=False):
    """Save model checkpoint with improved organization."""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_d': optimizer_d.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'scaler_d': scaler_d.state_dict() if scaler_d is not None else None,
        'epoch': epoch,
        'step': step,
        'config': config,
        'losses': losses
    }
    
    if is_epoch_end:
        # Save epoch checkpoint
        checkpoint_path = os.path.join(config.train.checkpoint_dir, f'epoch_{epoch+1:03d}.pt')
    else:
        # Save step checkpoint
        checkpoint_path = os.path.join(config.train.checkpoint_dir, f'step_{step:07d}.pt')
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def create_directories(config):
    """Create necessary directories for logging and checkpoints."""
    os.makedirs(config.train.log_dir, exist_ok=True)
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
    if not checkpoints:
        return None
        
    latest = max(checkpoints, key=os.path.getctime)
    return latest

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train VITS TTS model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    return parser.parse_args()

def dict_to_namespace(d):
    """Convert dictionary to namespace object."""
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create necessary directories
    os.makedirs(config['train']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['train']['log_dir'], exist_ok=True)
    
    # Convert config to namespace
    config = dict_to_namespace(config)
    
    # Set device and precision
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    precision = getattr(config.train, 'precision', 'fp32')
    
    # Enable anomaly detection in debug mode
    if args.verbose:
        torch.autograd.set_detect_anomaly(True)
    
    # Start training
    train(0, 1, config, args.verbose)

def train(rank, world_size, config, verbose=False):
    """Main training function."""
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    
    # Set up distributed training
    if world_size > 1:
        setup(rank, world_size)
    
    # Set device and precision
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    precision = getattr(config.train, 'precision', 'fp32')
    
    print("\n=== Training Configuration ===")
    print(f"Device: {device}")
    print(f"Precision: {precision}")
    print(f"Batch Size: {config.train.batch_size}")
    print(f"Learning Rate: {config.train.learning_rate}")
    print(f"Total Epochs: {config.train.epochs}")
    print("============================\n")
    
    # Create model and move to device
    model = VITS(config)
    model = model.to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Create discriminator
    discriminator = MultiPeriodDiscriminator().to(device)
    if world_size > 1:
        discriminator = DDP(discriminator, device_ids=[rank])

    # Create optimizers with gradient clipping
    optimizer = torch.optim.AdamW(
        model.parameters(),
        config.train.learning_rate,
        betas=config.train.betas,
        eps=config.train.eps,
        weight_decay=0.01)  # Add weight decay for regularization
        
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(),
        config.train.learning_rate,
        betas=config.train.betas,
        eps=config.train.eps,
        weight_decay=0.01)

    # Create gradient scaler for mixed precision training
    scaler = GradScaler() if precision == 'amp' else None
    scaler_d = GradScaler() if precision == 'amp' else None

    # Prepare dataset with pin memory and persistent workers
    train_dataset = TextMelDataset(config.data.training_files, config)
    train_dataset.set_training(True)  # Set training mode
    collate_fn = TextMelCollate()
    
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        train_sampler = None
        
    train_loader = DataLoader(
        train_dataset,
        num_workers=0,  # Disable multiprocessing for now
        shuffle=True if train_sampler is None else False,
        batch_size=config.train.batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn)

    # Create learning rate schedulers
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.train.learning_rate,
        epochs=config.train.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=1e4
    )
    
    scheduler_d = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_d,
        max_lr=config.train.learning_rate,
        epochs=config.train.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=1e4
    )

    # Load checkpoint if exists
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    
    checkpoint_path = find_latest_checkpoint(config.train.checkpoint_dir)
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        start_epoch, global_step, losses = load_checkpoint(checkpoint_path, model, optimizer, discriminator, optimizer_d, scaler, scaler_d, device)
        
        print(f"Resuming from epoch {start_epoch}, step {global_step}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    # Create tensorboard writer
    if rank == 0:
        writer = SummaryWriter(log_dir=config.train.log_dir)
    else:
        writer = None

    try:
        # Training loop
        for epoch in range(start_epoch, config.train.epochs):
            print(f"\nStarting training for {config.train.epochs} epochs")
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.train.epochs}")
            
            epoch_stats = defaultdict(float)
            
            for batch_idx, batch in enumerate(progress_bar):
                model.train()
                discriminator.train()
                
                # Move batch to device
                text_padded, input_lengths, mel_padded, output_lengths = [
                    x.to(device) for x in batch
                ]
                
                # Forward pass with mixed precision
                with amp.autocast():
                    # Generator forward pass
                    outputs = model(text_padded, input_lengths, mel_padded, output_lengths)
                    
                    # Discriminator forward pass
                    y_d_hat_r, y_d_hat_g, _, _ = discriminator(mel_padded, outputs['mel_output'])
                    
                    # Calculate losses
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    
                    loss_disc_all = loss_disc

                # Backward pass for discriminator
                optimizer_d.zero_grad()
                if scaler_d is not None:
                    scaler_d.scale(loss_disc_all).backward(retain_graph=True)
                    scaler_d.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    loss_disc_all.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                    optimizer_d.step()

                # Generator forward pass
                with amp.autocast():
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = discriminator(mel_padded, outputs['mel_output'])
                    
                    # Calculate generator losses
                    loss_mel = F.l1_loss(outputs['mel_output'], mel_padded) * 45.0
                    loss_kl = F.mse_loss(outputs['z'], outputs['z_p']) * 0.01
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    
                    loss_gen_all = loss_mel + loss_kl + loss_gen

                # Backward pass for generator
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss_gen_all).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_gen_all.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # Update learning rates
                scheduler.step()
                scheduler_d.step()

                # Update progress bar
                progress_bar.set_postfix({
                    'g_loss': f'{loss_gen_all.item():.4f}',
                    'd_loss': f'{loss_disc_all.item():.4f}',
                    'mel_loss': f'{loss_mel.item():.4f}'
                })

                # Update epoch stats
                epoch_stats['g_loss'] += loss_gen_all.item()
                epoch_stats['d_loss'] += loss_disc_all.item()
                epoch_stats['mel_loss'] += loss_mel.item()
                epoch_stats['kl_loss'] += loss_kl.item()
                epoch_stats['gen_loss'] += loss_gen.item()

                # Save checkpoint
                if global_step % config.train.checkpoint_interval == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        discriminator=discriminator,
                        optimizer_d=optimizer_d,
                        scaler=scaler,
                        scaler_d=scaler_d,
                        epoch=epoch,
                        step=global_step,
                        config=config,
                        losses={
                            'g_loss': loss_gen_all.item(),
                            'd_loss': loss_disc_all.item(),
                            'mel_loss': loss_mel.item(),
                            'kl_loss': loss_kl.item(),
                            'gen_loss': loss_gen.item()
                        }
                    )

                global_step += 1

            # Compute epoch averages
            for k in epoch_stats:
                epoch_stats[k] /= len(train_loader)

            # Save epoch checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                discriminator=discriminator,
                optimizer_d=optimizer_d,
                scaler=scaler,
                scaler_d=scaler_d,
                epoch=epoch,
                step=global_step,
                config=config,
                losses=epoch_stats,
                is_epoch_end=True
            )

            # Log epoch stats
            if rank == 0 and writer is not None:
                for k, v in epoch_stats.items():
                    writer.add_scalar(f'train/{k}', v, epoch)

    except Exception as e:
        print("\nTraining interrupted by error:", str(e))
        raise e

    finally:
        # Clean up
        if world_size > 1:
            cleanup()
        if writer is not None:
            writer.close()

def generate_samples(model, config, device, epoch, samples_dir):
    """Generate and save audio samples."""
    print(f"\nGenerating samples for epoch {epoch + 1}...")
    model.eval()
    
    sample_texts = [
        "Hello, this is a test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "I am learning to speak with artificial intelligence."
    ]
    
    with torch.no_grad():
        for i, text in enumerate(sample_texts):
            try:
                # Convert text to sequence
                sequence = text_to_sequence(text, config.data.text_cleaners)
                sequence = torch.LongTensor(sequence).unsqueeze(0).to(device)
                sequence_length = torch.LongTensor([sequence.size(1)]).to(device)
                
                # Generate audio
                audio, _, _ = model.infer(sequence, sequence_length)
                audio = audio[0].cpu().float().numpy()
                
                # Normalize audio to [-1, 1] range
                audio = audio / max(abs(audio.min()), abs(audio.max()))
                
                # Save audio
                sample_path = os.path.join(samples_dir, f'epoch_{epoch + 1:03d}_sample_{i + 1}.wav')
                sf.write(sample_path, audio, config.data.sampling_rate, 'PCM_16')
                print(f"Saved sample {i + 1} to {sample_path}")
            except Exception as e:
                print(f"Error generating sample {i + 1}: {str(e)}")
    
    model.train()

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Discriminator loss function."""
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    """Generator loss function."""
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

def feature_loss(fmap_r, fmap_g):
    """Feature matching loss."""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += F.l1_loss(rl, gl)
    return loss * 2

def load_checkpoint(checkpoint_path, model, optimizer, discriminator, optimizer_d, scaler, scaler_d, device):
    """Load checkpoint with improved organization."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model and optimizer states
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d'])
    
    # Load scaler states if they exist
    if checkpoint['scaler'] is not None and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    if checkpoint['scaler_d'] is not None and scaler_d is not None:
        scaler_d.load_state_dict(checkpoint['scaler_d'])
    
    return checkpoint['epoch'], checkpoint['step'], checkpoint.get('losses', None)

if __name__ == "__main__":
    main() 