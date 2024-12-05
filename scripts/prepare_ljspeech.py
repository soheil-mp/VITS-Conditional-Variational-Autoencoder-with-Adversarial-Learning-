import os
import wget
import zipfile
import argparse
import json
import yaml
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.audio import AudioProcessor
from src.utils.utils import HParams

def download_ljspeech():
    """Download and extract LJSpeech dataset."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download LJSpeech
    if not (data_dir / "LJSpeech-1.1").exists():
        print("Downloading LJSpeech-1.1...")
        wget.download(
            "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
            str(data_dir / "LJSpeech-1.1.tar.bz2")
        )
        
        # Extract
        print("\nExtracting LJSpeech-1.1...")
        import tarfile
        with tarfile.open(data_dir / "LJSpeech-1.1.tar.bz2", "r:bz2") as tar:
            tar.extractall(path=data_dir)
        
        # Remove archive
        (data_dir / "LJSpeech-1.1.tar.bz2").unlink()

def prepare_mels(config):
    """Prepare mel-spectrograms from audio files."""
    data_dir = Path("data")
    audio_dir = data_dir / "LJSpeech-1.1" / "wavs"
    mel_dir = data_dir / "mels"
    mel_dir.mkdir(exist_ok=True)
    
    # Create audio processor
    audio_processor = AudioProcessor(config)
    
    # Process each audio file
    audio_files = list(audio_dir.glob("*.wav"))
    print(f"Processing {len(audio_files)} audio files...")
    
    for audio_file in tqdm(audio_files):
        # Load audio
        wav, sr = audio_processor.load_wav(str(audio_file))
        
        # Convert to mel-spectrogram
        mel = audio_processor.get_mel(wav)
        
        # Save mel-spectrogram
        mel_path = mel_dir / f"{audio_file.stem}.npy"
        np.save(str(mel_path), mel.numpy())

def create_filelists():
    """Create train/validation filelists."""
    data_dir = Path("data")
    metadata_path = data_dir / "LJSpeech-1.1" / "metadata.csv"
    
    # Read metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = [line.strip().split("|") for line in f]
    
    # Split into train/validation (98%/2% split)
    np.random.seed(1234)
    np.random.shuffle(metadata)
    split = int(len(metadata) * 0.98)
    train_metadata = metadata[:split]
    val_metadata = metadata[split:]
    
    # Create filelists
    filelists_dir = Path("filelists")
    filelists_dir.mkdir(exist_ok=True)
    
    def write_filelist(filename, metadata):
        with open(filelists_dir / filename, "w", encoding="utf-8") as f:
            for item in metadata:
                mel_path = f"data/mels/{item[0]}.npy"
                text = item[2]  # Using normalized text
                f.write(f"{mel_path}|{text}\n")
    
    write_filelist("train_filelist.txt", train_metadata)
    write_filelist("val_filelist.txt", val_metadata)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                      help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = HParams(**config)
    
    # Download dataset
    download_ljspeech()
    
    # Prepare mel-spectrograms
    prepare_mels(config)
    
    # Create filelists
    create_filelists()
    
    print("Dataset preparation completed!")

if __name__ == "__main__":
    main() 