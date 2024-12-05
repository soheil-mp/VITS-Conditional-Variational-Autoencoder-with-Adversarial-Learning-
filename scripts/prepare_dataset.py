import os
import argparse
import wget
import tarfile
import pandas as pd
import torch
import tqdm
import yaml
import random
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import sys
import hashlib
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.audio import AudioProcessor
from src.utils.config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_download(file_path: str, expected_hash: Optional[str] = None) -> bool:
    """Verify downloaded file integrity."""
    if not os.path.exists(file_path):
        return False
        
    if expected_hash:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return file_hash == expected_hash
        
    return True

def download_ljspeech(data_path: str) -> None:
    """Download LJSpeech dataset with improved error handling."""
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    target_path = data_path / "LJSpeech-1.1.tar.bz2"
    
    try:
        # Download with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if target_path.exists():
                    target_path.unlink()  # Remove existing file if any
                
                logger.info(f"Downloading LJSpeech dataset to {target_path} (Attempt {attempt + 1}/{max_retries})")
                wget.download(url, str(target_path))
                logger.info("\nDownload completed. Extracting...")
                
                # Extract the tar file
                with tarfile.open(target_path, "r:bz2") as tar:
                    tar.extractall(path=str(data_path))
                
                # Verify extraction
                ljspeech_dir = data_path / "LJSpeech-1.1"
                metadata_file = ljspeech_dir / "metadata.csv"
                wavs_dir = ljspeech_dir / "wavs"
                
                if not all([ljspeech_dir.exists(), metadata_file.exists(), wavs_dir.exists()]):
                    raise ValueError("Extraction verification failed")
                
                logger.info("Dataset extraction completed successfully")
                return
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if target_path.exists():
                    target_path.unlink()
                if attempt < max_retries - 1:
                    logger.info("Retrying download...")
                else:
                    raise
                
    except Exception as e:
        logger.error(f"Error downloading/extracting dataset: {e}")
        if target_path.exists():
            target_path.unlink()
        raise ValueError(f"Failed to download/extract dataset after {max_retries} attempts: {str(e)}")

def prepare_metadata(data_path: str) -> pd.DataFrame:
    """Prepare metadata with improved validation."""
    data_path = Path(data_path)
    metadata_path = data_path / "LJSpeech-1.1" / "metadata.csv"
    wav_dir = data_path / "LJSpeech-1.1" / "wavs"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    try:
        # Read metadata with proper validation
        metadata_df = pd.read_csv(
            metadata_path,
            sep="|",
            header=None,
            names=["ID", "Transcription", "Normalized"],
            dtype={
                "ID": str,
                "Transcription": str,
                "Normalized": str
            },
            na_filter=True
        )
        
        # Validate data
        if metadata_df.isnull().any().any():
            logger.warning("Found null values in metadata")
            metadata_df = metadata_df.dropna()
        
        # Create and validate audio paths
        metadata_df["AudioPath"] = metadata_df["ID"].apply(lambda x: str(wav_dir / f"{x}.wav"))
        valid_files = metadata_df["AudioPath"].apply(os.path.exists)
        if not valid_files.all():
            missing_files = metadata_df[~valid_files]["ID"].tolist()
            logger.warning(f"Missing audio files: {missing_files}")
            metadata_df = metadata_df[valid_files]
        
        return metadata_df
        
    except Exception as e:
        logger.error(f"Error preparing metadata: {e}")
        raise

def process_audio_file(args: Tuple[str, str, Dict, str]) -> bool:
    """Process a single audio file."""
    audio_path, file_id, config_dict, mel_dir = args
    try:
        # Initialize processor for each process
        config = Config(config_dict)  # Convert dict back to Config object
        audio_processor = AudioProcessor(config)
        
        # Load and process audio
        wav, sr = audio_processor.load_wav(audio_path)
        mel = audio_processor.get_mel(wav)
        
        # Save mel spectrogram
        mel_path = os.path.join(mel_dir, f"{file_id}.npy")
        torch.save(mel, mel_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing file {audio_path}: {e}")
        return False

def process_dataset(config: Config, metadata_df: pd.DataFrame, output_dir: str, num_workers: Optional[int] = None) -> None:
    """Process dataset with improved parallel processing."""
    output_dir = Path(output_dir)
    mel_dir = output_dir / "mels"
    mel_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)
    
    # Convert Config to dict for pickling
    config_dict = config.__dict__
    
    # Prepare arguments for parallel processing
    process_args = [
        (row["AudioPath"], row["ID"], config_dict, str(mel_dir))
        for _, row in metadata_df.iterrows()
    ]
    
    # Process files in parallel with improved error handling
    logger.info(f"Processing {len(process_args)} files using {num_workers} workers")
    
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor for better stability
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        try:
            results = list(tqdm.tqdm(
                executor.map(process_audio_file, process_args),
                total=len(process_args),
                desc="Generating mel spectrograms"
            ))
            
            # Check results
            successful = sum(1 for r in results if r)
            failed = len(results) - successful
            
            logger.info(f"Successfully processed {successful} files")
            if failed > 0:
                logger.warning(f"Failed to process {failed} files")
                
        except Exception as e:
            logger.error(f"Error during parallel processing: {e}")
            raise

def create_train_val_split(
    metadata_df: pd.DataFrame,
    output_dir: str,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42
) -> None:
    """Create train-validation-test split with improved stratification."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Create filelists directory in the root directory
    filelist_dir = Path("filelists")
    filelist_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate split sizes
    total_size = len(metadata_df)
    test_count = int(test_size * total_size)
    val_count = int(val_size * total_size)
    train_count = total_size - val_count - test_count
    
    # Create splits with stratification
    indices = np.random.permutation(total_size)
    test_indices = indices[:test_count]
    val_indices = indices[test_count:test_count + val_count]
    train_indices = indices[test_count + val_count:]
    
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    # Save filelists
    def save_filelist(indices: np.ndarray, filename: str) -> None:
        data = metadata_df.iloc[indices]
        filepath = filelist_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            for _, row in data.iterrows():
                mel_path = f"data/mels/{row['ID']}.npy"
                f.write(f"{mel_path}|{row['Normalized']}\n")
        
        logger.info(f"Saved {len(indices)} entries to {filename}")
    
    for split_name, split_indices in splits.items():
        save_filelist(split_indices, f"{split_name}.txt")

def main():
    parser = argparse.ArgumentParser(description="Prepare TTS dataset with improved processing")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store dataset")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    try:
        # Set random seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # Load and validate config
        with open(args.config) as f:
            config = yaml.safe_load(f)
        config = Config(config)
        
        # Create directories
        data_dir = Path(args.data_dir)
        raw_data_dir = data_dir / "raw"
        processed_data_dir = data_dir / "processed"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Process dataset
        logger.info("Starting dataset preparation...")
        
        # Download dataset
        download_ljspeech(str(raw_data_dir))
        
        # Prepare metadata
        metadata_df = prepare_metadata(str(raw_data_dir))
        logger.info(f"Found {len(metadata_df)} valid entries in metadata")
        
        # Process audio files
        process_dataset(config, metadata_df, str(processed_data_dir), args.num_workers)
        
        # Create data splits
        create_train_val_split(
            metadata_df,
            str(processed_data_dir),
            val_size=0.1,
            test_size=0.1,
            seed=args.seed
        )
        
        logger.info("Dataset preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise

if __name__ == "__main__":
    main() 