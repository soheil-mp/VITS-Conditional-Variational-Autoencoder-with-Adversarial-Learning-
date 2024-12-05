import os
import torch
import argparse
import yaml
import numpy as np
import soundfile as sf
from src.models.vits.model import VITS
from src.text.text_processing import text_to_sequence
from src.utils.config import Config

def load_model(checkpoint_path, config):
    """Load VITS model from checkpoint."""
    model = VITS(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model.cuda()

def synthesize(model, text, config):
    """Synthesize speech from text."""
    # Convert text to sequence
    sequence = text_to_sequence(text, config.data.text_cleaners)
    sequence = torch.LongTensor(sequence).unsqueeze(0).cuda()
    
    # Get sequence length
    sequence_length = torch.LongTensor([sequence.size(1)]).cuda()
    
    # Inference
    with torch.no_grad():
        audio, _, _ = model.infer(sequence, sequence_length)
    
    return audio[0].cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--text', type=str, required=True, help='Text to synthesize')
    parser.add_argument('--output', type=str, default='output.wav', help='Output audio file path')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = Config(config)

    # Load model
    model = load_model(args.checkpoint, config)
    print(f"Loaded model from {args.checkpoint}")

    # Synthesize speech
    print(f"Synthesizing: {args.text}")
    audio = synthesize(model, args.text, config)

    # Save audio
    sf.write(args.output, audio, config.data.sampling_rate)
    print(f"Saved audio to {args.output}")

if __name__ == "__main__":
    main() 