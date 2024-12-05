# 🎙️ VITS: State-of-the-Art Text-to-Speech Implementation

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2106.06103-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2106.06103)

*A PyTorch implementation of VITS: Conditional Variational Autoencoder with Adversarial Learning*

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Training](#training)

</div>

## 🌟 Overview

This project implements VITS (Conditional Variational Autoencoder with Adversarial Learning), a state-of-the-art end-to-end Text-to-Speech model that directly generates waveforms from text. Key features include:

- End-to-end text-to-speech synthesis
- Parallel sampling for ultra-fast inference
- High-quality audio generation
- Multi-speaker support
- Emotion and style control

## 📊 Requirements

- Python 3.8+
- CUDA-compatible GPU (8GB+ VRAM)
- 16GB+ RAM
- 50GB+ disk space

## 🚀 Installation

1. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   
   # Linux/Mac
   source venv/bin/activate
   
   # Windows
   .\venv\Scripts\activate
   ```

2. **Install PyTorch**:
   ```bash
   # Windows/Linux with CUDA 11.8
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # CPU only
   pip install torch torchaudio
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```python
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   ```

## 📊 Dataset Preparation

### Linux/macOS
```bash
mkdir -p data/raw/LJSpeech-1.1
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -P data/raw
tar -xvf data/raw/LJSpeech-1.1.tar.bz2 -C data/raw
rm data/raw/LJSpeech-1.1.tar.bz2
```

### Windows (PowerShell)
```powershell
New-Item -ItemType Directory -Force -Path "data\raw\LJSpeech-1.1"
Invoke-WebRequest -Uri "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2" -OutFile "data\raw\LJSpeech-1.1.tar.bz2"
& 'C:\Program Files\7-Zip\7z.exe' x "data\raw\LJSpeech-1.1.tar.bz2" -o"data\raw"
& 'C:\Program Files\7-Zip\7z.exe' x "data\raw\LJSpeech-1.1.tar" -o"data\raw"
Remove-Item "data\raw\LJSpeech-1.1.tar*"
```

### Python (Cross-platform)
```python
import os, requests, tarfile
from pathlib import Path

data_dir = Path("data/raw/LJSpeech-1.1")
data_dir.mkdir(parents=True, exist_ok=True)

url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
archive_path = data_dir.parent / "LJSpeech-1.1.tar.bz2"

print("Downloading LJSpeech dataset...")
response = requests.get(url, stream=True)
with open(archive_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("Extracting dataset...")
with tarfile.open(archive_path, 'r:bz2') as tar:
    tar.extractall(path=data_dir.parent)
archive_path.unlink()
```

## 🎯 Training

1. **Prepare dataset**:
   ```bash
   python scripts/prepare_dataset.py --config configs/vits_config.yaml
   ```

2. **Start training**:
   ```bash
   # Single GPU
   python scripts/train.py --config configs/vits_config.yaml

   # Multi-GPU (e.g., 4 GPUs)
   python scripts/train.py --config configs/vits_config.yaml --world_size 4
   ```

3. **Monitor training**:
   ```bash
   # TensorBoard
   tensorboard --logdir data/logs

   # Weights & Biases monitoring is automatic if enabled in config
   ```

## 🎵 Inference

```python
from src.inference import VITS

# Initialize model
vits = VITS(checkpoint="path/to/checkpoint")

# Basic synthesis
audio = vits.synthesize(
    text="Hello, world!",
    speaker_id=0,
    speed_factor=1.0
)

# Save audio
vits.save_audio(audio, "output.wav")

# Batch processing
texts = [
    "First sentence.",
    "Second sentence.",
    "Third sentence."
]
audios = vits.synthesize_batch(texts, speaker_id=0)
```

## 🧠 Model Architecture

```
Text → [Text Encoder] → Hidden States
                              ↓
                    [Posterior Encoder]
                              ↓
                    [Flow Decoder] → Audio
                              ↓
              [Multi-Period Discriminator]
              [Multi-Scale Discriminator]
```

Key components:
1. **Text Encoder**: Transformer-based with multi-head attention
2. **Flow Decoder**: Normalizing flows with residual coupling
3. **Posterior Encoder**: WaveNet-style architecture
4. **Discriminators**: Multi-period and multi-scale for quality
5. **Voice Conversion**: Optional cross-speaker style transfer

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   ```bash
   # Reduce batch size in config
   # Enable gradient accumulation
   # Use mixed precision (fp16)
   ```

2. **Poor Audio Quality**:
   - Check preprocessing parameters
   - Verify loss convergence
   - Ensure proper normalization

3. **Slow Training**:
   - Enable mixed precision
   - Use DDP for multi-GPU
   - Optimize dataloader workers

## 📚 Citation

```bibtex
@inproceedings{kim2021vits,
  title={Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech},
  author={Kim, Jaehyeon and Kong, Jungil and Son, Juhee},
  booktitle={International Conference on Machine Learning},
  year={2021}
}
```

## 📜 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- [Official VITS Implementation](https://github.com/jaywalnut310/vits)
- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- [PyTorch](https://pytorch.org/)

<div align="center">
Made with ❤️ by the TTS Team

[Report Bug](https://github.com/yourusername/tts/issues) · [Request Feature](https://github.com/yourusername/tts/issues)
</div> 