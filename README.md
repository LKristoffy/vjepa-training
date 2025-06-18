# VJEPA Training Project

A PyTorch-based training pipeline for fine-tuning VJEPA2 (Video Joint Embedding Predictive Architecture) models on video classification tasks. This project is optimized for Apple Silicon Macs with MPS (Metal Performance Shaders) acceleration.

## Features

- 🚀 **MPS Device Support** - Optimized for Apple Silicon GPU acceleration
- 🎥 **Video Processing** - Uses torchcodec for efficient video decoding
- 🤖 **VJEPA2 Model** - Latest video understanding architecture from Meta
- 📊 **Weights & Biases Integration** - Automatic experiment tracking
- 🛠️ **Automated Setup** - One-command installation and training

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.12+
- Homebrew (for FFmpeg installation)

### Installation & Training

```bash
# Complete setup and training in one command
make all

# Or step by step:
make setup              # Install uv package manager
make install-system-deps # Install FFmpeg via Homebrew
make install            # Install Python dependencies
make update-transformers # Install transformers dev version (VJEPA2 support)
make download-data      # Download UCF101 subset dataset
make train              # Start training
```

## Project Structure

```
vjepa-training/
├── src/
│   ├── train.py           # Main training script with MPS support
│   └── download_data.py   # Dataset download utility
├── Makefile              # Automated build and run commands
├── pyproject.toml        # Project dependencies
└── README.md            # This file
```

## Available Commands

Run `make help` to see all available commands:

| Command | Description |
|---------|-------------|
| `make setup` | Set up development environment (install uv) |
| `make install-system-deps` | Install system dependencies (FFmpeg) |
| `make install` | Install Python dependencies |
| `make fix-torchcodec` | Fix torchcodec FFmpeg linking issues |
| `make update-transformers` | Install transformers dev version (VJEPA2 support) |
| `make download-data` | Download UCF101 subset dataset |
| `make train` | Run training script with MPS acceleration |
| `make clean` | Clean up generated files and cache |
| `make lint` | Run code linting |
| `make format` | Format code with ruff |
| `make all` | Complete setup and training pipeline |

## Model Details

- **Base Model**: `qubvel-hf/vjepa2-vitl-fpc16-256-ssv2`
- **Dataset**: UCF101 subset (automatically downloaded)
- **Architecture**: Vision Transformer Large with 16 frames per clip
- **Training Strategy**: Fine-tuning with frozen encoder, trainable classifier head

## Training Configuration

The training script includes several optimizations:

- **Device Detection**: Automatically uses MPS > CUDA > CPU
- **Gradient Accumulation**: 4 steps for effective larger batch sizes
- **Data Augmentation**: Random crops and horizontal flips
- **Mixed Precision**: FP32 for stability on MPS
- **Evaluation**: Validation accuracy computed each epoch

## Troubleshooting

### Common Issues

**1. torchcodec FFmpeg Error**
```bash
make fix-torchcodec  # Reinstall torchcodec with proper FFmpeg linking
```

**2. VJEPA2 Import Error**
```bash
make update-transformers  # Install latest transformers with VJEPA2 support
```

**3. MPS Not Available**
- Ensure you're on Apple Silicon Mac
- Update to latest macOS and PyTorch versions

### System Requirements

- **Memory**: 16GB+ RAM recommended for video processing
- **Storage**: 5GB+ free space for dataset and model weights
- **Network**: Stable internet for model and dataset downloads

## Development

### Code Quality

```bash
make lint    # Check code style
make format  # Auto-format code
make clean   # Clean up generated files
```

### Adding Dependencies

Edit `pyproject.toml` and run:
```bash
make install  # Sync new dependencies
```

## Dataset

The project uses the UCF101 subset dataset:
- **Source**: Hugging Face (`sayakpaul/ucf101-subset`)
- **Size**: ~2GB compressed
- **Classes**: Multiple action recognition categories
- **Splits**: Pre-divided train/validation/test sets

## Model Performance

The training script tracks:
- Training loss (with gradient accumulation)
- Validation accuracy per epoch
- Final test accuracy
- All metrics logged to Weights & Biases

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test with `make lint` and `make format`
4. Submit a pull request

## License

This project is open source. Please check individual model and dataset licenses:
- VJEPA2 model: Meta's license terms
- UCF101 dataset: Academic use license
- PyTorch and dependencies: Various open source licenses

## Acknowledgments

- Meta AI for the VJEPA2 architecture
- Hugging Face for model hosting and transformers library
- UCF for the action recognition dataset
- PyTorch team for MPS support

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Run `make help` for available commands
3. Ensure all system dependencies are installed
4. Verify you're using Apple Silicon Mac for MPS support

---

**Happy Training! 🚀**
