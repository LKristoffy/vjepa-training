.PHONY: help setup install download-data train clean lint format check-env install-system-deps fix-torchcodec update-transformers

# Default target
help:
	@echo "Available targets:"
	@echo "  setup             - Set up the development environment (install uv if needed)"
	@echo "  install-system-deps - Install system dependencies (FFmpeg via Homebrew)"
	@echo "  install           - Install project dependencies using uv"
	@echo "  fix-torchcodec    - Fix torchcodec FFmpeg linking issues (run if torchcodec fails)"
	@echo "  update-transformers - Update transformers to dev version (supports VJEPA2)"
	@echo "  download-data     - Download the UCF101 subset dataset"
	@echo "  train             - Run the training script"
	@echo "  clean             - Clean up generated files and cache"
	@echo "  lint              - Run linting checks"
	@echo "  format            - Format code with ruff"
	@echo "  check-env         - Check if required tools are installed"
	@echo "  check-system-deps - Check if system dependencies are installed"
	@echo "  all               - Run complete setup and training pipeline"

# Check if required tools are installed
check-env:
	@echo "Checking environment..."
	@command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting." >&2; exit 1; }
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Run 'make setup' first." >&2; exit 1; }
	@echo "Environment check passed!"

# Check if system dependencies are installed
check-system-deps:
	@echo "Checking system dependencies..."
	@command -v brew >/dev/null 2>&1 || { echo "Homebrew is required but not installed. Please install from https://brew.sh" >&2; exit 1; }
	@command -v ffmpeg >/dev/null 2>&1 || { echo "FFmpeg is not installed. Run 'make install-system-deps' first." >&2; exit 1; }
	@echo "System dependencies check passed!"

# Install system dependencies (FFmpeg via Homebrew)
install-system-deps:
	@echo "Installing system dependencies..."
	@if ! command -v brew >/dev/null 2>&1; then \
		echo "Homebrew is required to install FFmpeg. Please install from https://brew.sh"; \
		exit 1; \
	fi
	@if ! command -v ffmpeg >/dev/null 2>&1; then \
		echo "Installing FFmpeg..."; \
		brew install ffmpeg; \
	else \
		echo "FFmpeg is already installed"; \
	fi
	@echo "System dependencies installed successfully!"

# Set up the development environment
setup:
	@echo "Setting up development environment..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "Please restart your shell or run: source ~/.bashrc"; \
	else \
		echo "uv is already installed"; \
	fi

# Install project dependencies
install: check-env
	@echo "Installing project dependencies..."
	uv sync
	@echo "Dependencies installed successfully!"

# Fix torchcodec FFmpeg linking issues
fix-torchcodec: check-env check-system-deps
	@echo "Fixing torchcodec FFmpeg linking..."
	@echo "Reinstalling torchcodec to link with current FFmpeg..."
	uv pip uninstall torchcodec || true
	uv pip install torchcodec --no-cache-dir
	@echo "torchcodec reinstalled successfully!"

# Update transformers to development version (supports VJEPA2)
update-transformers: check-env
	@echo "Updating transformers to development version..."
	@echo "This may take a few minutes as it installs from source..."
	uv sync --upgrade
	@echo "Transformers updated successfully!"

# Download the UCF101 subset dataset
download-data: check-env
	@echo "Downloading UCF101 subset dataset..."
	@if [ ! -d "UCF101_subset" ]; then \
		uv run python src/download_data.py; \
	else \
		echo "Dataset already exists. Skipping download."; \
	fi

# Run the training script
train: check-env check-system-deps
	@echo "Starting training..."
	@if [ ! -d "UCF101_subset" ]; then \
		echo "Dataset not found. Downloading first..."; \
		$(MAKE) download-data; \
	fi
	@echo "Setting FFmpeg library paths for torchcodec..."
	@export DYLD_LIBRARY_PATH="$$(brew --prefix ffmpeg)/lib:$$DYLD_LIBRARY_PATH" && \
	export DYLD_FALLBACK_LIBRARY_PATH="$$(brew --prefix ffmpeg)/lib:$$DYLD_FALLBACK_LIBRARY_PATH" && \
	uv run python src/train.py

# Clean up generated files
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "Cleanup complete!"

# Run linting checks
lint: check-env
	@echo "Running linting checks..."
	uv run ruff check src/
	@echo "Linting complete!"

# Format code
format: check-env
	@echo "Formatting code..."
	uv run ruff format src/
	@echo "Code formatting complete!"

# Run everything in sequence
all: setup install-system-deps install download-data train

# Development targets
dev-install: install
	@echo "Installing development dependencies..."
	uv add --dev ruff pytest
	@echo "Development setup complete!"

# Quick start for new users
quickstart:
	@echo "Quick start setup..."
	$(MAKE) setup
	@echo "Please restart your shell and run 'make install' to continue"
