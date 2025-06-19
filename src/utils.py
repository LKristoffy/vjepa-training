import pathlib
from typing import Tuple, List, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Try to import torchcodec with fallback handling
try:
    from torchcodec.decoders import VideoDecoder
    from torchcodec.samplers import clips_at_random_indices
    TORCHCODEC_AVAILABLE = True
    print("✅ torchcodec imported successfully")
except ImportError as e:
    print(f"⚠️  torchcodec import failed: {e}")
    print("📝 Video processing functions will use fallback methods")
    TORCHCODEC_AVAILABLE = False
    VideoDecoder = None

from torchvision.transforms import v2

# Try to import transformers with fallback handling
try:
    from transformers import VJEPA2VideoProcessor, VJEPA2ForVideoClassification
    TRANSFORMERS_AVAILABLE = True
    print("✅ transformers with VJEPA2 imported successfully")
except ImportError as e:
    print(f"⚠️  VJEPA2 transformers import failed: {e}")
    print("📝 Model functions will require proper transformers installation")
    TRANSFORMERS_AVAILABLE = False
    VJEPA2VideoProcessor = None
    VJEPA2ForVideoClassification = None


def get_best_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_trained_model(model_path: str, device: Optional[torch.device] = None) -> Tuple[VJEPA2ForVideoClassification, VJEPA2VideoProcessor]:
    """
    Load a trained VJEPA2 model and processor from a saved checkpoint.
    
    Args:
        model_path: Path to the saved model directory
        device: Device to load the model on. If None, uses best available device.
    
    Returns:
        Tuple of (model, processor)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("VJEPA2 transformers not available. Please run 'make update-transformers' to install the development version.")
    
    if device is None:
        device = get_best_device()
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # Load processor and model
    processor = VJEPA2VideoProcessor.from_pretrained(model_path)
    model = VJEPA2ForVideoClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device
    )
    
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
    
    return model, processor


def print_model_info(model: VJEPA2ForVideoClassification):
    """
    Print detailed information about the model architecture and parameters.
    
    Args:
        model: The VJEPA2 model to analyze
    """
    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    
    # Model type and configuration
    print(f"Model Type: {type(model).__name__}")
    print(f"Model Config: {model.config.model_type}")
    print(f"Architecture: {model.config.architectures}")
    
    # Model dimensions
    config = model.config
    print(f"\nModel Dimensions:")
    print(f"  - Hidden Size: {config.hidden_size}")
    print(f"  - Number of Attention Heads: {config.num_attention_heads}")
    print(f"  - Number of Hidden Layers: {config.num_hidden_layers}")
    if hasattr(config, 'intermediate_size'):
        print(f"  - Intermediate Size: {config.intermediate_size}")
    if hasattr(config, 'mlp_dim'):
        print(f"  - MLP Dimension: {config.mlp_dim}")
    
    # Video processing parameters
    print(f"\nVideo Processing:")
    print(f"  - Image Size: {config.image_size}")
    print(f"  - Patch Size: {config.patch_size}")
    print(f"  - Frames per Clip: {config.frames_per_clip}")
    print(f"  - Tubelet Size: {config.tubelet_size}")
    
    # Classification head
    print(f"\nClassification:")
    print(f"  - Number of Labels: {config.num_labels}")
    if hasattr(config, 'id2label') and config.id2label:
        print(f"  - Label Classes: {list(config.id2label.values())[:5]}..." if len(config.id2label) > 5 else f"  - Label Classes: {list(config.id2label.values())}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameters:")
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")
    print(f"  - Frozen Parameters: {total_params - trainable_params:,}")
    
    print("=" * 60)


def visualize_model_architecture(model: VJEPA2ForVideoClassification):
    """
    Create a visual representation of the model architecture.
    
    Args:
        model: The VJEPA2 model to visualize
    """
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    
    # Print the model structure
    print(model)
    
    # Create a simple architecture diagram
    config = model.config
    
    print(f"\nArchitecture Flow:")
    print(f"Input Video ({config.frames_per_clip} frames, {config.image_size}x{config.image_size})")
    print("    ↓")
    print(f"Patch Embedding ({config.patch_size}x{config.patch_size} patches)")
    print("    ↓")
    print(f"Transformer Encoder ({config.num_hidden_layers} layers)")
    print("    ↓")
    print(f"Classification Head ({config.hidden_size} → {config.num_labels})")
    print("    ↓")
    print("Output Predictions")


def preprocess_video_opencv(video_path: str, frames_per_clip: int = 16, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Fallback video preprocessing using OpenCV when torchcodec is not available.
    
    Args:
        video_path: Path to the video file
        frames_per_clip: Number of frames to sample
        target_size: Target size for frames (height, width)
    
    Returns:
        Preprocessed video tensor
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frame indices
    if total_frames < frames_per_clip:
        frame_indices = list(range(total_frames))
        # Pad with repeated frames if needed
        while len(frame_indices) < frames_per_clip:
            frame_indices.extend(frame_indices[:frames_per_clip - len(frame_indices)])
    else:
        frame_indices = np.linspace(0, total_frames - 1, frames_per_clip, dtype=int)
    
    # Extract frames
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame
            frame = cv2.resize(frame, target_size)
            # Convert to tensor and normalize
            frame = torch.from_numpy(frame).float() / 255.0
            # Change from HWC to CHW
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
    
    cap.release()
    
    # Stack frames into video tensor
    video_tensor = torch.stack(frames, dim=0)  # Shape: (T, C, H, W)
    video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension: (1, T, C, H, W)
    
    return video_tensor


def preprocess_video(video_path: str, processor: VJEPA2VideoProcessor, frames_per_clip: int = 16) -> torch.Tensor:
    """
    Preprocess a video file for model inference.
    
    Args:
        video_path: Path to the video file
        processor: VJEPA2 video processor
        frames_per_clip: Number of frames to sample from the video
    
    Returns:
        Preprocessed video tensor
    """
    if not TORCHCODEC_AVAILABLE:
        print("⚠️  Using OpenCV fallback for video processing")
        target_size = (processor.crop_size["height"], processor.crop_size["width"])
        return preprocess_video_opencv(video_path, frames_per_clip, target_size)
    
    # Load video using torchcodec
    decoder = VideoDecoder(video_path)
    
    # Sample frames
    clip = clips_at_random_indices(
        decoder,
        num_clips=1,
        num_frames_per_clip=frames_per_clip,
        num_indices_between_frames=3,
    ).data
    
    # Apply center crop transform
    transform = v2.CenterCrop((processor.crop_size["height"], processor.crop_size["width"]))
    video_tensor = transform(clip)
    
    return video_tensor


def run_inference(
    model: VJEPA2ForVideoClassification, 
    processor: VJEPA2VideoProcessor, 
    video_path: str,
    device: Optional[torch.device] = None,
    top_k: int = 5
) -> Tuple[List[str], List[float], torch.Tensor]:
    """
    Run inference on a video file and return predictions.
    
    Args:
        model: Trained VJEPA2 model
        processor: VJEPA2 video processor
        video_path: Path to the video file
        device: Device to run inference on
        top_k: Number of top predictions to return
    
    Returns:
        Tuple of (predicted_labels, confidence_scores, video_tensor)
    """
    if device is None:
        device = get_best_device()
    
    print(f"Running inference on: {video_path}")
    
    # Preprocess video
    video_tensor = preprocess_video(video_path, processor, model.config.frames_per_clip)
    
    # Prepare input
    inputs = processor(video_tensor, return_tensors="pt").to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=-1)
    
    # Convert to lists
    predicted_labels = []
    confidence_scores = []
    
    for i in range(top_k):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        
        if hasattr(model.config, 'id2label') and model.config.id2label:
            label = model.config.id2label[idx]
        else:
            label = f"Class_{idx}"
        
        predicted_labels.append(label)
        confidence_scores.append(prob)
    
    return predicted_labels, confidence_scores, video_tensor


def display_video_frames(video_tensor: torch.Tensor, num_frames: int = 8, figsize: Tuple[int, int] = (15, 8)):
    """
    Display frames from a video tensor.
    
    Args:
        video_tensor: Video tensor of shape (1, frames, channels, height, width)
        num_frames: Number of frames to display
        figsize: Figure size for the plot
    """
    # Remove batch dimension and convert to numpy
    if video_tensor.dim() == 5:
        video_tensor = video_tensor.squeeze(0)  # Remove batch dimension
    
    frames = video_tensor.cpu().numpy()
    total_frames = frames.shape[0]
    
    # Select frames to display
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(2, num_frames // 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, frame_idx in enumerate(frame_indices):
        frame = frames[frame_idx]
        
        # Convert from (C, H, W) to (H, W, C) and normalize
        if frame.shape[0] == 3:  # RGB
            frame = np.transpose(frame, (1, 2, 0))
        
        # Normalize to [0, 1] if needed
        if frame.max() > 1.0:
            frame = frame / 255.0
        
        # Ensure values are in [0, 1]
        frame = np.clip(frame, 0, 1)
        
        axes[i].imshow(frame)
        axes[i].set_title(f'Frame {frame_idx + 1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def display_predictions(predicted_labels: List[str], confidence_scores: List[float], title: str = "Top Predictions"):
    """
    Display prediction results as a horizontal bar chart.
    
    Args:
        predicted_labels: List of predicted class labels
        confidence_scores: List of confidence scores
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(predicted_labels))
    bars = ax.barh(y_pos, confidence_scores, alpha=0.8)
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predicted_labels)
    ax.invert_yaxis()  # Top prediction at the top
    ax.set_xlabel('Confidence Score')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, confidence_scores)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.show()


def analyze_video_with_model(
    model_path: str, 
    video_path: str, 
    display_frames: bool = True,
    display_preds: bool = True,
    top_k: int = 5
):
    """
    Complete pipeline to analyze a video with a trained model.
    
    Args:
        model_path: Path to the saved model
        video_path: Path to the video file
        display_frames: Whether to display video frames
        display_preds: Whether to display predictions
        top_k: Number of top predictions to show
    """
    print("🎬 VJEPA2 Video Analysis Pipeline")
    print("=" * 50)
    
    # Load model
    model, processor = load_trained_model(model_path)
    
    # Print model info
    print_model_info(model)
    
    # Run inference
    predicted_labels, confidence_scores, video_tensor = run_inference(
        model, processor, video_path, top_k=top_k
    )
    
    # Display results
    print(f"\n🎯 PREDICTIONS FOR: {pathlib.Path(video_path).name}")
    print("-" * 40)
    for i, (label, score) in enumerate(zip(predicted_labels, confidence_scores)):
        print(f"{i+1}. {label}: {score:.3f} ({score*100:.1f}%)")
    
    if display_frames:
        print("\n📺 Video Frames:")
        display_video_frames(video_tensor)
    
    if display_preds:
        print("\n📊 Prediction Visualization:")
        display_predictions(predicted_labels, confidence_scores, 
                          f"Predictions for {pathlib.Path(video_path).name}")
    
    return model, processor, predicted_labels, confidence_scores
