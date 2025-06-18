import pathlib
import random
import wandb
from functools import partial
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_random_indices
from torchvision.transforms import v2
from transformers import VJEPA2VideoProcessor, VJEPA2ForVideoClassification

def get_best_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class VideoDataset(Dataset):
    """Custom dataset for loading video files and labels."""

    def __init__(self, file_paths: List[pathlib.Path], label_map: dict):
        self.file_paths = file_paths
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        decoder = VideoDecoder(path)
        label = self.label_map[path.parts[2]]
        return decoder, label


def video_collate_fn(
    samples: List[tuple], frames_per_clip: int, transforms: v2.Compose
):
    """Sample clips and apply transforms to a batch."""
    clips, labels = [], []
    for decoder, lbl in samples:
        clip = clips_at_random_indices(
            decoder,
            num_clips=1,
            num_frames_per_clip=frames_per_clip,
            num_indices_between_frames=3,
        ).data
        clips.append(clip)
        labels.append(lbl)

    videos = torch.cat(clips, dim=0)
    videos = transforms(videos)
    return videos, torch.tensor(labels)


def split_dataset(paths: List[pathlib.Path]):
    """Split paths into train/val/test based on directory keyword."""
    train, val, test = [], [], []
    for p in paths:
        if "train" in p.parts:
            train.append(p)
        elif "val" in p.parts:
            val.append(p)
        elif "test" in p.parts:
            test.append(p)
        else:
            raise ValueError(f"Unrecognized split for path: {p}")
    return train, val, test


def evaluate(
    loader: DataLoader,
    model: VJEPA2ForVideoClassification,
    processor: VJEPA2VideoProcessor,
    device: torch.device,
) -> float:
    """Compute accuracy over a dataset."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for vids, labels in loader:
            inputs = processor(vids, return_tensors="pt").to(device)
            labels = labels.to(device)
            logits = model(**inputs).logits
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def main():
    # Configuration
    batch_size = 2
    num_workers = 8
    lr = 1e-5
    num_epochs = 5
    accumulation_steps = 4
    use_wandb = True  # Set to False to disable wandb logging

    device = get_best_device()
    print(f"Using device: {device}")

    # Load and shuffle data paths
    root = pathlib.Path("UCF101_subset")
    all_paths = list(root.rglob("*.avi"))
    random.shuffle(all_paths)

    train_paths, val_paths, test_paths = split_dataset(all_paths)
    print(f"Splits -> train: {len(train_paths)}, val: {len(val_paths)}, test: {len(test_paths)}")

    # Label mappings
    classes = sorted({p.parts[2] for p in all_paths})
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}

    # Datasets
    train_ds = VideoDataset(train_paths, label2id)
    val_ds = VideoDataset(val_paths, label2id)
    test_ds = VideoDataset(test_paths, label2id)

    # Model & processor
    model_name = "qubvel-hf/vjepa2-vitl-fpc16-256-ssv2"
    processor = VJEPA2VideoProcessor.from_pretrained(model_name)
    model = VJEPA2ForVideoClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Transforms
    train_transforms = v2.Compose([
        v2.RandomResizedCrop((processor.crop_size["height"], processor.crop_size["width"])),
        v2.RandomHorizontalFlip(),
    ])
    eval_transforms = v2.Compose([
        v2.CenterCrop((processor.crop_size["height"], processor.crop_size["width"]))
    ])

    # DataLoaders (disable pin_memory for MPS compatibility)
    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(video_collate_fn, frames_per_clip=model.config.frames_per_clip, transforms=train_transforms),
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(video_collate_fn, frames_per_clip=model.config.frames_per_clip, transforms=eval_transforms),
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(video_collate_fn, frames_per_clip=model.config.frames_per_clip, transforms=eval_transforms),
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    # Freeze base encoder
    for param in model.vjepa2.parameters():
        param.requires_grad = False

    # Optimizer and loss
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)


    # Initialize wandb if enabled
    run = None
    if use_wandb:
        run = wandb.init(
            project="vjepa-training",
            name="vjepa2-ucf101-finetuning",
        )

    # Training loop with gradient accumulation and evaluation
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, (vids, labels) in enumerate(train_loader, start=1):
            inputs = processor(vids, return_tensors="pt").to(device)
            labels = labels.to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            running_loss += loss.item()

            if step % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(f"Epoch {epoch} Step {step}: Accumulated Loss = {running_loss:.4f}")
                if run:
                    run.log({"train/loss": loss})
                running_loss = 0.0

        # End of epoch evaluation
        val_acc = evaluate(val_loader, model, processor, device)
        print(f"Epoch {epoch} Validation Accuracy: {val_acc:.4f}")
        if run:
            run.log({"val/accuracy": val_acc})


    # Final test evaluation
    test_acc = evaluate(test_loader, model, processor, device)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    if run:
        run.log({"test/accuracy": test_acc})
        run.finish()

    # Ensure model directory exists
    model_dir = pathlib.Path("./model")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save the model and processor
    model_save_path = "./model/vjepa2-vitl-fpc16-256-ssv2-uvf101"
    model.save_pretrained(model_save_path)
    processor.save_pretrained(model_save_path)
    print(f"Model and processor saved to {model_save_path}")


    # Upload the model to Hub
    # model.push_to_hub("ariG23498/vjepa2-vitl-fpc16-256-ssv2-uvf101")
    # processor.push_to_hub("ariG23498/vjepa2-vitl-fpc16-256-ssv2-uvf101")

if __name__ == "__main__":
    main()
