from huggingface_hub import hf_hub_download
import tarfile
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for downloading the UCF101 subset dataset."""
    dataset_id: str = "sayakpaul/ucf101-subset"
    filename: str = "UCF101_subset.tar.gz"

def download_dataset(config: DatasetConfig):
    fpath = hf_hub_download(repo_id=config.dataset_id, filename=config.filename, repo_type="dataset")
    with tarfile.open(fpath) as t:
        t.extractall(".")


if __name__ == "__main__":
    config = DatasetConfig()
    download_dataset(config)
    print(f"Dataset {config.dataset_id} downloaded and extracted successfully.")