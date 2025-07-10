import json
from pathlib import Path
from typing import List, Tuple
 
torch = __import__('torch')
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
 
from .transforms import build_transforms
 
 
class DataStatistics:
    """
    Compute or load dataset mean and standard deviation for normalization.
    """
 
    @staticmethod
    def compute_mean_std(
        dataloader: DataLoader
    ) -> Tuple[List[float], List[float]]:
        """
        Compute channel-wise mean and std across a DataLoader.
 
        Args:
            dataloader: DataLoader yielding batches of images [B, C, H, W].
 
        Returns:
            Tuple of two lists: means and stds for each channel.
        """
        n_channels = next(iter(dataloader))[0].shape[1]
        mean = torch.zeros(n_channels)
        std = torch.zeros(n_channels)
        total_samples = 0
 
        for imgs, _ in dataloader:
            batch_samples = imgs.size(0)
            imgs = imgs.view(batch_samples, n_channels, -1)
            mean += imgs.mean(2).sum(0)
            std += imgs.std(2).sum(0)
            total_samples += batch_samples
 
        mean /= total_samples
        std /= total_samples
        return mean.tolist(), std.tolist()
 
    @staticmethod
    def get_mean_std(
        data_dir: Path,
        image_size: int,
        batch_size: int = 32,
        num_workers: int = 4,
        cache_file: str = "mean_std.json"
    ) -> Tuple[List[float], List[float]]:
        """
        Load mean and std from cache or compute and save.
 
        Args:
            data_dir: Path to dataset root (expects 'train' subfolder).
            image_size: Size for resizing images.
            batch_size: Batch size for DataLoader.
            num_workers: Number of worker processes.
            cache_file: Filename under data_dir to load/save stats.
 
        Returns:
            Tuple of mean list and std list.
        """
        cache_path = data_dir / cache_file
        if cache_path.exists():
            with cache_path.open('r') as f:
                stats = json.load(f)
            return stats['mean'], stats['std']
 
        # Create DataLoader without normalization
        transform = build_transforms(mode='test', image_size=image_size)
        dataset = ImageFolder(str(data_dir / 'train'), transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        mean, std = DataStatistics.compute_mean_std(loader)
 
        # Save to cache
        with cache_path.open('w') as f:
            json.dump({'mean': mean, 'std': std}, f, indent=2)
 
        return mean, std