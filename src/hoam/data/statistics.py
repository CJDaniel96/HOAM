import json
from pathlib import Path
from typing import List, Tuple, Union
 
torch = __import__('torch')
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
 
CACHE_VERSION = 2
CACHE_COMPUTED_ON = "raw_resized_tensor"

 
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
        channel_sum = torch.zeros(n_channels)
        channel_squared_sum = torch.zeros(n_channels)
        total_pixels = 0
 
        for imgs, _ in dataloader:
            imgs = imgs.view(imgs.size(0), n_channels, -1)
            channel_sum += imgs.sum(dim=(0, 2))
            channel_squared_sum += (imgs ** 2).sum(dim=(0, 2))
            total_pixels += imgs.size(0) * imgs.size(2)
 
        mean = channel_sum / total_pixels
        variance = channel_squared_sum / total_pixels - mean ** 2
        std = torch.sqrt(torch.clamp(variance, min=0.0))
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
            if (
                stats.get('version') == CACHE_VERSION
                and stats.get('computed_on') == CACHE_COMPUTED_ON
            ):
                return stats['mean'], stats['std']
 
        if image_size is None:
            raise ValueError(
                f"No cached stats found at {cache_path} and image_size is None, "
                "so stats cannot be computed. Provide a mean_std.json (see "
                "DataStatistics.load_mean_std) or pass an image_size."
            )

        # Compute statistics on resized raw tensors. Do not reuse
        # build_transforms('test'), because it applies normalization.
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
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
            json.dump({
                'mean': mean,
                'std': std,
                'version': CACHE_VERSION,
                'computed_on': CACHE_COMPUTED_ON,
            }, f, indent=2)
 
        return mean, std

    @staticmethod
    def load_mean_std(path: Union[str, Path]) -> Tuple[List[float], List[float]]:
        """
        Load mean and std from a JSON file produced by get_mean_std.

        Args:
            path: Path to a JSON file containing {"mean": [...], "std": [...]}.

        Returns:
            Tuple of mean list and std list.
        """
        with open(path) as f:
            stats = json.load(f)
        return stats['mean'], stats['std']
