import shutil
from pathlib import Path
from typing import Optional, Union

import joblib
import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
from torchvision.datasets import ImageFolder

from .data.statistics import DataStatistics
from .data.transforms import build_transforms
from .utils import load_model


def _resolve_output_path(save_dir: Path, path: Union[str, Path]) -> Path:
    output_path = Path(path)
    if output_path.is_absolute():
        return output_path
    return save_dir / output_path


def build_knn_index(
    model_structure: str,
    model_path: Union[str, Path],
    data_dir: Union[str, Path],
    save_dir: Union[str, Path],
    embedding_size: int = 128,
    image_size: int = 224,
    mean_std_file: Optional[Union[str, Path]] = None,
    backbone_name: Optional[str] = None,
    index_path: Union[str, Path] = "knn.index",
    dataset_pkl: Union[str, Path] = "dataset.pkl",
    threshold: float = 0.5,
    batch_size: int = 64,
    num_workers: int = 0,
    device: Optional[str] = None,
) -> tuple[Path, Path, Path]:
    """
    Build and save a KNN reference index from the training split.

    Returns:
        (index_file, dataset_file, mean_std_file)
    """
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if mean_std_file:
        mean_std_path = Path(mean_std_file)
        mean, std = DataStatistics.load_mean_std(mean_std_path)
    else:
        mean, std = DataStatistics.get_mean_std(
            data_dir,
            image_size=image_size,
            num_workers=num_workers,
        )
        mean_std_path = data_dir / "mean_std.json"

    save_mean_std_path = save_dir / "mean_std.json"
    if mean_std_path.exists() and mean_std_path.resolve() != save_mean_std_path.resolve():
        shutil.copy(str(mean_std_path), str(save_mean_std_path))
        mean_std_path = save_mean_std_path

    transforms = build_transforms('val', image_size, mean, std)
    dataset = ImageFolder(data_dir / 'train', transforms)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(
        model_structure,
        str(model_path),
        embedding_size,
        device=device,
        backbone_name=backbone_name,
    )

    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=threshold)
    inf_model = InferenceModel(model, match_finder=match_finder, data_device=device)
    inf_model.train_knn(dataset, batch_size=batch_size)

    index_file = _resolve_output_path(save_dir, index_path)
    dataset_file = _resolve_output_path(save_dir, dataset_pkl)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.parent.mkdir(parents=True, exist_ok=True)

    inf_model.save_knn_func(str(index_file))
    joblib.dump(dataset, str(dataset_file))

    return index_file, dataset_file, mean_std_path
