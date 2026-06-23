import json
from pathlib import Path
from typing import Dict, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.metrics import silhouette_score


def evaluate_model_on_testset(
    model: torch.nn.Module,
    test_dataset: Dataset,
    save_dir: Union[str, Path],
    batch_size: int = 64,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate a model on a test dataset using metric-learning metrics.

    Args:
        model: Trained model mapping images to embeddings.
        test_dataset: Dataset for evaluation (with transforms applied).
        save_dir: Directory to save metrics (JSON + CSV).
        batch_size: Batch size for the DataLoader.
        device: Device to run inference on ('cuda' or 'cpu').

    Returns:
        Dictionary of evaluation metrics.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    model.to(device).eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            embeddings = model(imgs)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)

    # Query and reference are the same set. Passing only (query, labels) lets
    # AccuracyCalculator set ref_includes_query=True and exclude each point's
    # self-match; otherwise precision_at_1 is trivially 1.0.
    acc_calc = AccuracyCalculator(
        include=('precision_at_1', 'mean_average_precision', 'mean_reciprocal_rank'),
        k=10,
    )
    metrics = acc_calc.get_accuracy(embeddings, labels)

    # Clustering quality of the embedding space.
    metrics['silhouette_score'] = float(
        silhouette_score(embeddings.numpy(), labels.numpy())
    )

    with open(save_path / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(save_path / 'test_metrics.csv', index=False)

    return metrics
