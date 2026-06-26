import json
from pathlib import Path
from typing import Dict, Union

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import silhouette_score

from .metrics import classification_metrics, save_classification_outputs


def _retrieval_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
) -> Dict[str, float]:
    """
    Compute retrieval metrics without FAISS-backed helpers.

    The PyTorch Metric Learning AccuracyCalculator defaults to FAISS, which can
    segfault on some CPU-only/macOS environments. A small cosine-similarity
    implementation is enough for evaluation sets handled by this CLI.
    """
    if embeddings.size(0) < 2:
        raise ValueError("Need at least 2 samples for retrieval metrics.")

    emb = F.normalize(embeddings.float(), p=2, dim=1)
    labels = labels.view(-1)
    n = emb.size(0)
    k_eff = max(1, min(k, n - 1))

    similarities = emb @ emb.T
    similarities.fill_diagonal_(float("-inf"))
    ranked_indices = similarities.argsort(dim=1, descending=True)[:, :k_eff]
    ranked_labels = labels[ranked_indices]
    query_labels = labels.unsqueeze(1)
    matches = ranked_labels.eq(query_labels)

    precision_at_1 = matches[:, 0].float().mean().item()
    average_precisions = []
    reciprocal_ranks = []

    ranks = torch.arange(1, k_eff + 1, dtype=torch.float32)
    for query_idx in range(n):
        query_matches = matches[query_idx].float()
        total_relevant = int(labels.eq(labels[query_idx]).sum().item()) - 1
        if total_relevant <= 0:
            average_precisions.append(0.0)
            reciprocal_ranks.append(0.0)
            continue

        cumulative_matches = torch.cumsum(query_matches, dim=0)
        precision_at_ranks = cumulative_matches / ranks
        ap = (precision_at_ranks * query_matches).sum() / min(total_relevant, k_eff)
        average_precisions.append(float(ap.item()))

        relevant_rank = torch.nonzero(query_matches, as_tuple=False)
        if relevant_rank.numel() == 0:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / float(relevant_rank[0].item() + 1))

    return {
        "precision_at_1": float(precision_at_1),
        "mean_average_precision": float(sum(average_precisions) / n),
        "mean_reciprocal_rank": float(sum(reciprocal_ranks) / n),
    }


def evaluate_model_on_testset(
    model: torch.nn.Module,
    test_dataset: Dataset,
    save_dir: Union[str, Path],
    batch_size: int = 64,
    device: str = 'cuda',
    knn_k: int = 1,
) -> Dict[str, float]:
    """
    Evaluate a model on a test dataset using metric-learning metrics.

    Args:
        model: Trained model mapping images to embeddings.
        test_dataset: Dataset for evaluation (with transforms applied).
        save_dir: Directory to save metrics (JSON + CSV).
        batch_size: Batch size for the DataLoader.
        device: Device to run inference on ('cuda' or 'cpu').
        knn_k: k for the leave-one-out k-NN used for classification metrics.

    Returns:
        Dictionary of evaluation metrics. Also writes test_metrics.{json,csv},
        classification_report.csv and confusion_matrix.csv to save_dir.
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

    # Query and reference are the same set, so exclude each point's self-match;
    # otherwise precision_at_1 would be trivially 1.0.
    metrics = _retrieval_metrics(embeddings, labels, k=10)

    # Clustering quality of the embedding space.
    metrics['silhouette_score'] = float(
        silhouette_score(embeddings.numpy(), labels.numpy())
    )

    # Classification metrics via leave-one-out k-NN: accuracy / precision /
    # recall / F1 / confusion matrix / confidence. Class names come from the
    # dataset when available (e.g. ImageFolder.classes).
    class_names = getattr(test_dataset, 'classes', None)
    cls_summary, per_class, confusion_df = classification_metrics(
        embeddings, labels, class_names=class_names, k=knn_k
    )
    metrics.update(cls_summary)
    save_classification_outputs(per_class, confusion_df, save_path)

    with open(save_path / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(save_path / 'test_metrics.csv', index=False)

    return metrics
