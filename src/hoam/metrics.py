"""Classification-style metrics for the embedding model.

The model produces embeddings, not class logits, so predicted labels are
obtained by leave-one-out k-NN over the evaluation set: each sample is
classified by its nearest neighbours, excluding itself. This is consistent
with the retrieval metrics reported alongside (1-NN accuracy == precision@1).
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neighbors import NearestNeighbors


def knn_predict(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int = 1,
    metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leave-one-out k-NN prediction over a single embedding set.

    Each sample is labelled by majority vote of its ``k`` nearest neighbours,
    excluding itself (ties resolve to the smallest class id).

    Args:
        embeddings: Tensor of shape (N, D).
        labels: Integer class ids of shape (N,).
        k: Number of neighbours to vote over.
        metric: Distance metric for NearestNeighbors.

    Returns:
        (preds, confidence) as numpy arrays. ``confidence`` is the cosine
        similarity to the nearest neighbour.
    """
    emb = embeddings.detach().cpu().numpy().astype("float32")
    y = labels.detach().cpu().numpy().astype(np.int64)
    n = len(y)
    if n < 2:
        raise ValueError("Need at least 2 samples for leave-one-out k-NN.")
    k_eff = max(1, min(k, n - 1))

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric).fit(emb)
    dist, idx = nn.kneighbors(emb)
    neigh_labels = y[idx[:, 1:]]          # drop the self column
    nearest_dist = dist[:, 1]             # nearest non-self distance

    if k_eff == 1:
        preds = neigh_labels[:, 0]
    else:
        preds = np.array([np.bincount(row).argmax() for row in neigh_labels])

    if metric == "cosine":
        confidence = 1.0 - nearest_dist   # cosine similarity
    else:
        confidence = 1.0 / (1.0 + nearest_dist)
    return preds.astype(np.int64), confidence.astype(np.float64)


def classification_metrics(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    k: int = 1,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Compute accuracy / precision / recall / F1 / confusion matrix via k-NN.

    Returns:
        (summary, per_class_df, confusion_df) where ``summary`` holds scalar
        metrics, ``per_class_df`` holds per-class precision/recall/F1/support,
        and ``confusion_df`` is the confusion matrix labelled by class name.
    """
    preds, confidence = knn_predict(embeddings, labels, k=k)
    y = labels.detach().cpu().numpy().astype(np.int64)

    num_classes = int(max(int(y.max()), int(preds.max()))) + 1
    label_ids = list(range(num_classes))
    if class_names is None:
        class_names = [str(i) for i in label_ids]

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y, preds, labels=label_ids, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y, preds, labels=label_ids, average="weighted", zero_division=0
    )
    p_c, r_c, f1_c, support = precision_recall_fscore_support(
        y, preds, labels=label_ids, average=None, zero_division=0
    )

    summary = {
        "knn_k": int(k),
        "accuracy": float(accuracy_score(y, preds)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
        "mean_confidence": float(confidence.mean()),
    }
    per_class = pd.DataFrame(
        {
            "class": class_names,
            "precision": p_c,
            "recall": r_c,
            "f1": f1_c,
            "support": support,
        }
    )
    cm = confusion_matrix(y, preds, labels=label_ids)
    confusion_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    return summary, per_class, confusion_df


def save_classification_outputs(
    per_class: pd.DataFrame,
    confusion_df: pd.DataFrame,
    save_dir: Union[str, Path],
) -> None:
    """Write per-class metrics and the confusion matrix as labelled CSVs."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(save_path / "classification_report.csv", index=False)
    confusion_df.to_csv(save_path / "confusion_matrix.csv")
