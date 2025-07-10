from pathlib import Path

from typing import Union, Dict

import json
 
import torch

import pandas as pd

from torch.utils.data import DataLoader, Dataset

from pytorch_metric_learning.utils.inference import InferenceModel

from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from sklearn.metrics import silhouette_score
 
 
def evaluate_model_on_testset(

    model: torch.nn.Module,

    test_dataset: Dataset,

    save_dir: Union[str, Path],

    batch_size: int = 64,

    device: str = 'cuda'

) -> Dict[str, float]:

    """

    Evaluate a model on a test dataset using metric learning metrics.
 
    Args:

        model: Trained PyTorch model.

        test_dataset: Dataset for evaluation (with transforms applied).

        save_dir: Directory to save metrics and CSV.

        batch_size: Batch size for DataLoader.

        device: Device ('cuda' or 'cpu').
 
    Returns:

        Dictionary of evaluation metrics.

    """

    save_path = Path(save_dir)

    save_path.mkdir(parents=True, exist_ok=True)
 
    # Prepare inference model

    inference_model = InferenceModel(model=model)

    loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        pin_memory=True

    )
 
    all_embeddings = []

    all_labels = []

    model.to(device).eval()
 
    with torch.no_grad():

        for imgs, labels in loader:

            imgs = imgs.to(device)

            feats = inference_model(imgs)

            all_embeddings.append(feats.cpu())

            all_labels.append(labels.cpu())
 
    embeddings = torch.cat(all_embeddings)

    labels = torch.cat(all_labels)
 
    # Compute accuracy metrics

    acc_calc = AccuracyCalculator(

        include=('precision_at_1', 'mean_average_precision', 'mean_reciprocal_rank'),

        k=10

    )

    metrics = acc_calc.get_accuracy(

        embeddings, labels, embeddings, labels

    )

    # Silhouette score

    metrics['silhouette_score'] = float(

        silhouette_score(embeddings.numpy(), labels.numpy())

    )
 
    # Save results

    # JSON

    with open(save_path / 'test_metrics.json', 'w') as f:

        json.dump(metrics, f, indent=2)

    # CSV

    pd.DataFrame([metrics]).to_csv(save_path / 'test_metrics.csv', index=False)
 
    return metrics
