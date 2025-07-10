import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from pytorch_metric_learning.losses import SubCenterArcFaceLoss, TripletMarginLoss
from pytorch_metric_learning.miners import BatchHardMiner
 
 
def is_hybrid_loss(obj: nn.Module) -> bool:
    """
    Check if a criterion implements the hybrid margin interface.
 
    Returns:
        True if has attributes for hybrid loss components.
    """
    return (
        hasattr(obj, 'subcenter_arcface')
        and hasattr(obj, 'triplet_loss')
        and hasattr(obj, 'center_loss_centers')
    )
 
 
class HybridMarginLoss(nn.Module):
    """
    Hybrid margin loss combining:
      1) SubCenterArcFaceLoss
      2) TripletMarginLoss with batch-hard mining
      3) Center loss
 
    Args:
        num_classes: Number of target classes.
        embedding_size: Size of the embeddings.
        subcenter_margin: Angular margin for SubCenterArcFace.
        subcenter_scale: Scale for SubCenterArcFace.
        sub_centers: Number of sub-centers per class.
        triplet_margin: Margin for TripletMarginLoss.
        center_loss_weight: Weight for center loss component.
    """
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        subcenter_margin: float = 0.4,
        subcenter_scale: float = 30.0,
        sub_centers: int = 3,
        triplet_margin: float = 0.3,
        center_loss_weight: float = 0.01,
    ) -> None:
        super().__init__()
        # SubCenter ArcFace component
        self.subcenter_arcface = SubCenterArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
            sub_centers=sub_centers,
            margin=subcenter_margin,
            scale=subcenter_scale,
        )
        # Triplet margin loss
        self.triplet_loss = TripletMarginLoss(margin=triplet_margin)
        # Trainable centers for center loss
        self.center_loss_centers = nn.Parameter(
            torch.randn(num_classes, embedding_size), requires_grad=True
        )
        self.center_loss_weight = center_loss_weight
 
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute hybrid loss.
 
        Args:
            embeddings: Tensor of shape (B, D).
            labels: LongTensor of shape (B,).
 
        Returns:
            Combined scalar loss.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
 
        # ArcFace part
        loss_arc = self.subcenter_arcface(embeddings, labels)
 
        # Triplet part with batch-hard mining
        miner = BatchHardMiner()
        hard_pairs = miner(embeddings, labels)
        loss_triplet = self.triplet_loss(embeddings, labels, hard_pairs)
 
        # Center loss: L2 between embeddings and their class centers
        centers_batch = self.center_loss_centers[labels]
        loss_center = F.mse_loss(embeddings, centers_batch)
 
        # Weighted sum
        loss = loss_arc + loss_triplet + self.center_loss_weight * loss_center
        return loss