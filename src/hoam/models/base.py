from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
 
 
class ArcFace(nn.Module):
    """
    Implements the ArcFace loss module.
    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition".
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 64.0,
        margin: float = 0.50,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
 
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
 
        # pre-compute cos(m) and sin(m)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.criterion = nn.CrossEntropyLoss()
 
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            embeddings (Tensor): Input feature matrix of shape (N, in_features).
            labels (Tensor): Ground-truth labels of shape (N,).
 
        Returns:
            Tuple[Tensor, Tensor]: (loss, logits)
        """
        # normalize features and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
 
        # cosine similarity
        cosine = F.linear(embeddings, weight_norm)
        sine = torch.sqrt((1.0 - cosine ** 2).clamp(min=1e-6))
 
        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
 
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
 
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale
 
        loss = self.criterion(logits, labels)
        return loss, logits
 
 
class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    """
    def __init__(
        self,
        p: float = 3.0,
        eps: float = 1e-6,
        learn_p: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=learn_p)
 
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).
        Returns:
            Tensor: Pooled feature of shape (B, C).
        """
        return self.gem(x, self.p, self.eps)
 
    @staticmethod
    def gem(x: Tensor, p: Tensor, eps: float) -> Tensor:
        return F.adaptive_avg_pool2d(x.clamp(min=eps).pow(p), (1, 1)).pow(1.0 / p).squeeze(-1).squeeze(-1)
 
 
class LaplacianLayer(nn.Module):
    """
    Depthwise Laplacian convolution layer (fixed weights).
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape (1,1,3,3)
        weight = kernel.repeat(channels, 1, 1, 1)  # shape (C,1,3,3)
 
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )
        self.conv.weight = nn.Parameter(weight, requires_grad=False)
 
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
 
 
class OrthogonalFusion(nn.Module):
    """
    Fuse local feature maps and global feature vectors via orthogonal decomposition.
    """
    def __init__(
        self,
        local_dim: int,
        global_dim: int,
    ) -> None:
        super().__init__()
        if local_dim != global_dim:
            self.projector = nn.Linear(global_dim, local_dim)
        else:
            self.projector = nn.Identity()
 
    def forward(self, local_feat: Tensor, global_feat: Tensor) -> Tensor:
        """
        Args:
            local_feat (Tensor): Local maps, shape (B, C_l, H, W)
            global_feat (Tensor): Global vectors, shape (B, C_g)
        Returns:
            Tensor: Fused feature maps, shape (B, C_l + C_l, H, W)
        """
        B, C_l, H, W = local_feat.shape
        g = self.projector(global_feat)  # (B, C_l)
        g_norm = g.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
        u = g / g_norm  # (B, C_l)
 
        flat = local_feat.view(B, C_l, -1)  # (B, C_l, H*W)
        proj = (u.unsqueeze(1) @ flat)  # (B,1,H*W)
        proj = (u.unsqueeze(2) * proj).view(B, C_l, H, W)  # (B,C_l,H,W)
 
        orth = local_feat - proj
        global_map = g.view(B, C_l, 1, 1).expand_as(orth)
        return torch.cat([global_map, orth], dim=1)