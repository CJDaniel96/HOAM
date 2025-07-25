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
    def __init__(self, channels=1280):
        """
        初始化時需要傳入固定的 channel 數量。
        """
        super().__init__()
        # 創建一個深度可分離卷積
        # in_channels = out_channels = groups = channels
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            groups=channels, # 關鍵：設置 groups = in_channels 實現深度可分離卷積
            bias=False
        )

        # 創建拉普拉斯核
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).float()
        # 將核的形狀擴展為 (out_channels, 1, H, W)，即 (channels, 1, 3, 3)
        # 每個輸出的 channel 都使用這個相同的核
        self.conv.weight.data = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.conv.weight.requires_grad = False # 鎖定權重

    def forward(self, x):
        return self.conv(x)
    

class LearnableEdgeLayer(nn.Module):
    """
    可學習的邊緣檢測層，使用深度可分離卷積
    """
    def __init__(self, channels=1280, kernel_size=3, use_laplacian_init=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        # 深度可分離卷積
        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,  # 深度可分離
            bias=False
        )
        # 可選：使用拉普拉斯核初始化
        if use_laplacian_init:
            self._initialize_with_laplacian()
        # 批次歸一化（可選）
        self.bn = nn.BatchNorm2d(channels)
    def _initialize_with_laplacian(self):
        """使用拉普拉斯核初始化權重"""
        if self.kernel_size == 3:
            # 標準拉普拉斯核
            laplacian_kernel = torch.tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ]).float()
        else:
            # 為其他核大小創建近似的拉普拉斯核
            laplacian_kernel = torch.zeros(self.kernel_size, self.kernel_size)
            center = self.kernel_size // 2
            laplacian_kernel[center, center] = -4
            laplacian_kernel[center-1, center] = 1
            laplacian_kernel[center+1, center] = 1
            laplacian_kernel[center, center-1] = 1
            laplacian_kernel[center, center+1] = 1
        # 擴展到所有通道
        with torch.no_grad():
            self.depthwise_conv.weight.data = laplacian_kernel.unsqueeze(0).unsqueeze(0).repeat(
                self.channels, 1, 1, 1
            )
    def forward(self, x):
        edge_features = self.depthwise_conv(x)
        edge_features = self.bn(edge_features)
        return edge_features
 
 
class OrthogonalFusion(nn.Module):
    def __init__(self, input_dim_local=1280, input_dim_global=1280):
        super().__init__()
        if input_dim_global != input_dim_local:
            self.projector = nn.Linear(input_dim_global, input_dim_local)
        else:
            self.projector = nn.Identity()

    def forward(self, local_feat, global_feat):
        B, C_local, H, W = local_feat.shape
        global_feat = self.projector(global_feat)

        global_feat_norm = torch.norm(global_feat, p=2, dim=1, keepdim=True) + 1e-6
        global_unit = global_feat / global_feat_norm
        local_flat = local_feat.view(B, C_local, -1)

        projection = torch.bmm(global_unit.unsqueeze(1), local_flat)
        projection = torch.bmm(global_unit.unsqueeze(2), projection).view(B, C_local, H, W)

        orthogonal_comp = local_feat - projection
        global_map = global_feat.unsqueeze(-1).unsqueeze(-1).expand_as(orthogonal_comp)

        return torch.cat([global_map, orthogonal_comp], dim=1)
    
class GlobalPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)