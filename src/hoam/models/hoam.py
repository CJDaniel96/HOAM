from typing import Tuple
import torch
import torch.nn as nn
from timm import create_model
from .base import OrthogonalFusion, LaplacianLayer
 
 
def _get_backbone_channels(
    backbone: nn.Module,
) -> Tuple[int, int]:
    """
    Helper to retrieve local and global feature channel sizes from a timm feature extractor.
 
    Args:
        backbone: timm model with features_only=True
 
    Returns:
        Tuple of (local_channels, global_channels)
    """
    # feature_info stores metadata about each stage
    info = getattr(backbone, "feature_info")
    channels = info.channels()
    # take the penultimate and last feature maps
    return channels[-2], channels[-1]
 
 
class HOAM(nn.Module):
    """
    HOAM model: orthogonal fusion of local and global features from an EfficientNet backbone.
    """
    def __init__(
        self,
        backbone_name: str = "efficientnetv2_s",
        pretrained: bool = False,
        embedding_size: int = 128,
    ) -> None:
        super().__init__()
        # backbone feature extractor
        self.backbone = create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
        )
        local_ch, global_ch = _get_backbone_channels(self.backbone)
 
        # local branch: conv + multihead attention
        self.local_conv = nn.Sequential(
            nn.Conv2d(local_ch, global_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(global_ch),
            nn.SiLU(),
        )
        self.attn = nn.MultiheadAttention(embed_dim=global_ch, num_heads=8)
 
        # global branch: pooling + linear
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(global_ch, global_ch)
 
        # fusion and head
        self.fusion = OrthogonalFusion(local_dim=global_ch, global_dim=global_ch)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(global_ch * 2, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # extract features
        feats = self.backbone(x)
        local_map = self.local_conv(feats[-2])  # [B, C, H, W]
        B, C, H, W = local_map.shape
        # prepare for attention: [HW, B, C]
        local_flat = local_map.flatten(2).permute(2, 0, 1)
        attn_out, _ = self.attn(local_flat, local_flat, local_flat)
        local_feat = attn_out.permute(1, 2, 0).view(B, C, H, W)
 
        global_feat = self.global_pool(feats[-1]).view(B, -1)
        global_feat = self.global_fc(global_feat)
 
        fused = self.fusion(local_feat, global_feat)
        return self.head(fused)
 
 
class HOAMV2(nn.Module):
    """
    HOAMV2: adds a Laplacian filter on local features and dual pooling on global features.
    """
    def __init__(
        self,
        backbone_name: str = "efficientnetv2_s",
        pretrained: bool = False,
        embedding_size: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
        )
        local_ch, global_ch = _get_backbone_channels(self.backbone)
 
        # local branch: conv + laplacian + attention
        self.local_conv = nn.Sequential(
            nn.Conv2d(local_ch, global_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(global_ch),
            nn.SiLU(),
        )
        self.laplacian = LaplacianLayer(global_ch)
        self.attn = nn.MultiheadAttention(embed_dim=global_ch, num_heads=8)
 
        # global branch: conv + avg&max pooling
        self.global_conv = nn.Sequential(
            nn.Conv2d(global_ch, global_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(global_ch),
            nn.SiLU(),
        )
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.global_max = nn.AdaptiveMaxPool2d(1)
 
        # fusion and head
        # after fusion: channels = 2 * global_ch
        self.fusion = OrthogonalFusion(local_dim=global_ch, global_dim=2 * global_ch)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(global_ch * 2, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        # local path
        local = self.local_conv(feats[-2])
        local = self.laplacian(local)
        B, C, H, W = local.shape
        local_flat = local.flatten(2).permute(2, 0, 1)
        attn_out, _ = self.attn(local_flat, local_flat, local_flat)
        local_feat = attn_out.permute(1, 2, 0).view(B, C, H, W)
 
        # global path
        g = self.global_conv(feats[-1])
        avg = self.global_avg(g)
        mx = self.global_max(g)
        global_feat = torch.cat([avg, mx], dim=1).view(B, -1)
 
        fused = self.fusion(local_feat, global_feat)
        return self.head(fused)