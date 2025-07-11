from typing import Tuple
import torch
import torch.nn as nn
from timm import create_model
from .base import OrthogonalFusion, LaplacianLayer, GlobalPooling
 
 
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
    MLGModel class for extracting features using EfficientNetV2.

    Attributes:
        backbone (torch.nn.Module): EfficientNetV2 model for feature extraction.
        local_branch (torch.nn.Module): Multi-head self-attention for local features.
        global_branch (torch.nn.Module): Global feature extraction.
        fusion (torch.nn.Module): Orthogonal fusion of local and global features.
        head (torch.nn.Module): Head layers.

    Methods:
        forward(x): Forward pass of the model.
    """
    def __init__(self, model_name='efficientnetv2_s', pretrained=False, features_only=True, embedding_size=128) -> None:
        """
        Initialize the MLGModel instance.
        """
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=features_only)
        
        # Get the output feature shapes by passing a dummy input through the backbone
        dummy_input = torch.randn(1, 3, 224, 224)  # Adjust the input size as needed
        features = self.backbone(dummy_input)
        local_in_channels = features[-2].shape[1]
        global_in_channels = features[-1].shape[1]
        
        # Local branch with multi-head self-attention
        self.local_branch_conv = nn.Sequential(
            nn.Conv2d(local_in_channels, 1280, 1, 1, bias=False), 
            nn.BatchNorm2d(1280, 0.001), 
            nn.SiLU()
        )
        self.local_branch_attention = nn.MultiheadAttention(embed_dim=1280, num_heads=8)
        
        # Global branch
        self.global_branch = nn.Sequential(
            nn.Conv2d(global_in_channels, 1280, 1, 1, bias=False), 
            nn.BatchNorm2d(1280, 0.001), 
            nn.SiLU(), 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Orthogonal fusion
        self.orthogonal_fusion = OrthogonalFusion()
        
        # Head for final embedding
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280*2, embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size), 
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        features = self.backbone(x)
        local_features = self.local_branch_conv(features[-2])
        local_features = local_features.view(local_features.shape[0], local_features.shape[1], -1)
        local_features = local_features.permute(2, 0, 1)
        local_features, _ = self.local_branch_attention(local_features, local_features, local_features)
        local_features = local_features.permute(1, 2, 0)
        local_features = local_features.view(-1, 1280, 14, 14)
        global_features = self.global_branch(features[-1])
        # Orthogonal fusion
        feat = self.orthogonal_fusion(local_features, global_features)

        # Final embedding
        embedding = self.head(feat)

        return embedding
 
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
    def __init__(self, model_name='efficientnetv2_s', pretrained=False, features_only=True, embedding_size=128) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=features_only)
        dummy_input = torch.randn(1, 3, 224, 224)
        features = self.backbone(dummy_input)
        local_in_channels = features[-2].shape[1]
        global_in_channels = features[-1].shape[1]

        self.local_branch_conv = nn.Sequential(
            nn.Conv2d(local_in_channels, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280, 0.001),
            nn.SiLU()
        )
        self.local_laplacian = LaplacianLayer()
        self.local_branch_attention = nn.MultiheadAttention(embed_dim=1280, num_heads=8)

        self.global_branch = nn.Sequential(
            nn.Conv2d(global_in_channels, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280, 0.001),
            nn.SiLU()
        )
        self.global_pool = GlobalPooling()

        self.orthogonal_fusion = OrthogonalFusion(1280, 2560)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280 * 2, embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        features = self.backbone(x)
        local_features = self.local_branch_conv(features[-2])
        local_features = self.local_laplacian(local_features)
        B, C, H, W = local_features.shape
        local_flat = local_features.view(B, C, -1).permute(2, 0, 1)
        local_attended, _ = self.local_branch_attention(local_flat, local_flat, local_flat)
        local_features = local_attended.permute(1, 2, 0).view(B, C, H, W)

        global_features = self.global_branch(features[-1])
        global_features = self.global_pool(global_features).view(B, -1)

        fused = self.orthogonal_fusion(local_features, global_features)
        embedding = self.head(fused)
        return embedding