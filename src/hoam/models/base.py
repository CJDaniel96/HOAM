import torch
import torch.nn as nn


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
