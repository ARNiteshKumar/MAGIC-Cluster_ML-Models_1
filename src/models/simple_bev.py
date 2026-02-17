"""
Simple-BEV Model Architecture
Reference: https://github.com/aharley/simple_bev

Input:  [B, N_cams, 3, H, W]
Output: [B, num_classes, bev_H, bev_W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVEncoder(nn.Module):
    """ResNet-style backbone to extract per-camera features."""

    def __init__(self, in_channels: int = 3, out_channels: int = 128):
        super().__init__()
        self.conv1  = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.relu   = nn.ReLU(inplace=True)
        self.pool   = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64,  64,  2)
        self.layer2 = self._make_layer(64,  128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.proj   = nn.Conv2d(256, out_channels, 1)

    def _make_layer(self, ic, oc, n, stride=1):
        L = [nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True)]
        for _ in range(n - 1):
            L += [nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True)]
        return nn.Sequential(*L)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        return self.proj(self.layer3(self.layer2(self.layer1(x))))


class BEVSplat(nn.Module):
    """Project camera features onto the BEV grid."""
    def __init__(self, feat_dim=128, bev_h=200, bev_w=200):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (self.bev_h // 4, self.bev_w // 4))
        return self.bev_conv(x)


class BEVDecoder(nn.Module):
    """Upsample BEV features and predict segmentation classes."""
    def __init__(self, in_channels=128, num_classes=8):
        super().__init__()
        self.up1 = nn.Sequential(nn.ConvTranspose2d(in_channels, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, num_classes, 1))
    def forward(self, x):
        return self.head(self.up2(self.up1(x)))


class SimpleBEVModel(nn.Module):
    """Simple-BEV Multi-camera BEV Perception."""
    def __init__(self, ncams=6, feat_dim=128, bev_h=200, bev_w=200, num_classes=8):
        super().__init__()
        self.ncams   = ncams
        self.encoder = BEVEncoder(3, feat_dim)
        self.splat   = BEVSplat(feat_dim, bev_h, bev_w)
        self.fusion  = nn.Sequential(
            nn.Conv2d(feat_dim * ncams, feat_dim, 1), nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True))
        self.decoder = BEVDecoder(feat_dim, num_classes)
    def forward(self, imgs):
        B, N, C, H, W = imgs.shape
        feats = self.encoder(imgs.view(B * N, C, H, W))
        bev   = self.splat(feats)
        _, fC, bH, bW = bev.shape
        bev   = bev.view(B, N * fC, bH, bW)
        return self.decoder(self.fusion(bev))

def build_model(cfg):
    return SimpleBEVModel(ncams=cfg["model"]["ncams"], feat_dim=cfg["model"]["feat_dim"],
        bev_h=cfg["model"]["bev_h"], bev_w=cfg["model"]["bev_w"], num_classes=cfg["model"]["num_classes"])
