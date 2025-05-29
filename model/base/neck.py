import torch
import torch.nn as nn
from .components import Conv, C2f

class Neck(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.kernelSize = 3
        self.stride = 2

        # 定义输入/输出通道数
        # 来自 Backbone 的特征图通道数
        ch_feat1_bk = int(256 * w)  # P3 level from backbone (80x80)
        ch_feat2_bk = int(512 * w)  # P4 level from backbone (40x40)
        ch_feat3_bk = int(512 * w * r) # P5 level from backbone (20x20)

        # Neck 输出的特征图通道数 (与期望输出对应)
        # C: (B, 512 * w, 40, 40) -> ch_feat2_bk
        # X: (B, 256 * w, 80, 80) -> ch_feat1_bk
        # Y: (B, 512 * w, 40, 40) -> ch_feat2_bk
        # Z: (B, 512 * w * r, 20, 20) -> ch_feat3_bk

        # --- 自顶向下路径 (Top-down path) ---

        # 1. 处理 feat3_bk (P5_in) 到 P4_td (输出 C)
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 输入通道: 上采样后的feat3_bk通道 + feat2_bk通道
        # 输出通道: ch_feat2_bk (对应 C 和 Y 的通道)
        self.c2f_p4_td = C2f(c1=ch_feat3_bk + ch_feat2_bk, c2=ch_feat2_bk, n=n, shortcut=True, e=0.5)

        # 2. 处理 P4_td 到 P3_out (输出 X)
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 输入通道: 上采样后的P4_td通道 + feat1_bk通道
        # 输出通道: ch_feat1_bk (对应 X 的通道)
        self.c2f_p3_out = C2f(c1=ch_feat2_bk + ch_feat1_bk, c2=ch_feat1_bk, n=n, shortcut=True, e=0.5)


        # --- 自底向上路径 (Bottom-up path) ---

        # 3. 处理 P3_out 到 P4_out (输出 Y)
        # P3_out (来自c2f_p3_out) 需要下采样
        self.p3_downsample_conv = Conv(c1=ch_feat1_bk, c2=ch_feat1_bk, k=self.kernelSize, s=self.stride, p=1) # 输出通道保持ch_feat1_bk
        # 输入通道: 下采样后的P3_out通道 + P4_td通道
        # 输出通道: ch_feat2_bk (对应 Y 的通道)
        self.c2f_p4_out = C2f(c1=ch_feat1_bk + ch_feat2_bk, c2=ch_feat2_bk, n=n, shortcut=True, e=0.5)

        # 4. 处理 P4_out 到 P5_out (输出 Z)
        # P4_out (来自c2f_p4_out) 需要下采样
        self.p4_downsample_conv = Conv(c1=ch_feat2_bk, c2=ch_feat2_bk, k=self.kernelSize, s=self.stride, p=1) # 输出通道保持ch_feat2_bk
        # 输入通道: 下采样后的P4_out通道 + feat3_bk通道 (来自Backbone的P5原始输入)
        # 输出通道: ch_feat3_bk (对应 Z 的通道)
        self.c2f_p5_out = C2f(c1=ch_feat2_bk + ch_feat3_bk, c2=ch_feat3_bk, n=n, shortcut=True, e=0.5)

        # raise NotImplementedError("Neck::__init__")

    def forward(self, feat1, feat2, feat3):
        """
        Input shape:
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        Output shape:
            C: (B, 512 * w, 40, 40)
            X: (B, 256 * w, 80, 80)
            Y: (B, 512 * w, 40, 40)
            Z: (B, 512 * w * r, 20, 20)
        """

        # --- 自顶向下路径 ---
        # P5_td_in is feat3
        p5_upsampled = self.p5_upsample(feat3)  # (20x20 -> 40x40)
        # Concat P5_upsampled with feat2 (backbone P4)
        p4_td_concat = torch.cat([p5_upsampled, feat2], dim=1)
        p4_td = self.c2f_p4_td(p4_td_concat)  # Output C: P4_td

        p4_upsampled = self.p4_upsample(p4_td)    # (40x40 -> 80x80)
        # Concat P4_upsampled with feat1 (backbone P3)
        p3_out_concat = torch.cat([p4_upsampled, feat1], dim=1)
        p3_out = self.c2f_p3_out(p3_out_concat) # Output X: P3_out


        # --- 自底向上路径 ---
        p3_downsampled = self.p3_downsample_conv(p3_out) # (80x80 -> 40x40)
        # Concat p3_downsampled with p4_td (from top-down path)
        p4_out_concat = torch.cat([p3_downsampled, p4_td], dim=1)
        p4_out = self.c2f_p4_out(p4_out_concat) # Output Y: P4_out

        p4_downsampled = self.p4_downsample_conv(p4_out) # (40x40 -> 20x20)
        # Concat p4_downsampled with feat3 (original P5 from backbone)
        # This is a common PANet connection pattern.
        p5_out_concat = torch.cat([p4_downsampled, feat3], dim=1)
        p5_out = self.c2f_p5_out(p5_out_concat) # Output Z: P5_out

        return p4_td, p3_out, p4_out, p5_out # Corresponds to C, X, Y, Z
        # raise NotImplementedError("Neck::forward")
