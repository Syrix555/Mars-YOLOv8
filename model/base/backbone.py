import torch.nn as nn
from .components import Conv, C2f, SPPF

class Backbone(nn.Module):
    """
    Reference: resources/yolov8.jpg
    """
    def __init__(self, w, r, n):
        super().__init__()
        self.imageChannel = 3
        self.kernelSize = 3
        self.stride = 2

        # 辅助函数，用于根据w和r计算实际通道数
        def scale_channels(base_ch, width_mult, ratio_mult=1.0):
            """缩放通道数，并确保至少为1。"""
            return max(1, int(base_ch * width_mult * ratio_mult))

        # Stem (初始主干层)
        # 输入: (B, 3, 640, 640) -> 输出: (B, ch_stem_out, 320, 320)
        ch_stem_out = scale_channels(64, w)
        self.stem = Conv(self.imageChannel, ch_stem_out, self.kernelSize, self.stride, p=1)

        # Stage 1 (产生 feat0)
        ch_s1_out = scale_channels(128, w)
        self.s1_conv = Conv(ch_stem_out, ch_s1_out, self.kernelSize, self.stride, p=1)
        self.s1_c2f = C2f(ch_s1_out, ch_s1_out, n=n, shortcut=True, e=0.5)

        # Stage 2 (产生 feat1)
        ch_s2_out = scale_channels(256, w)
        self.s2_conv = Conv(ch_s1_out, ch_s2_out, self.kernelSize, self.stride, p=1)
        self.s2_c2f = C2f(ch_s2_out, ch_s2_out, n=n, shortcut=True, e=0.5)

        # Stage 3 (产生 feat2)
        ch_s3_out = scale_channels(512, w)
        self.s3_conv = Conv(ch_s2_out, ch_s3_out, self.kernelSize, self.stride, p=1)
        self.s3_c2f = C2f(ch_s3_out, ch_s3_out, n=n, shortcut=True, e=0.5)

        # Stage 4 (产生 feat3)
        ch_s4_out = scale_channels(512, w, r)
        self.s4_conv = Conv(ch_s3_out, ch_s4_out, self.kernelSize, self.stride, p=1)
        self.s4_c2f = C2f(ch_s4_out, ch_s4_out, n=n, shortcut=True, e=0.5)
        self.sppf = SPPF(ch_s4_out, ch_s4_out, k=5)

        # "增加"的部分：调用权重初始化方法
        self._initialize_weights()

    # "增加"的部分：权重初始化方法
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 你的Conv模块中 nn.Conv2d 的 bias=False
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # SiLU激活函数特性接近ReLU
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None: # 确保权重存在 (affine=True时)
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None: # 确保偏置存在 (affine=True时)
                    nn.init.constant_(m.bias, 0)
            # 可以为其他类型的层（如nn.Linear, 如果存在）添加初始化逻辑
        # raise NotImplementedError("Backbone::__init__")

    def forward(self, x):
        """
        Input shape: (B, 3, 640, 640)
        Output shape:
            feat0: (B, 128 * w, 160, 160)
            feat1: (B, 256 * w, 80, 80)
            feat2: (B, 512 * w, 40, 40)
            feat3: (B, 512 * w * r, 20, 20)
        """
        x_stem = self.stem(x)
        x_s1_cv = self.s1_conv(x_stem)
        feat0 = self.s1_c2f(x_s1_cv)
        x_s2_cv = self.s2_conv(feat0)
        feat1 = self.s2_c2f(x_s2_cv)
        x_s3_cv = self.s3_conv(feat1)
        feat2 = self.s3_c2f(x_s3_cv)
        x_s4_cv = self.s4_conv(feat2)
        x_s4_c2f = self.s4_c2f(x_s4_cv)
        feat3 = self.sppf(x_s4_c2f)

        return feat0, feat1, feat2, feat3
        # raise NotImplementedError("Backbone::forward")
