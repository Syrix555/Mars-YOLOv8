import torch
import torch.nn as nn
import timm
from timm.models.swin_transformer import SwinTransformer # 直接导入SwinTransformer类以获取更多控制权
# 假设你的 components.py 中有 SPPF
# from .components import Conv, C2f, SPPF # Conv 和 C2f 将不再直接使用
from .components import SPPF # 只需要 SPPF

class SwinTransformer(nn.Module):
    def __init__(self, w, r, n, # YOLOv8 scaling factors, n might be less relevant
                 img_size=640,     # Input image size
                 in_chans=3,       # Input image channels
                 swin_model_name='swin_tiny_patch4_window7_224', # Base Swin model
                 pretrained=True): # Whether to load pretrained weights
        super().__init__()

        # --- Swin Transformer Configuration ---
        # 你可以根据 w 选择不同的 swin_model_name 或调整 embed_dim
        # 例如:
        if w <= 0.25: # Example: YOLOv8n -> Smallest Swin
            # Note: 'swin_tiny_patch4_window7_224' is trained on 224x224.
            # For 640x640 input, window size and patch size need to be appropriate.
            # Timm handles mismatched input sizes by adjusting positional embeddings if pretrained.
            # We might need to adjust `embed_dim` based on `w` if not using predefined models.
            self.swin_model_name = 'swin_tiny_patch4_window7_224'
            base_embed_dim = 96 # Swin-T
            depths = [2, 2, 6, 2] # Swin-T depths
            num_heads = [3, 6, 12, 24] # Swin-T num_heads
        elif w <= 0.5: # Example: YOLOv8s -> Swin-T or slightly larger
            self.swin_model_name = 'swin_small_patch4_window7_224' # Example
            base_embed_dim = 96 # Swin-S
            depths = [2, 2, 18, 2] # Swin-S depths
            num_heads = [3, 6, 12, 24] # Swin-S num_heads
        else: # Example: YOLOv8m/l/x -> Swin-B or custom
            self.swin_model_name = 'swin_base_patch4_window7_224'
            base_embed_dim = 128 # Swin-B
            depths = [2, 2, 18, 2] # Swin-B depths
            num_heads = [4, 8, 16, 32] # Swin-B num_heads

        # For simplicity, let's pick one configuration and allow `w` to scale its initial embed_dim.
        # This is a more direct way to use 'w' than just picking model names.
        # scaled_embed_dim = max(32, int(base_embed_dim * w)) # Ensure a minimum embed_dim
        # Ensure embed_dim is divisible by num_heads for each stage.
        # This might require more complex logic or fixing num_heads based on scaled_embed_dim.
        # For this example, we'll use the default num_heads for swin_tiny and adjust embed_dim
        # if it makes sense. A common approach is to pick a pre-defined Swin variant.

        # Let's use a fixed Swin config for clarity and load it.
        # We want features from multiple stages.
        # Swin typically has 4 stages.
        # Stage 0 output (after patch_embed + first layer): stride 4
        # Stage 1 output: stride 8
        # Stage 2 output: stride 16
        # Stage 3 output: stride 32

        # We will use timm's SwinTransformer directly to get intermediate features.
        # Note: The `features_only=True` argument is helpful, but accessing intermediate
        # features directly from the `stages` attribute provides more control.

        self.out_indices = (0, 1, 2, 3) # Swin有4个stage

        self.swin = timm.create_model(
            self.swin_model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True, # We will manually pick features from stages
            img_size=img_size,
            out_indices=self.out_indices # 指定需要输出哪些stage的特征
            # We can override parameters if needed, e.g., for custom embed_dim not matching a predefined model:
            # embed_dim=scaled_embed_dim,
            # depths=depths,
            # num_heads=num_heads,
            # window_size=7, # Ensure this matches the model name or set appropriately
            # patch_size=4
        )

        # --- Determine output channels from Swin stages ---
        # This depends on the chosen Swin model (e.g., Swin-T: 96, 192, 384, 768)
        # For 'swin_tiny_patch4_window7_224':
        # self.swin.num_features returns the channels of the *final* stage output (before head)
        # We need the channel counts for each of the 4 stages.
        # patch_embed output channels: self.swin.embed_dim
        # Stage 0 channels: self.swin.embed_dim
        # Stage 1 channels: self.swin.embed_dim * 2
        # Stage 2 channels: self.swin.embed_dim * 4
        # Stage 3 channels: self.swin.embed_dim * 8

        feature_info_dicts = self.swin.feature_info.get_dicts()
        ch_swin_s0 = feature_info_dicts[0]['num_chs'] # Swin stage 0 output (stride 4)
        ch_swin_s1 = feature_info_dicts[1]['num_chs'] # Swin stage 1 output (stride 8)
        ch_swin_s2 = feature_info_dicts[2]['num_chs'] # Swin stage 2 output (stride 16)
        ch_swin_s3 = feature_info_dicts[3]['num_chs'] # Swin stage 3 output (stride 32)

        # --- Define target channel sizes based on original YOLOv8 scaling ---
        # These are the channel sizes the Neck will expect.
        # Auxiliary function from your original code
        def scale_channels(base_ch, width_mult, ratio_mult=1.0):
            return max(1, int(base_ch * width_mult * ratio_mult))

        # Target channels for feat0, feat1, feat2, feat3
        # Original feat0 was after s1_c2f (stride 8 from input, but your comment says stride 4 for output map)
        # Let's assume:
        # feat0: output of Swin Stage 0 (total stride 4)
        # feat1: output of Swin Stage 1 (total stride 8)
        # feat2: output of Swin Stage 2 (total stride 16)
        # feat3: output of Swin Stage 3 (total stride 32)

        # Original YOLOv8 stem out was 64*w. s1_conv out (feat0_in) was 128*w.
        # Let's try to match the *output* channels of your original stages.
        # feat0: (B, 128 * w, H/4, W/4) - Corresponds to Swin Stage 0 (stride 4)
        # feat1: (B, 256 * w, H/8, W/8) - Corresponds to Swin Stage 1 (stride 8)
        # feat2: (B, 512 * w, H/16, W/16) - Corresponds to Swin Stage 2 (stride 16)
        # feat3: (B, 512 * w * r, H/32, W/32) - Corresponds to Swin Stage 3 (stride 32)

        self.ch_feat0_out = scale_channels(128, w)
        self.ch_feat1_out = scale_channels(256, w)
        self.ch_feat2_out = scale_channels(512, w)
        self.ch_feat3_out = scale_channels(512, w, r)

        # --- Projection layers to match channel dimensions ---
        # These will take the output of each Swin stage and project it to the required channel dim.
        self.proj_feat0 = nn.Conv2d(ch_swin_s0, self.ch_feat0_out, kernel_size=1)
        self.proj_feat1 = nn.Conv2d(ch_swin_s1, self.ch_feat1_out, kernel_size=1)
        self.proj_feat2 = nn.Conv2d(ch_swin_s2, self.ch_feat2_out, kernel_size=1)
        self.proj_feat3_pre_sppf = nn.Conv2d(ch_swin_s3, self.ch_feat3_out, kernel_size=1)

        # SPPF layer (applied on the last projected feature map)
        self.sppf = SPPF(self.ch_feat3_out, self.ch_feat3_out, k=5) # k=5 is common


    def forward(self, x):
            # Input x: (B, 3, H, W), e.g., (B, 3, 640, 640)
            # 当 self.swin 是通过 timm.create_model(..., features_only=True, out_indices=...) 创建时，
            # 它会直接返回一个包含指定层级输出的列表。

            # 1. 通过 Swin Transformer 特征提取器获取多尺度特征
            # swin_outputs 将是一个列表，例如 [stage0_output, stage1_output, stage2_output, stage3_output]
            # 每个元素的形状类似于 (B, C_stage, H_stage, W_stage)
            swin_outputs = self.swin(x)

            # 2. 将提取到的特征传递给各自的投影层，以匹配期望的通道数
            # 假设 self.out_indices = (0, 1, 2, 3) 对应 Swin 的四个主要输出尺度
            # 并且 swin_outputs 列表中的元素顺序与 out_indices 一致

            # feat0 对应 Swin 的第一个输出尺度 (例如，总步长 4)
            feat0_from_swin = swin_outputs[0].permute(0, 3, 1, 2)
            feat0 = self.proj_feat0(feat0_from_swin)

            # feat1 对应 Swin 的第二个输出尺度 (例如，总步长 8)
            feat1_from_swin = swin_outputs[1].permute(0, 3, 1, 2)
            feat1 = self.proj_feat1(feat1_from_swin)

            # feat2 对应 Swin 的第三个输出尺度 (例如，总步长 16)
            feat2_from_swin = swin_outputs[2].permute(0, 3, 1, 2)
            feat2 = self.proj_feat2(feat2_from_swin)

            # feat3 对应 Swin 的第四个输出尺度 (例如，总步长 32)
            feat3_from_swin = swin_outputs[3].permute(0, 3, 1, 2)
            feat3_pre_sppf = self.proj_feat3_pre_sppf(feat3_from_swin)

            # 3. 对最后一个特征图应用 SPPF
            feat3 = self.sppf(feat3_pre_sppf)

            # 预期的输出形状 (与你之前代码中的注释一致)
            # feat0: (B, 128 * w, H/4, W/4)
            # feat1: (B, 256 * w, H/8, W/8)
            # feat2: (B, 512 * w, H/16, W/16)
            # feat3: (B, 512 * w * r, H/32, W/32)

            # (可选) 调试时可以取消注释以验证形状
            # print(f"Input shape: {x.shape}")
            # print(f"swin_outputs[0] shape: {swin_outputs[0].shape} -> feat0 shape: {feat0.shape}")
            # print(f"swin_outputs[1] shape: {swin_outputs[1].shape} -> feat1 shape: {feat1.shape}")
            # print(f"swin_outputs[2] shape: {swin_outputs[2].shape} -> feat2 shape: {feat2.shape}")
            # print(f"swin_outputs[3] shape: {swin_outputs[3].shape} -> feat3_pre_sppf shape: {feat3_pre_sppf.shape} -> feat3 shape: {feat3.shape}")

            return feat0, feat1, feat2, feat3
