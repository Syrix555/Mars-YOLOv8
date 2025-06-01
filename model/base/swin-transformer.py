import torch
import torch.nn as nn
import timm
from timm.models.swin_transformer import SwinTransformer # 直接导入SwinTransformer类以获取更多控制权
# 假设你的 components.py 中有 SPPF
# from .components import Conv, C2f, SPPF # Conv 和 C2f 将不再直接使用
from .components import SPPF # 只需要 SPPF

class SwinTransformerBackbone(nn.Module):
    def __init__(self, w, r, n, # YOLOv8 scaling factors, n might be less relevant
                 img_size=640,     # Input image size
                 in_chans=3,       # Input image channels
                 swin_model_name='swin_tiny', # Base Swin model
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
            self.swin_model_name = 'swin_tiny'
            base_embed_dim = 96 # Swin-T
            depths = [2, 2, 6, 2] # Swin-T depths
            num_heads = [3, 6, 12, 24] # Swin-T num_heads
        elif w <= 0.5: # Example: YOLOv8s -> Swin-T or slightly larger
            self.swin_model_name = 'swin_small' # Example
            base_embed_dim = 96 # Swin-S
            depths = [2, 2, 18, 2] # Swin-S depths
            num_heads = [3, 6, 12, 24] # Swin-S num_heads
        else: # Example: YOLOv8m/l/x -> Swin-B or custom
            self.swin_model_name = 'swin_base'
            base_embed_dim = 128 # Swin-B
            depths = [2, 2, 18, 2] # Swin-B depths
            num_heads = [4, 8, 16, 32] # Swin-B num_heads

        # For simplicity, let's pick one configuration and allow `w` to scale its initial embed_dim.
        # This is a more direct way to use 'w' than just picking model names.
        scaled_embed_dim = max(32, int(base_embed_dim * w)) # Ensure a minimum embed_dim
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

        self.swin = timm.create_model(
            self.swin_model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=False, # We will manually pick features from stages
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

        swin_embed_dim = self.swin.embed_dim # e.g., 96 for Swin-T
        ch_swin_s0 = swin_embed_dim
        ch_swin_s1 = swin_embed_dim * 2
        ch_swin_s2 = swin_embed_dim * 4
        ch_swin_s3 = swin_embed_dim * 8

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

        # Weight initialization
        self._initialize_weights()


    def _initialize_weights(self):
        # Initialize projection layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): # Should not be present in projection if not added
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): # Swin has Linear layers, but timm handles their init
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Swin Transformer weights are typically initialized within the timm model loading (especially if pretrained)
        # or by its own internal _init_weights method.


    def forward(self, x):
        # Input x: (B, 3, H, W), e.g., (B, 3, 640, 640)

        # --- Pass through Swin Transformer ---
        # SwinTransformer's forward_features method is what we need.
        # It goes through patch_embed and all stages.
        # The internal structure of timm's SwinTransformer:
        # x = self.swin.patch_embed(x) -> (B, H/4 * W/4, embed_dim)
        # x = self.swin.pos_drop(x)
        # features = []
        # for stage in self.swin.stages:
        #     x = stage(x)
        #     features.append(x.permute(0, 2, 1).reshape(B, C, H', W')) # Reshape to (B,C,H,W)

        B, _, H_in, W_in = x.shape

        # 1. Patch Embedding
        x_patch_embed = self.swin.patch_embed(x) # (B, num_patches, embed_dim)
        # num_patches = (H_in/patch_size) * (W_in/patch_size)
        # H0, W0 = H_in // self.swin.patch_embed.patch_size[0], W_in // self.swin.patch_embed.patch_size[1]
        if self.swin.absolute_pos_embed is not None:
             x_patch_embed = x_patch_embed + self.swin.absolute_pos_embed
        x_patch_embed = self.swin.pos_drop(x_patch_embed)

        # Store features from each stage
        swin_features = []

        # Stage 0
        s0_out = self.swin.stages[0](x_patch_embed) # (B, H/4*W/4, C0) where C0 = embed_dim
        H0, W0 = H_in // 4, W_in // 4
        s0_out_reshaped = s0_out.permute(0, 2, 1).reshape(B, self.swin.stages[0].dim, H0, W0)
        swin_features.append(s0_out_reshaped)

        # Stage 1
        s1_out = self.swin.stages[1](s0_out) # (B, H/8*W/8, C1) where C1 = embed_dim*2
        H1, W1 = H_in // 8, W_in // 8
        s1_out_reshaped = s1_out.permute(0, 2, 1).reshape(B, self.swin.stages[1].dim, H1, W1)
        swin_features.append(s1_out_reshaped)

        # Stage 2
        s2_out = self.swin.stages[2](s1_out) # (B, H/16*W/16, C2) where C2 = embed_dim*4
        H2, W2 = H_in // 16, W_in // 16
        s2_out_reshaped = s2_out.permute(0, 2, 1).reshape(B, self.swin.stages[2].dim, H2, W2)
        swin_features.append(s2_out_reshaped)

        # Stage 3
        s3_out = self.swin.stages[3](s2_out) # (B, H/32*W/32, C3) where C3 = embed_dim*8
        H3, W3 = H_in // 32, W_in // 32
        s3_out_reshaped = s3_out.permute(0, 2, 1).reshape(B, self.swin.stages[3].dim, H3, W3)
        swin_features.append(s3_out_reshaped)

        # --- Project features to target channel sizes ---
        feat0 = self.proj_feat0(swin_features[0]) # (B, ch_feat0_out, H/4, W/4)
        feat1 = self.proj_feat1(swin_features[1]) # (B, ch_feat1_out, H/8, W/8)
        feat2 = self.proj_feat2(swin_features[2]) # (B, ch_feat2_out, H/16, W/16)
        feat3_pre_sppf = self.proj_feat3_pre_sppf(swin_features[3]) # (B, ch_feat3_out, H/32, W/32)

        # Apply SPPF
        feat3 = self.sppf(feat3_pre_sppf) # (B, ch_feat3_out, H/32, W/32)

        # Expected output shapes:
        # feat0: (B, 128 * w, H/4, W/4)  e.g., (B, 64/128/256, 160, 160) for w=0.5/1.0/2.0 (if base=128)
        # feat1: (B, 256 * w, H/8, W/8)  e.g., (B, 128/256/512, 80, 80)
        # feat2: (B, 512 * w, H/16, W/16) e.g., (B, 256/512/1024, 40, 40)
        # feat3: (B, 512 * w * r, H/32, W/32) e.g., (B, 256/512/1024 * r, 20, 20)

        # Verify shapes (optional, for debugging)
        # print(f"feat0: {feat0.shape}")
        # print(f"feat1: {feat1.shape}")
        # print(f"feat2: {feat2.shape}")
        # print(f"feat3: {feat3.shape}")

        return feat0, feat1, feat2, feat3
