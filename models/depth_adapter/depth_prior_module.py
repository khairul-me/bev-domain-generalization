"""
depth_prior_module.py — Task 7.3 and 7.4

DepthPriorModule: Frozen Depth Anything V2 ViT-S encoder +
lightweight trainable adapter that projects depth features
into BEVFormer's feature space.

Architecture:
  1. Frozen DAv2 ViT-S encoder → intermediate features (no depth values)
  2. Trainable adapter (2-3 conv layers, ~1M params) → C-dim features
  3. Feature fusion: addition with BEVFormer backbone output

Intrinsics Normalization (Step 7.4):
  Before feeding to DAv2, optionally warp the image so it appears
  as if taken by a canonical camera (nuScenes CAM_FRONT on average).
  This makes DAv2 features consistent across domains.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

sys.path.insert(0, 'E:/Auto_Image/Depth-Anything-V2')

# ── Canonical camera (nuScenes CAM_FRONT average) ───────────────────────────
K_CANONICAL = np.array([
    [1266.417, 0.,       816.267],
    [0.,       1266.417, 491.507],
    [0.,       0.,       1.     ]
], dtype=np.float64)


def normalize_by_intrinsics(img_np, K_source, K_canonical=K_CANONICAL):
    """
    Warp img_np so it appears as if captured by K_canonical.
    
    Args:
        img_np: H×W×C numpy array (uint8 or float32)
        K_source: 3×3 source camera intrinsics matrix
        K_canonical: 3×3 canonical target intrinsics (default: nuScenes avg)
    
    Returns:
        warped: H×W×C numpy array, same dtype as input
    
    Why this works:
        The homography H = K_canonical @ inv(K_source) maps pixel coordinates
        in the source image to where they WOULD appear in the canonical camera.
        Applying this warp normalizes the image for DAv2, making its features
        camera-agnostic regardless of the source intrinsics.
    
    Note:
        This is one of the paper's novel contributions. It combines:
        (1) canonical camera normalization (removes intrinsic bias)
        (2) frozen foundation depth model (provides domain-agnostic features)
        → together, these give truly domain-generalizable depth features.
    """
    H_warp = K_canonical @ np.linalg.inv(K_source)
    h, w = img_np.shape[:2]
    warped = cv2.warpPerspective(img_np, H_warp, (w, h), 
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
    return warped


def normalize_by_intrinsics_batch(imgs_tensor, K_list, K_canonical=K_CANONICAL):
    """
    Batch version of normalize_by_intrinsics.
    
    Args:
        imgs_tensor: B×C×H×W tensor (normalized, float32)
        K_list: list of B 3×3 numpy arrays (one per image in the batch)
        K_canonical: canonical intrinsics matrix
    
    Returns:
        warped_tensor: B×C×H×W tensor, same device as input
    """
    device = imgs_tensor.device
    dtype = imgs_tensor.dtype
    B, C, H, W = imgs_tensor.shape
    
    warped_list = []
    imgs_np = imgs_tensor.detach().cpu().numpy()  # B×C×H×W
    
    for i in range(B):
        img = imgs_np[i].transpose(1, 2, 0)  # H×W×C
        # Denormalize if necessary (expecting [0,1] range for cv2)
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        
        K_src = K_list[i] if K_list is not None else K_canonical
        warped = normalize_by_intrinsics(img, K_src, K_canonical)
        
        # Re-normalize to float [0,1]
        warped = warped.astype(np.float32) / 255.0
        warped_list.append(torch.from_numpy(warped.transpose(2, 0, 1)))
    
    return torch.stack(warped_list, dim=0).to(device=device, dtype=dtype)


class DepthPriorModule(nn.Module):
    """
    Depth Prior Module for domain-generalizable depth features.
    
    Integrates frozen Depth Anything V2 ViT-S as a geometric prior
    with a lightweight trainable adapter that maps depth features
    into BEVFormer's feature space.
    
    Args:
        dav2_encoder: Frozen DAv2 ViT-S encoder (pretrained model)  
        in_channels: Number of channels from DAv2 encoder output
        adapter_channels: Number of channels in the adapter output (= BEVFormer img_neck channels)
        use_intrinsics_norm: Whether to apply intrinsics normalization before DAv2
        use_depth_prior: Flag to enable/disable (for ablation studies)
    
    Parameter count:
        DAv2 ViT-S encoder: ~24.8M (FROZEN - no gradient)
        Adapter: ~300K trainable params (configurable)
        Total trainable: ~300K
    """
    
    def __init__(
        self,
        dav2_model=None,
        in_channels=384,           # DAv2 ViT-S highest-resolution output channel
        adapter_channels=256,      # BEVFormer img feature channels (tiny uses 256)  
        use_intrinsics_norm=True,
        use_depth_prior=True,
        fusion_mode='add',          # 'add' or 'concat'
        dav2_checkpoint=None,       # Path to DAv2 checkpoint
    ):
        super().__init__()
        self.use_depth_prior = use_depth_prior
        self.use_intrinsics_norm = use_intrinsics_norm
        self.fusion_mode = fusion_mode
        
        if use_depth_prior:
            # Load DAv2 model
            if dav2_model is not None:
                self.dav2 = dav2_model
            else:
                self._load_dav2(dav2_checkpoint)
            
            # CRITICAL: freeze all DAv2 parameters
            # This is what gives us domain generalization — we don't corrupt
            # the foundation model's learned depth prior with task-specific gradients
            for name, param in self.dav2.named_parameters():
                param.requires_grad = False
            self.dav2.eval()  # Always in eval mode (no BN/Dropout updates)
            
            # Lightweight trainable adapter
            # Maps from DAv2 feature channels to BEVFormer feature channels
            # ~300K parameters total at default settings
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, adapter_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(adapter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(adapter_channels, adapter_channels, kernel_size=1),
                nn.BatchNorm2d(adapter_channels),
            )
            
            # Learnable scalar to scale depth features before fusion
            # Initialized to 0.1 to be conservative — grows during training
            self.depth_scale = nn.Parameter(torch.ones(1) * 0.1)
            
            # Count and report parameters
            adapter_params = sum(p.numel() for p in self.adapter.parameters())
            dav2_params = sum(p.numel() for p in self.dav2.parameters())
            print(f"[DepthPriorModule] DAv2 params (frozen): {dav2_params/1e6:.1f}M")
            print(f"[DepthPriorModule] Adapter params (trainable): {adapter_params/1e3:.0f}K")
    
    def _load_dav2(self, checkpoint_path):
        """Load Depth Anything V2 ViT-S model."""
        from depth_anything_v2.dpt import DepthAnythingV2
        
        model_cfg = {
            'encoder': 'vits', 
            'features': 64, 
            'out_channels': [48, 96, 192, 384]
        }
        self.dav2 = DepthAnythingV2(**model_cfg)
        
        if checkpoint_path is not None and checkpoint_path != '':
            import torch
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.dav2.load_state_dict(state_dict)
            print(f"[DepthPriorModule] Loaded DAv2 checkpoint: {checkpoint_path}")
        else:
            print("[DepthPriorModule] WARNING: No DAv2 checkpoint provided. Using random init.")
    
    def extract_dav2_features(self, img):
        """
        Extract intermediate features from frozen DAv2 encoder.
        
        Args:
            img: B×C×H×W tensor (normalized RGB, any size)
        
        Returns:
            feats: B×C×H'×W' spatial feature tensor
        
        Note:
            DAv2 uses ViT-S with patch size 14. Input size must be divisible by 14.
            We automatically resize to the nearest valid size and resize features back.
        """
        with torch.no_grad():
            B, C, H, W = img.shape
            
            # Round up to nearest multiple of 14 (ViT patch size)
            H14 = ((H + 13) // 14) * 14
            W14 = ((W + 13) // 14) * 14
            
            if H14 != H or W14 != W:
                img_resized = F.interpolate(img, size=(H14, W14), 
                                             mode='bilinear', align_corners=False)
            else:
                img_resized = img
            
            feats = self.dav2.pretrained(img_resized)
            
            # Handle various output formats from DINOv2
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            
            # If patch tokens (3D: B × num_patches × C) → convert to spatial
            if feats.dim() == 3:
                Hp, Wp = H14 // 14, W14 // 14
                feats = feats.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
        
        return feats
    
    def forward(self, img_feats_backbone, img_raw=None, K_list=None, img_shape=None):
        """
        Forward pass: extract depth features and fuse with backbone features.
        
        Args:
            img_feats_backbone: list of B×N_cam×C×H×W tensors from img_neck
            img_raw: B×N_cam×C×H×W raw input images (before normalization? after)
            K_list: list of camera intrinsics per image [N_cam × 3×3 numpy arrays]
            img_shape: (H, W) of input images
        
        Returns:
            fused_feats: same structure as img_feats_backbone but with depth prior
        """
        if not self.use_depth_prior or img_raw is None:
            return img_feats_backbone
        
        B, N_cam, C_raw, H_raw, W_raw = img_raw.shape
        
        # Flatten cameras: treat each camera image independently
        imgs_flat = img_raw.reshape(B * N_cam, C_raw, H_raw, W_raw)
        
        # Step 1: Intrinsics normalization (optional, for cross-domain)
        if self.use_intrinsics_norm and K_list is not None:
            # K_list: list of B lists, each containing N_cam 3×3 matrices
            K_flat = []
            for batch_K in K_list:
                K_flat.extend(batch_K)  # flatten to BN_cam matrices
            imgs_flat = normalize_by_intrinsics_batch(imgs_flat, K_flat)
        
        # Step 2: Extract frozen DAv2 features
        try:
            depth_feats = self.extract_dav2_features(imgs_flat)
            # Handle different output formats from DINOv2
            if isinstance(depth_feats, (list, tuple)):
                depth_feats = depth_feats[-1]  # Use last layer
            
            # depth_feats shape: B*N_cam × C_dav2 × H' × W'
            # If DINOv2 returns patches: reshape to spatial
            if depth_feats.dim() == 3:
                # B*N_cam × num_patches × C → to spatial: needs H/14, W/14
                Hp = H_raw // 14
                Wp = W_raw // 14
                depth_feats = depth_feats.permute(0, 2, 1).reshape(
                    B * N_cam, -1, Hp, Wp
                )
        except Exception as e:
            print(f"[DepthPriorModule] DAv2 feature extraction error: {e}")
            return img_feats_backbone
        
        # Step 3: Adapter — map depth features to BEVFormer feature space
        # Ensure depth_feats is 4D spatial: B*N_cam × C × H' × W'
        if depth_feats.dim() != 4:
            # Shouldn't happen after extract_dav2_features fix, but defensive
            print(f"[DepthPriorModule] Unexpected depth_feats dim: {depth_feats.dim()}, skipping")
            return img_feats_backbone
        
        # Resize to match backbone feature map size
        if len(img_feats_backbone) > 0:
            ref_feat = img_feats_backbone[0]
            target_h, target_w = ref_feat.shape[-2], ref_feat.shape[-1]
            if depth_feats.shape[-2:] != (target_h, target_w):
                depth_feats = F.interpolate(
                    depth_feats.float(), size=(target_h, target_w), 
                    mode='bilinear', align_corners=False
                )

        
        adapted_feats = self.adapter(depth_feats.float())  # B*N_cam × C_adapter × H' × W'
        
        # Step 4: Fusion — add scaled depth features to backbone features
        fused_feats = []
        for scale_feat in img_feats_backbone:
            # scale_feat: B × N_cam × C × H' × W'
            BN, C, Hf, Wf = scale_feat.shape[0] * scale_feat.shape[1], \
                             scale_feat.shape[2], scale_feat.shape[-2], scale_feat.shape[-1]
            
            feat_flat = scale_feat.reshape(B * N_cam, C, Hf, Wf)
            
            # Resize adapted_feats to match this scale
            if adapted_feats.shape[-2:] != (Hf, Wf):
                depth_at_scale = F.interpolate(
                    adapted_feats, size=(Hf, Wf), mode='bilinear', align_corners=False
                )
            else:
                depth_at_scale = adapted_feats
            
            if self.fusion_mode == 'add':
                fused = feat_flat + self.depth_scale * depth_at_scale
            else:  # concat is handled differently (requires different adapter_channels)
                fused = torch.cat([feat_flat, depth_at_scale], dim=1)
            
            # Reshape back to B × N_cam × C × H × W
            fused = fused.reshape(scale_feat.shape)
            fused_feats.append(fused)
        
        return fused_feats
    
    def assert_frozen(self):
        """Sanity check: verify DAv2 parameters are truly frozen."""
        for name, param in self.dav2.named_parameters():
            assert not param.requires_grad, f"DAv2 parameter {name} is NOT frozen!"
        print("[DepthPriorModule] ✓ All DAv2 parameters confirmed frozen.")


# ── Test the module in isolation ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing DepthPriorModule in isolation...")
    print("(Requires DAv2 checkpoint at checkpoints/depth_anything_v2_vits.pth)")
    
    import os
    ckpt = "E:/bev_research/checkpoints/depth_anything_v2_vits.pth"
    
    # Test parameter counting without checkpoint
    module = DepthPriorModule(
        dav2_checkpoint=None,
        in_channels=384,
        adapter_channels=256,
        use_depth_prior=True,
        use_intrinsics_norm=True,
    )
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    print(f"\nParameter Summary:")
    print(f"  Total params: {total/1e6:.2f}M")
    print(f"  Trainable (adapter only): {trainable/1e3:.0f}K")
    print(f"  Frozen (DAv2): {(total-trainable)/1e6:.2f}M")
    
    # Verify freezing
    module.assert_frozen()
    print("\n[OK] DepthPriorModule tests passed.")
