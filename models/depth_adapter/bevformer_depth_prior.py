"""
bevformer_depth_prior.py — Task 7.5

BEVFormerWithDepthPrior: BEVFormer-Tiny-fp16 extended with frozen
Depth Anything V2 ViT-S as a domain-agnostic geometric prior.

Key modification:
  In extract_img_feat(), after the backbone extracts image features,
  the DepthPriorModule injects depth-aware features by:
    1. (Optional) normalizing images by intrinsics
    2. Running frozen DAv2-ViTS to get depth features  
    3. Adapting with trainable conv layers
    4. Adding scaled features to backbone output

This is a minimal, non-invasive modification — BEVFormer's
SCA, TSA, and detection head are UNCHANGED.
"""
import sys
import torch
import torch.nn as nn
import copy

sys.path.insert(0, 'E:/Auto_Image/BEVFormer')
sys.path.insert(0, 'E:/Auto_Image/BEVFormer/projects')
sys.path.insert(0, 'E:/bev_research/models/depth_adapter')

try:
    from mmcv.runner import force_fp32, auto_fp16
    from mmdet.models import DETECTORS
    from mmdet3d.core import bbox3d2result
    from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
    from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
    mmd_available = True
except ImportError:
    mmd_available = False
    print("[BEVFormerWithDepthPrior] MMDetection3D not importable in this context.")

from depth_prior_module import DepthPriorModule


if mmd_available:
    @DETECTORS.register_module()
    class BEVFormerWithDepthPrior(MVXTwoStageDetector):
        """
        BEVFormer extended with frozen Depth Anything V2 depth prior.
        
        All BEVFormer components are unchanged. The only addition is:
        - DepthPriorModule injected after img_backbone + img_neck
        - Controlled by `use_depth_prior` flag for clean ablations
        
        Training strategy:
        - FROZEN: DAv2 encoder, ResNet-50 backbone (img_backbone)
        - TRAINABLE: depth adapter (~300K params), BEV encoder attention, detection head
        """
        
        def __init__(self,
                     use_grid_mask=False,
                     pts_voxel_layer=None,
                     pts_voxel_encoder=None,
                     pts_middle_encoder=None,
                     pts_fusion_layer=None,
                     img_backbone=None,
                     pts_backbone=None,
                     img_neck=None,
                     pts_neck=None,
                     pts_bbox_head=None,
                     img_roi_head=None,
                     img_rpn_head=None,
                     train_cfg=None,
                     test_cfg=None,
                     pretrained=None,
                     video_test_mode=False,
                     # ── New args for Depth Prior ──────────────────────────
                     use_depth_prior=True,
                     dav2_checkpoint=None,
                     depth_adapter_channels=256,
                     use_intrinsics_norm=True,
                     depth_fusion_mode='add',
                     ):
            super(BEVFormerWithDepthPrior, self).__init__(
                pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder, pts_fusion_layer,
                img_backbone, pts_backbone, img_neck, pts_neck, pts_bbox_head,
                img_roi_head, img_rpn_head, train_cfg, test_cfg, pretrained)
            
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
            self.use_grid_mask = use_grid_mask
            self.fp16_enabled = False
            self.video_test_mode = video_test_mode
            self.prev_frame_info = {
                'prev_bev': None, 'scene_token': None,
                'prev_pos': 0, 'prev_angle': 0,
            }
            
            # ── Depth Prior Module ────────────────────────────────────────
            self.depth_prior = DepthPriorModule(
                dav2_checkpoint=dav2_checkpoint,
                in_channels=384,               # DAv2 ViT-S highest channel
                adapter_channels=depth_adapter_channels,
                use_intrinsics_norm=use_intrinsics_norm,
                use_depth_prior=use_depth_prior,
                fusion_mode=depth_fusion_mode,
            )
            
            # Freeze img_backbone (ResNet-50) — preserve nuScenes features
            if img_backbone is not None:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
                print("[BEVFormerWithDepthPrior] img_backbone frozen.")
            
            self._report_trainable_params()
        
        def _report_trainable_params(self):
            """Report trainable parameter statistics."""
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"[BEVFormerWithDepthPrior] Total params: {total/1e6:.1f}M")
            print(f"[BEVFormerWithDepthPrior] Trainable params: {trainable/1e6:.2f}M")
            print(f"[BEVFormerWithDepthPrior] Frozen params: {(total-trainable)/1e6:.1f}M")
        
        def extract_img_feat(self, img, img_metas, len_queue=None):
            """
            Modified extract_img_feat: after backbone, inject depth prior.
            
            Injection point: AFTER img_neck (FPN output), BEFORE BEV encoder.
            This is the cleanest injection point — features are at the right
            spatial resolution and channel dimension for the SCA to consume.
            """
            B = img.size(0)
            if img is None:
                return None
            
            # ── Standard BEVFormer backbone ──────────────────────────────
            img_raw = img.clone()  # Keep raw image for DAv2 (before grid mask)
            
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
                img_raw = img_raw.reshape(B * N, C, H, W)
            
            if self.use_grid_mask:
                img = self.grid_mask(img)
            
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
            
            if self.with_img_neck:
                img_feats = self.img_neck(img_feats)
            
            # ── Depth Prior Injection ──────────────────────────────────
            # Extract camera intrinsics from img_metas if available
            K_list = None
            if img_metas is not None and self.depth_prior.use_intrinsics_norm:
                try:
                    K_list = []
                    for meta in img_metas:
                        if isinstance(meta, list):
                            meta = meta[0]
                        if 'cam_intrinsic' in meta:
                            K_list.append(meta['cam_intrinsic'])
                        elif 'lidar2img' in meta:
                            # Fallback: extract K from lidar2img
                            K_list.append(None)
                        else:
                            K_list = None
                            break
                except Exception:
                    K_list = None
            
            # Reshape img_feats for DepthPriorModule
            img_feats_reshaped = []
            for img_feat in img_feats:
                BN, C_feat, H_feat, W_feat = img_feat.size()
                N_cam = BN // B
                img_feats_reshaped.append(img_feat.view(B, N_cam, C_feat, H_feat, W_feat))
            
            img_raw_batch = img_raw.view(B, N_cam, C, H, W) if img_raw.dim() == 2 else \
                            img_raw.reshape(B, -1, img_raw.shape[-3], img_raw.shape[-2], img_raw.shape[-1])
            
            # Apply depth prior
            img_feats_with_depth = self.depth_prior(
                img_feats_reshaped, img_raw=img_raw_batch, K_list=K_list
            )
            
            # ── Reshape back to BEVFormer expected format ─────────────
            final_feats = []
            for feat in img_feats_with_depth:
                B_, N_cam_, C_, H_, W_ = feat.shape
                if len_queue is not None:
                    final_feats.append(feat.view(B_ // len_queue, len_queue, N_cam_, C_, H_, W_))
                else:
                    final_feats.append(feat.view(B_, N_cam_, C_, H_, W_))
            
            return final_feats
        
        # ── All other methods are IDENTICAL to BEVFormer ─────────────────
        # (inherit forward_train, forward_test, simple_test, etc.)
        
        @auto_fp16(apply_to=('img',))
        def extract_feat(self, img, img_metas=None, len_queue=None):
            """Extract features from images."""
            return self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        def forward_pts_train(self, pts_feats, gt_bboxes_3d, gt_labels_3d,
                               img_metas, gt_bboxes_ignore=None, prev_bev=None):
            outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
            return losses
        
        def forward(self, return_loss=True, **kwargs):
            if return_loss:
                return self.forward_train(**kwargs)
            else:
                return self.forward_test(**kwargs)
        
        def obtain_history_bev(self, imgs_queue, img_metas_list):
            self.eval()
            with torch.no_grad():
                prev_bev = None
                bs, len_queue, num_cams, C, H, W = imgs_queue.shape
                imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
                img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
                for i in range(len_queue):
                    img_metas = [each[i] for each in img_metas_list]
                    if not img_metas[0]['prev_bev_exists']:
                        prev_bev = None
                    img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                    prev_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev, only_bev=True)
                self.train()
                return prev_bev
        
        @auto_fp16(apply_to=('img', 'points'))
        def forward_train(self, points=None, img_metas=None, gt_bboxes_3d=None,
                           gt_labels_3d=None, gt_labels=None, gt_bboxes=None,
                           img=None, proposals=None, gt_bboxes_ignore=None,
                           img_depth=None, img_mask=None):
            len_queue = img.size(1)
            prev_img = img[:, :-1, ...]
            img = img[:, -1, ...]
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
            img_metas = [each[len_queue - 1] for each in img_metas]
            if not img_metas[0]['prev_bev_exists']:
                prev_bev = None
            img_feats = self.extract_feat(img=img, img_metas=img_metas)
            losses = dict()
            losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d,
                                                img_metas, gt_bboxes_ignore, prev_bev)
            losses.update(losses_pts)
            return losses
        
        def forward_test(self, img_metas, img=None, **kwargs):
            for var, name in [(img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError(f'{name} must be a list, but got {type(var)}')
            img = [img] if img is None else img
            if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
                self.prev_frame_info['prev_bev'] = None
            self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']
            if not self.video_test_mode:
                self.prev_frame_info['prev_bev'] = None
            tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
            tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
            if self.prev_frame_info['prev_bev'] is not None:
                img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
                img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
            else:
                img_metas[0][0]['can_bus'][-1] = 0
                img_metas[0][0]['can_bus'][:3] = 0
            new_prev_bev, bbox_results = self.simple_test(
                img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
            self.prev_frame_info['prev_pos'] = tmp_pos
            self.prev_frame_info['prev_angle'] = tmp_angle
            self.prev_frame_info['prev_bev'] = new_prev_bev
            return bbox_results
        
        def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
            outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)
            bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
            bbox_results = [bbox3d2result(bboxes, scores, labels)
                           for bboxes, scores, labels in bbox_list]
            return outs['bev_embed'], bbox_results
        
        def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
            img_feats = self.extract_feat(img=img, img_metas=img_metas)
            bbox_list = [dict() for _ in range(len(img_metas))]
            new_prev_bev, bbox_pts = self.simple_test_pts(img_feats, img_metas, prev_bev, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
            return new_prev_bev, bbox_list


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("BEVFormerWithDepthPrior module defined.")
    if not mmd_available:
        print("MMDetection3D not available — class defined but not registered.")
    else:
        print("Registered in DETECTORS registry as 'BEVFormerWithDepthPrior'.")
