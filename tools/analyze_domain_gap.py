"""
analyze_domain_gap.py — Task 6 prerequisite (can run with or without full datasets).
Quantifies the intrinsic parameter gap between nuScenes and KITTI.
Generates a domain gap analysis figure saved to experiments/baseline/domain_gap_analysis.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os

# ─── Camera Intrinsics ───────────────────────────────────────────────────────
# nuScenes (CAM_FRONT average from 850 calibrated sensors)
K_nuscenes = np.array([
    [1266.417, 0.,       816.267],
    [0.,       1266.417, 491.507],
    [0.,       0.,       1.     ]
])
IMG_SIZE_NUSCENES = (1600, 900)

# KITTI (from calib/000000.txt, P2 matrix)
K_kitti = np.array([
    [721.537, 0.,      609.559],
    [0.,      721.537, 172.851],
    [0.,      0.,      1.     ]
])
IMG_SIZE_KITTI = (1242, 375)

# ─── Simulation: Project 3D objects at various depths ────────────────────────
def project_points(K, points_3d):
    """Project Nx3 3D points to 2D using intrinsics K."""
    pts_h = (K @ points_3d.T).T
    pts_2d = pts_h[:, :2] / pts_h[:, 2:3]
    return pts_2d

def simulate_depth_distributions(K, img_size, label=""):
    """Simulate how objects at different depths appear in image coordinates."""
    W, H = img_size
    # Common object depths in autonomous driving (cars at 2m to 80m)
    depths = np.linspace(2, 80, 200)
    # Object at lane center, ego vehicle height
    car_widths   = 1.8  # typical car width in meters
    car_heights  = 1.5  # typical car height
    
    # For each depth, compute projected bounding box height in pixels
    projected_heights = []
    projected_widths = []
    for d in depths:
        # Top and bottom of a car at depth d
        top_3d    = np.array([[0, -car_heights/2, d]])
        bottom_3d = np.array([[0,  car_heights/2, d]])
        top_2d    = project_points(K, top_3d)
        bottom_2d = project_points(K, bottom_3d)
        ph = abs(bottom_2d[0,1] - top_2d[0,1])
        projected_heights.append(ph)
        
        # Left/right edges
        left_3d  = np.array([[-car_widths/2, 0, d]])
        right_3d = np.array([[ car_widths/2, 0, d]])
        left_2d  = project_points(K, left_3d)
        right_2d = project_points(K, right_3d)
        pw = abs(right_2d[0,0] - left_2d[0,0])
        projected_widths.append(pw)
    
    return depths, np.array(projected_heights), np.array(projected_widths)

# ─── Figure 1: Projected object size vs depth ────────────────────────────────
depths_ns, h_ns, w_ns = simulate_depth_distributions(K_nuscenes, IMG_SIZE_NUSCENES, "nuScenes")
depths_kt, h_kt, w_kt = simulate_depth_distributions(K_kitti, IMG_SIZE_KITTI, "KITTI")

# ─── Figure 2: Where a 3D point grid projects in each camera ──────────────
def compute_projection_map(K, img_size, grid_depths=[5, 10, 20, 40]):
    W, H = img_size
    results = {}
    for d in grid_depths:
        # Grid of 3D points at this depth
        x_vals = np.linspace(-15, 15, 5)
        y_vals = np.array([-0.5, 0, 0.5])
        pts = []
        for x in x_vals:
            for y in y_vals:
                pts.append([x, y, d])
        pts = np.array(pts)
        proj = project_points(K, pts)
        # Filter to image bounds
        valid = (proj[:,0] >= 0) & (proj[:,0] < W) & (proj[:,1] >= 0) & (proj[:,1] < H)
        results[d] = (proj[valid, 0] / W, proj[valid, 1] / H)  # normalized coords
    return results

# ─── BEV Reference Point Mismatch Analysis ───────────────────────────────────
def bev_reference_point_error(K_source, K_target, bev_range=50, num_points=20):
    """
    Compute how much BEV reference points shift when intrinsics change.
    This is the core mechanism by which cross-domain failure occurs.
    """
    # Sample 3D points in BEV space
    x_vals = np.linspace(-bev_range/2, bev_range/2, num_points)
    z_vals = np.linspace(1, bev_range, num_points)
    errors = []
    for x in x_vals:
        for z in z_vals:
            pt_3d = np.array([[x, 0, z]])
            proj_src = project_points(K_source, pt_3d)[0]
            proj_tgt = project_points(K_target, pt_3d)[0]
            err = np.linalg.norm(proj_src - proj_tgt)
            errors.append({'x': x, 'z': z, 'err': err})
    return errors

errors = bev_reference_point_error(K_nuscenes, K_kitti)
err_vals = np.array([e['err'] for e in errors])
z_vals_err = np.array([e['z'] for e in errors])

# ─── Create publication-quality figure ───────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#1a1a2e')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

colors = {'nuScenes': '#4ecdc4', 'KITTI': '#ff6b6b'}

# Panel 1: Projected object height vs depth
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(depths_ns, h_ns, color=colors['nuScenes'], lw=2.5, label=f'nuScenes (fx={K_nuscenes[0,0]:.0f})')
ax1.plot(depths_kt, h_kt, color=colors['KITTI'],    lw=2.5, label=f'KITTI (fx={K_kitti[0,0]:.0f})')
ax1.set_xlabel('Depth (m)', color='white', fontsize=11)
ax1.set_ylabel('Projected Height (px)', color='white', fontsize=11)
ax1.set_title('Car Height Projection vs Depth', color='white', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, facecolor='#2a2a3e', labelcolor='white')
ax1.set_facecolor('#0f0f23')
ax1.tick_params(colors='white')
ax1.spines[:].set_color('#444')

# Panel 2: Projection ratio
ax2 = fig.add_subplot(gs[0, 1])
ratio = h_ns / h_kt
ax2.plot(depths_ns, ratio, color='#ffd93d', lw=2.5)
ax2.axhline(y=K_nuscenes[0,0]/K_kitti[0,0], color='white', linestyle='--', alpha=0.5, 
            label=f'fx ratio = {K_nuscenes[0,0]/K_kitti[0,0]:.2f}×')
ax2.set_xlabel('Depth (m)', color='white', fontsize=11)
ax2.set_ylabel('Height Ratio (nuScenes/KITTI)', color='white', fontsize=11)
ax2.set_title('Projection Scale Ratio', color='white', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, facecolor='#2a2a3e', labelcolor='white')
ax2.set_facecolor('#0f0f23')
ax2.tick_params(colors='white')
ax2.spines[:].set_color('#444')

# Panel 3: BEV reference point error
ax3 = fig.add_subplot(gs[0, 2])
scatter = ax3.scatter([e['z'] for e in errors], [e['x'] for e in errors], 
                       c=[e['err'] for e in errors], cmap='plasma', s=15, vmin=0, vmax=300)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Pixel Error', color='white', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
ax3.set_xlabel('Depth Z (m)', color='white', fontsize=11)
ax3.set_ylabel('X (m)', color='white', fontsize=11)
ax3.set_title('BEV Ref Point Error (nuScenes→KITTI)\n(px error when cross-domain)', 
               color='white', fontsize=11, fontweight='bold')
ax3.set_facecolor('#0f0f23')
ax3.tick_params(colors='white')
ax3.spines[:].set_color('#444')

# Panel 4: Intrinsics bar chart
ax4 = fig.add_subplot(gs[1, 0])
props = ['fx', 'fy', 'cx', 'cy']
vals_ns = [K_nuscenes[0,0], K_nuscenes[1,1], K_nuscenes[0,2], K_nuscenes[1,2]]
vals_kt = [K_kitti[0,0], K_kitti[1,1], K_kitti[0,2], K_kitti[1,2]]
x = np.arange(len(props))
bars1 = ax4.bar(x - 0.2, vals_ns, 0.38, label='nuScenes', color=colors['nuScenes'], alpha=0.85)
bars2 = ax4.bar(x + 0.2, vals_kt, 0.38, label='KITTI',    color=colors['KITTI'],    alpha=0.85)
ax4.set_xticks(x); ax4.set_xticklabels(props, color='white')
ax4.set_title('Camera Intrinsics Comparison', color='white', fontsize=12, fontweight='bold')
ax4.set_ylabel('Pixel Value', color='white', fontsize=11)
ax4.legend(fontsize=9, facecolor='#2a2a3e', labelcolor='white')
ax4.set_facecolor('#0f0f23')
ax4.tick_params(colors='white')
ax4.spines[:].set_color('#444')

# Panel 5: Error distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(err_vals, bins=30, color='#f7b731', edgecolor='#c67c00', alpha=0.85)
ax5.axvline(err_vals.mean(), color='white', linestyle='--', lw=2, label=f'Mean={err_vals.mean():.1f}px')
ax5.set_xlabel('BEV Ref Point Pixel Error (px)', color='white', fontsize=11)
ax5.set_ylabel('Count', color='white', fontsize=11)
ax5.set_title('Distribution of Reference Point Errors', color='white', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9, facecolor='#2a2a3e', labelcolor='white')
ax5.set_facecolor('#0f0f23')
ax5.tick_params(colors='white')
ax5.spines[:].set_color('#444')

# Panel 6: Summary text
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('#0f0f23')
ax6.axis('off')
summary = (
    "DOMAIN GAP SUMMARY\n"
    "─────────────────────────────\n\n"
    f"  Focal Length\n"
    f"  nuScenes fx = {K_nuscenes[0,0]:.1f} px\n"
    f"  KITTI fx    = {K_kitti[0,0]:.1f} px\n"
    f"  Ratio        = {K_nuscenes[0,0]/K_kitti[0,0]:.2f}×\n\n"
    f"  Image Size\n"
    f"  nuScenes: 1600×900\n"
    f"  KITTI:    1242×375\n\n"
    f"  BEV Ref Point Error\n"
    f"  Mean:   {err_vals.mean():.1f} px\n"
    f"  Max:    {err_vals.max():.1f} px\n\n"
    "  IMPACT:\n"
    "  Same 3D point projects to\n"
    f"  ~{K_nuscenes[0,0]/K_kitti[0,0]:.1f}× different image position\n"
    "  → BEVFormer samples from\n"
    "    wrong image regions\n"
    "  → Depth/detection failure"
)
ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
         fontsize=10, color='white', fontfamily='monospace',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#1a1a3e', alpha=0.8))

plt.suptitle('Cross-Domain Camera Intrinsics Analysis: nuScenes → KITTI\n'
             'Quantifying the Core Motivation for Depth Anything V2 Integration',
             fontsize=14, fontweight='bold', color='white', y=0.98)

out_dir = "E:/bev_research/experiments/baseline"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "domain_gap_analysis.png")
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
plt.close()
print(f"\nKey statistics:")
print(f"  Focal length ratio: {K_nuscenes[0,0]/K_kitti[0,0]:.3f}×")
print(f"  Mean BEV ref point error: {err_vals.mean():.1f} px")
print(f"  Max BEV ref point error:  {err_vals.max():.1f} px")
print(f"  At 20m depth, height projection ratio: {h_ns[np.abs(depths_ns - 20).argmin()] / h_kt[np.abs(depths_kt - 20).argmin()]:.3f}×")
