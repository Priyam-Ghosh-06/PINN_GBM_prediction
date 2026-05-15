import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from nibabel.processing import resample_from_to

# ==========================================
# 1. SETUP & LOAD DATA
# ==========================================
base_dir = os.path.abspath(os.getcwd())
tumor_mask_path = os.path.join(base_dir, "patient_tumor_mask.nii.gz")
phi_field_path  = os.path.join(base_dir, "phi_field.nii.gz")

print("Loading and aligning fields...")
phi_img  = nib.load(phi_field_path)
phi_data = phi_img.get_fdata()

raw_tumor_img      = nib.load(tumor_mask_path)
tumor_img_aligned  = resample_from_to(raw_tumor_img, phi_img, order=0)
tumor_data         = tumor_img_aligned.get_fdata() > 0

# ==========================================
# 2. CALCULATE THE TUMOR CENTROID
# ==========================================
print("Calculating geometric prior (Tumor Centroid)...")
tumor_indices = np.argwhere(tumor_data)   # shape (N, 3), voxel-space
centroid_vox  = np.mean(tumor_indices, axis=0)
print(f"Computed Centroid in voxel space (x, y, z): {centroid_vox}")

# ==========================================
# 3. CREATE THE BOUNDARY PROBABILITY MAP
# ==========================================
print("Calculating tumor boundary distances...")
dist_out = distance_transform_edt(~tumor_data)
dist_in  = distance_transform_edt(tumor_data)

# Absolute distance from the exact tumor boundary (in voxels).
# NOTE: At boundary voxels both dist_in and dist_out are ~0,
#       so boundary_distance ≈ 0 → boundary_prob ≈ 1 (maximum weight). Correct.
boundary_distance = dist_out + dist_in

sigma_halo    = 3.0
boundary_prob = np.exp(-(boundary_distance**2) / (2 * sigma_halo**2))

brain_mask            = phi_data > 0.5
boundary_prob[~brain_mask] = 0.0
boundary_prob        /= np.sum(boundary_prob)   # valid probability distribution

# ==========================================
# 4. SAMPLE THE CLOUDS
# ==========================================
print("Sampling point clouds...")
num_boundary_points = 50000   # dense near boundary → drives L_data
num_uniform_points  = 50000   # uniform over brain  → drives L_PDE

x, y, z = np.indices(phi_data.shape)
coords   = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float64)
# coords[:,0..2] are INTEGER voxel indices at this point — raw, unnormalized.

# --- CLOUD 1: Dense Boundary Points (Data) ---
flat_prob        = boundary_prob.ravel()
boundary_indices = np.random.choice(len(coords), size=num_boundary_points,
                                    p=flat_prob, replace=False)
X_boundary = coords[boundary_indices]                            # voxel-space float64
C_boundary = tumor_data.ravel()[boundary_indices].reshape(-1, 1).astype(np.float64)

# --- CLOUD 2: Uniform Residual Points (PDE) ---
brain_indices   = np.where(brain_mask.ravel())[0]
uniform_indices = np.random.choice(brain_indices, size=num_uniform_points, replace=False)
X_uniform = coords[uniform_indices]                              # voxel-space float64

# ==========================================
# 5. NORMALIZATION  ← THE CRITICAL NEW STEP
# ==========================================
# WHY THIS IS HERE, NOT IN train_pikan.py:
#   Chebyshev polynomials T_n(x) are only stable on x ∈ [-1, 1].
#   Raw voxel coords (e.g. x ≈ 116.7) cause T_10(116.7) ≈ 10^23.7,
#   producing the observed L ≈ 10^26 and ||∇|| ≈ 10^48 before any weight update.
#   We compute the min/max from the BRAIN MASK (not the full grid) so the
#   normalization is tight around the actual data support.
#
# WHAT IS SAVED:
#   coord_mins, coord_maxs  — must be reloaded in train_pikan.py and
#                             inference_pikan.py to invert back to voxel/mm space.
#
# FORMULA:
#   x_norm = 2 * (x - x_min) / (x_max - x_min) - 1   →  x_norm ∈ [-1, 1]
#   s      = (x_max - x_min) / 2                       →  ∂x/∂x_norm = s
#   Laplacian correction: ∂²c/∂x² = (1/s²) * ∂²c/∂x_norm²   (see handoff)

print("Normalizing coordinates to [-1, 1] using brain-mask support...")

# Compute normalization bounds from all sampled brain points combined.
all_sampled = np.vstack([X_boundary, X_uniform])   # (100000, 3)
coord_mins  = all_sampled.min(axis=0)              # shape (3,) — one per axis
coord_maxs  = all_sampled.max(axis=0)              # shape (3,)
coord_range = coord_maxs - coord_mins              # (x_max - x_min) per axis

# Guard: if any axis has zero range (degenerate volume), raise early.
assert np.all(coord_range > 0), (
    f"Degenerate coordinate range detected: {coord_range}. "
    "Check that brain_mask spans more than one voxel per axis."
)

def normalize(pts):
    """Map pts from voxel space to [-1, 1]^3 using pre-computed bounds."""
    return 2.0 * (pts - coord_mins) / coord_range - 1.0

X_boundary_norm = normalize(X_boundary)   # shape (50000, 3), all values ∈ [-1, 1]
X_uniform_norm  = normalize(X_uniform)    # shape (50000, 3)

# Sanity check — every coordinate must be strictly within [-1, 1].
assert X_boundary_norm.min() >= -1.0 - 1e-9 and X_boundary_norm.max() <= 1.0 + 1e-9, \
    "Boundary points exceed [-1,1] after normalization."
assert X_uniform_norm.min()  >= -1.0 - 1e-9 and X_uniform_norm.max()  <= 1.0 + 1e-9, \
    "Uniform points exceed [-1,1] after normalization."

# Normalize the centroid using the same bounds so train_pikan.py can initialize
# the learnable seed parameter directly in normalized space.
centroid_norm = normalize(centroid_vox.reshape(1, 3)).flatten()   # shape (3,)
print(f"Centroid in normalized space: {centroid_norm}")
print(f"Normalization scale s per axis (mm/unit): {coord_range / 2.0}")
# ↑ Save and pass this s into train_pikan.py.  The PDE loss needs:
#   laplacian_physical = laplacian_normalized / s^2
#   where s = coord_range / 2, one scalar per spatial axis.

# ==========================================
# 6. ADD THE TIME DIMENSION (t)
# ==========================================
# t is already normalized: t_scan = 1.0 represents the acquisition time-point.
# Residual points sample t ∈ [0, 1] uniformly — this is intentional,
# allowing the network to encode the full temporal trajectory.
#
# IMPORTANT NOTE ON TEMPORAL IDENTIFIABILITY (see handoff §4):
#   With a single MRI, t_scan=1.0 is a convention, NOT a physical duration.
#   The product (D * t) is unidentifiable without a second time-point.
#   Document this limitation; the model will find A solution, not THE solution.

t_scan = 1.0

T_boundary    = np.full((num_boundary_points, 1), t_scan)
Points_Data   = np.hstack((X_boundary_norm, T_boundary))     # (50000, 4): x̃,ỹ,z̃,t

T_uniform     = np.random.uniform(0, t_scan, size=(num_uniform_points, 1))
Points_Residual = np.hstack((X_uniform_norm, T_uniform))     # (50000, 4): x̃,ỹ,z̃,t

# ==========================================
# 7. EXPORT ARRAYS FOR PYTORCH
# ==========================================
print(f"\nGenerated {len(Points_Data)} Data Points     (Shape: {Points_Data.shape})")
print(f"Generated {len(Points_Residual)} Residual Points (Shape: {Points_Residual.shape})")

np.save("PINN_Data_Points.npy",      Points_Data)
np.save("PINN_Data_C.npy",           C_boundary)
np.save("PINN_Residual_Points.npy",  Points_Residual)
np.save("PINN_Tumor_Centroid.npy",   centroid_norm)     # NOW in normalized space

# CRITICAL: these must be loaded by train_pikan.py and inference_pikan.py
# to (a) initialize the seed parameter in the right space, and
# (b) apply the 1/s² Laplacian correction in the PDE loss.
np.save("PINN_Coord_Mins.npy",   coord_mins)    # shape (3,)
np.save("PINN_Coord_Maxs.npy",   coord_maxs)    # shape (3,)
np.save("PINN_Coord_Scale_s.npy", coord_range / 2.0)  # shape (3,): s_x, s_y, s_z

print("\nNormalization parameters saved:")
print(f"  coord_mins  : {coord_mins}")
print(f"  coord_maxs  : {coord_maxs}")
print(f"  scale s     : {coord_range / 2.0}  (use as 1/s² in PDE Laplacian)")

print("\nData extraction complete. Ready for train_pikan.py.")
