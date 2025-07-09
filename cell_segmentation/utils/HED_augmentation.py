import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import torch
import torch.nn.functional as F
import math
import random


### Instructions to find the matrix M for each slide ###

# 1. Open the slide in QuPath
# 2. Select image type = Brightfield H&E
# 3. Follow the instruction given here: https://qupath.readthedocs.io/en/stable/docs/tutorials/separating_stains.html
#    In summary, you need to:
#       a. Draw a region of interest (ROI) that contains background + H staining + E staining (the smaller the region, the better)
#       b. Go to the menu: Analyze -> Estimate stain vectors
#       c. Stick the box saying "Exclude unrecognized colors (H&E only)"
#       d. Click "Auto" and we get the current M matrix
# 4. Copy the M matrix values and paste them below (NB: the last row is residual as we do not have DAB staining)

# For slide that are not detected as Brightfield H&E in QuPath (like breast_s0 for instance), do the following:
# 1. Open the slide in Fiji using a serie that is intermediate in terms of resolution
# 2. Do Image -> Color -> Make Composite
# 3. Do Image -> Color -> Stack to RGB
# 4. Save using File -> Save As -> OME-TIFF
# 5. Open the OME-TIFF in QuPath and this time it should be detected as Brightfield H&E
# 6. Follow the steps 2 to 4 above

#########################################################

# M matrices dictionary
# M_matrices (dict): Dictionary mapping slide_id -> M matrix.

M_matrices = {
    # M matrix for breast_s0 slide
    "breast_s0": np.array([
        [0.588, 0.757, 0.284],
        [0.223, 0.946, 0.235],
        [-0.225, -0.185, 0.957]
    ], dtype=np.float32),

    # M matrix for breast_s1 slide
    "breast_s1": np.array([
        [0.58, 0.758, 0.299],
        [0.219, 0.912, 0.347],
        [-0.025, -0.35, 0.936]
    ], dtype=np.float32),

    # M matrix for breast_s3 slide
    "breast_s3": np.array([
        [0.567, 0.782, 0.26],
        [0.193, 0.952, 0.237],
        [-0.154, -0.21, 0.966]
    ], dtype=np.float32),

    # M matrix for breast_s6 slide
    "breast_s6": np.array([
        [0.549, 0.803, 0.231],
        [0.119, 0.95, 0.288],
        [0.025, -0.293, 0.956]
    ], dtype=np.float32),

    # M matrix for lung_s1 slide
    "lung_s1": np.array([
        [0.396, 0.875, 0.278],
        [0.126, 0.948, 0.293],
        [-0.028, -0.292, 0.956]
    ], dtype=np.float32),

    # M matrix for lung_s3 slide
    "lung_s3": np.array([
        [0.329, 0.916, 0.229],
        [0.101, 0.951, 0.291],
        [0.205, -0.306, 0.93]
    ], dtype=np.float32),

    # M matrix for skin_s1 slide
    "skin_s1": np.array([
        [0.455, 0.846, 0.278],
        [0.111, 0.928, 0.356],
        [0.122, -0.368, 0.922]
    ], dtype=np.float32),

    # M matrix for skin_s2 slide
    "skin_s2": np.array([
        [0.577, 0.757, 0.308],
        [0.164, 0.941, 0.296],
        [-0.15, -0.273, 0.95]
    ], dtype=np.float32),

    # M matrix for skin_s3 slide
    "skin_s3": np.array([
        [0.449, 0.852, 0.27],
        [0.105, 0.948, 0.3],
        [-0,  -0.302,  0.953]
    ], dtype=np.float32),

    # M matrix for skin_s4 slide
    "skin_s4": np.array([
        [0.449, 0.86, 0.241],
        [0.114, 0.934, 0.339],
        [0.19,  -0.355,  0.915]
    ], dtype=np.float32),

    # M matrix for pancreatic_s0 slide
    "pancreatic_s0": np.array([
        [0.393, 0.879, 0.269],
        [0.065, 0.947, 0.316],
        [0.069, -0.32, 0.945]
    ], dtype=np.float32),

    # M matrix for pancreatic_s1 slide
    "pancreatic_s1": np.array([
        [0.412, 0.867, 0.281],
        [0.113, 0.945, 0.306],
        [-0.002, -0.308, 0.951]
    ], dtype=np.float32),

    # M matrix for pancreatic_s2 slide
    "pancreatic_s2": np.array([
        [0.448, 0.854, 0.265],
        [0.139, 0.94, 0.312],
        [0.054, -0.321, 0.945]
    ], dtype=np.float32),

    # M matrix for heart_s0 slide
    "heart_s0": np.array([
        [0.579, 0.764, 0.284],
        [0.143, 0.948, 0.284],
        [-0.114, -0.269, 0.956]
    ], dtype=np.float32),

    # M matrix for colon_s1 slide
    "colon_s1": np.array([
        [0.261, 0.923, 0.282],
        [0.094, 0.951, 0.294],
        [0.015, -0.296, 0.955]
    ], dtype=np.float32),

    # M matrix for colon_s2 slide
    "colon_s2": np.array([
        [0.343, 0.894, 0.287],
        [0.091, 0.925, 0.368],
        [0.24, -0.379, 0.894]
    ], dtype=np.float32),

    # M matrix for kidney_s0 slide
    "kidney_s0": np.array([
        [0.539, 0.804, 0.249],
        [0.18, 0.941, 0.287],
        [-0.01, -0.29, 0.957]
    ], dtype=np.float32),

    # M matrix for kidney_s1 slide
    "kidney_s1": np.array([
        [0.658, 0.725, 0.202],
        [0.228, 0.932, 0.281],
        [0.034, -0.296, 0.955]
    ], dtype=np.float32),

    # M matrix for liver_s0 slide
    "liver_s0": np.array([
        [0.411, 0.871, 0.269],
        [0.11, 0.937, 0.333],
        [0.122, -0.345, 0.931]
    ], dtype=np.float32),

    # M matrix for liver_s1 slide
    "liver_s1": np.array([
        [0.637, 0.745, 0.2],
        [0.186, 0.928, 0.321],
        [0.11, -0.345, 0.932]
    ], dtype=np.float32),

    # M matrix for tonsil_s0 slide
    "tonsil_s0": np.array([
        [0.557, 0.774, 0.303],
        [0.111, 0.94, 0.321],
        [-0.079, -0.314, 0.946]
    ], dtype=np.float32),

    # M matrix for tonsil_s1 slide
    "tonsil_s1": np.array([
        [0.484, 0.828, 0.284],
        [0.102, 0.945, 0.31],
        [-0.03, -0.309, 0.951]
    ], dtype=np.float32),

    # M matrix for lymph_node_s0 slide
    "lymph_node_s0": np.array([
        [0.475, 0.837, 0.273],
        [0.155, 0.934, 0.321],
        [0.041, -0.33, 0.943]
    ], dtype=np.float32),

    # M matrix for ovary_s0 slide
    "ovary_s0": np.array([
        [0.451, 0.854, 0.258],
        [0.128, 0.948, 0.291],
        [0.012, -0.295, 0.955]
    ], dtype=np.float32),

    # M matrix for ovary_s1 slide
    "ovary_s1": np.array([
        [0.557, 0.793, 0.248],
        [0.146, 0.943, 0.298],
        [0.005, -0.302, 0.953]
    ], dtype=np.float32),

    # M matrix for brain_s0 slide
    "brain_s0": np.array([
        [0.503, 0.816, 0.286],
        [0.139, 0.941, 0.309],
        [-0.043, -0.306, 0.951]
    ], dtype=np.float32),

    # M matrix for bone_marrow_s0 slide
    "bone_marrow_s0": np.array([
        [0.554, 0.786, 0.274],
        [0.075, 0.958, 0.276],
        [-0.093, -0.269, 0.959]
    ], dtype=np.float32),

    # M matrix for bone_marrow_s1 slide
    "bone_marrow_s1": np.array([
        [0.462, 0.833, 0.304],
        [0.1, 0.962, 0.254],
        [-0.212, -0.229, 0.95]
    ], dtype=np.float32),

    # M matrix for bone_s0 slide
    "bone_s0": np.array([
        [0.5, 0.815, 0.293],
        [0.105, 0.959, 0.264],
        [-0.159, -0.246, 0.956]
    ], dtype=np.float32),

    # M matrix for prostate_s0 slide
    "prostate_s0": np.array([
        [0.484, 0.824, 0.293],
        [0.132, 0.947, 0.291],
        [-0.101, -0.28, 0.955]
    ], dtype=np.float32),

    # M matrix for cervix_s0 slide
    "cervix_s0": np.array([
        [0.559, 0.785, 0.267],
        [0.15, 0.955, 0.257],
        [-0.124, -0.24, 0.963]
    ], dtype=np.float32)
}



class HEDAugAlbumentations(ImageOnlyTransform):
    """
    A simple callable that applies HED-based color augmentation 
    if random.random() < p. Works with Albumentations if you wrap it
    in a Compose that doesn’t drop extra keys.

    Args:
        sigma (float): Standard deviation for random scaling/offset in HED space.
        p (float): Probability of applying this transform.

    Inspired from the paper: Faryna, K., Van der Laak, J. and Litjens, G., 2021, February. Tailoring automated data augmentation to H&E-stained histopathology. In Medical imaging with deep learning.
    """

    def __init__(self, sigma=0.03, p=0.25):
        
        self.sigma = sigma          # Histopathology HED augmentation is often done with something like sigma in [0.01..0.05].
                                    # sigma=0.02 is a fairly common 
                                    # If patches have high inter-slide variation (different labs, scanners, or staining protocols), you might increase sigma (e.g., 0.03 or 0.04) to simulate larger color shifts.
                                    # If large color changes degrade performance or produce unrealistic “cartoonish” patches, lower sigma (e.g., 0.01) so that the augmentation is more subtle.
        self.p = p

        # Matrix M: from HED -> RGB (approx). We'll invert it for RGB -> HED
        # See here for explanations: Quantification of histochemical staining by color deconvolution, Arnout Ruifrok, Dennis A Johnston, January 2001
        # Default values coming from default values in QuPath
        M_default = np.array([
            [0.651, 0.701, 0.29],
            [0.216, 0.801, 0.558],
            [0.316, -0.598, 0.737]
        ], dtype=np.float32)
        self.M_default = torch.tensor(M_default, dtype=torch.float32)
        self.RGB2HED_default = torch.inverse(self.M_default)

        self.M_matrices = {k: torch.tensor(v, dtype=torch.float32) for k,v in M_matrices.items()}
        self.RGB2HEDs = {k: torch.inverse(v) for k,v in self.M_matrices.items()}
        
    
    def __call__(self, **data):
        """
        data is a dictionary that will contain "image", "mask", "slide_id", etc.
        Return data with the updated "image".
        """
        # If there's no slide_id or image, just return
        if "image" not in data:
            print("[WARNING] There is no image in data for H&E specific augmentation")
            return data

        slide_id = data.get("slide_id", None)
        img = data["image"]
        
        # Probability check
        if random.random() >= self.p:
            # Don’t apply the augmentation
            return data

        # If we do apply it:
        # Choose the M and RGB2HED matrices for the given slide_id
        if slide_id in self.M_matrices:
            M = self.M_matrices[slide_id]
            RGB2HED = self.RGB2HEDs[slide_id]
        else:
            print(f"[WARNING] Using default M, slide_id='{slide_id}' not found")
            M = self.M_default
            RGB2HED = self.RGB2HED_default

        # Convert from NumPy [H,W,C] to a float Torch tensor [C,H,W] in [0..1]
        #    (Albumentations typically provides 8-bit images, so we scale them).
        if img.dtype != np.float32 and img.dtype != np.float64:
            img_t = torch.from_numpy(img.astype(np.float32))
        else:
            img_t = torch.from_numpy(img)
        
        # shape = (H,W,C). Permute to (C,H,W)
        img_t = img_t.permute(2, 0, 1)  # [C,H,W]

        # Scale to [0..1] if needed
        if img_t.max() > 1.0:
            img_t = img_t / 255.0

        eps = 1e-6
        C, H, W = img_t.shape
        flat = img_t.reshape(C, -1).T  # [N,3] with , N = H*W
        # Get optical density (OD) space
        S = torch.matmul(-torch.log(flat + eps), RGB2HED)

        # Random shift in HED
        alpha = torch.normal(mean=1.0, std=self.sigma, size=(1,3))
        beta  = torch.normal(mean=0.0, std=self.sigma, size=(1,3))
        S_hat = alpha * S + beta

        # Convert back to "RGB" space
        out = torch.exp(-torch.matmul(S_hat, M))

        # Reshape back to (C,H,W)
        out = out.T.reshape(C, H, W)

        # Clip to [0..1]
        out = torch.clamp(out, 0.0, 1.0)

        # Convert back to NumPy with shape (H,W,C) in [0..255]
        out = (out * 255.0).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

        data["image"] = out


        # # Checking
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # axes[0].imshow(img)
        # axes[0].set_title("Before HED Augmentation")
        # axes[0].axis("off")
        # axes[1].imshow(out)
        # axes[1].set_title("After HED Augmentation")
        # axes[1].axis("off")
        # plt.tight_layout()
        # plt.show(block=False)
        # plt.pause(10)  # Display time
        # plt.close(fig)

        return data
    



class ComposeWithExtra(A.Compose):
    def __call__(self, **kwargs):
        """
        kwargs might be {'image':..., 'mask':..., 'slide_id':..., ...}
        We ensure 'slide_id' isn't dropped.
        """
        # Let's build a dictionary for Albumentations
        data = {}
        if 'image' in kwargs:
            data['image'] = kwargs['image']
        if 'mask' in kwargs:
            data['mask'] = kwargs['mask']

        # We keep slide_id or any other metadata
        if 'slide_id' in kwargs:
            data['slide_id'] = kwargs['slide_id']

        # Run the parent compose
        result = super().__call__(**data)

        # Reattach leftover keys to final output
        for k, v in kwargs.items():
            if k not in result:
                result[k] = v

        return result