import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn
import yaml
import sys
from pathlib import Path
import zipfile
import io
from natsort import natsorted
from scipy.sparse import csr_matrix
import time

# Add the root directory of our project to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[2]))

from base_ml.base_loss import FocalTverskyLoss, DiceLoss, MSELossMaps, MSGELossMaps, XentropyLoss, MCFocalTverskyLoss
from cell_segmentation.datasets.pannuke import PanNukeDataset



################################################ TO FILL ##################################################################################


# Remapping of the original class names to the new class names if cell_cat_id is built from a previous physically existing dataset / Use None if not needed
# Should be in the following format:
    # *** List of original class names (index corresponds to type_map value):
    # orig_class_names = ["Background", "T_NK", "B_Plasma", "Myeloid", "Blood_vessel", "Fibroblast_Myofibroblast", "Epithelial", "Specialized", "Melanocyte", "Dead"]
    # *** Dict mapping group names to list of original names:
    # grouping = {"Immune": ["T_NK", "B_Plasma", "Myeloid"], "Stromal": ["Blood_vessel", "Fibroblast_Myofibroblast"]}
    # *** Dict mapping final category names to new indices:
    # new_class_mapping = {"Background": 0, "Immune": 1, "Stromal": 2, "Epithelial": 3, "Specialized": 4, "Melanocyte": 5, "Dead": 6}
orig_class_names = None
grouping = None
new_class_mapping = None

# Parameters
batch_size = 30
num_classes = 10   # Including background
ignore_cat = []   # Categories to ignore = very small weight in loss computations
num_tissues = 13
height, width = 256, 256   # Patch dimensions
N = 5   # Number of repetitions for scenarios containing some random values

# Losses
    # NP
np_ft_loss = FocalTverskyLoss()
np_dice_loss = DiceLoss()
    # HV
hv_mse_loss = MSELossMaps()
hv_msge_loss = MSGELossMaps()
    # NT
nt_bce_loss = XentropyLoss(class_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # put very small weight for ignore cats
nt_dice_loss = DiceLoss([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # put very small weight for ignore cats
nt_ft_loss = MCFocalTverskyLoss(num_classes=num_classes, class_weights=[0.044, 0.086, 0.101, 0.09, 0.106, 0.088, 0.044, 0.194, 0.15, 0.142]) # put very small weight for ignore cats
    # Tissue
tissue_ce_loss = nn.CrossEntropyLoss()

# Define the weights for each loss component
weight_np_ft = 1.0
weight_np_dice = 1.0
weight_hv_mse = 1.0
weight_hv_msge = 1.0
weight_nt_bce = 1.0
weight_nt_dice = 1.0
weight_nt_ft = 1.0
weight_tissue_ce = 1.0

# ID
cell_cat_id = 'ct_1'   # Should be the cell_cat_id for the folder that physically exists => if the dataset is built from a previous dataset, the cell_cat_id should be the one of the previous dataset (use the remapping option below to remap the classes)
dataset_id = 'ds_2'
training_id = 'training_2'

# Paths
data_zip_folder = f'/Volumes/DD_FGS/MICS/data_HE2CellType/HE2CT/prepared_datasets_cat/{cell_cat_id}/ALL'   # Contains images.zip, label.zip, types.csv
dataset_folder = f'/Volumes/DD_FGS/MICS/data_HE2CellType/HE2CT/training_datasets/{dataset_id}'   # Contains cell_count_{set}.csv for set in ['train', 'valid', 'test'] with patch ids selection, and the dataset_config.yaml
save_path = f'/Volumes/DD_FGS/MICS/data_HE2CellType/HE2CT/trainings/{training_id}/train_losses_extrema_before.csv'

###########################################################################################################################################




def build_type_mapping(orig_class_names, grouping, new_class_mapping):
    """
    Build a vectorized mapping array.
    
    Parameters:
      orig_class_names (list of str): Original class names in order. Index i corresponds to class i.
      grouping (dict): Mapping of group names to lists of original class names that should be grouped.
      new_class_mapping (dict): Mapping of final category names (including groups and standalone classes) to new integer labels.
      
    Returns:
      np.ndarray: A 1D array 'mapping' where mapping[original_index] = new_index.
    """
    n = len(orig_class_names)
    mapping = np.empty(n, dtype=np.int32)
    # Assume background (index 0) always maps to 0
    mapping[0] = 0
    for i in range(1, n):
        orig_name = orig_class_names[i]
        new_val = None
        # Check if the original name belongs to any group
        for group, members in grouping.items():
            if orig_name in members:
                new_val = new_class_mapping[group]
                break
        # If not in a group, then look up directly in the new mapping
        if new_val is None:
            if orig_name in new_class_mapping:
                new_val = new_class_mapping[orig_name]
            else:
                raise ValueError(f"No mapping provided for original class '{orig_name}'")
        mapping[i] = new_val
    return mapping




class MaskDataset(Dataset):
    
    def __init__(self, data_zip_folder, dataset_folder, num_classes, orig_class_names=None, grouping=None, new_class_mapping=None):
        
        self.data_zip_folder = data_zip_folder
        self.dataset_folder = dataset_folder
        self.num_classes = num_classes

        # Save mapping info for remapping the type_map
        self.orig_class_names = orig_class_names
        self.grouping = grouping
        self.new_class_mapping = new_class_mapping

        self.labels_zip_path = os.path.join(self.data_zip_folder, "labels.zip")
        self.zip_file = None  # Initialize as None; open in the worker process
        self.tissue_csv = pd.read_csv(os.path.join(self.data_zip_folder, "types.csv"))

        # Load the list of image names from cell_count_train.csv
        self.mask_names_list = pd.read_csv(os.path.join(self.dataset_folder, "cell_count_train.csv"))['Image'].tolist()
        # Replace the .png extension with .npz
        self.mask_names_list = [mask_name.replace(".png", "")  + ".npz" for mask_name in self.mask_names_list]
        print(f"[INFO] Found {len(self.mask_names_list)} masks for train set.")

        # Verify that the listed masks exist in the ZIP file
        with zipfile.ZipFile(self.labels_zip_path, 'r') as mask_zip:
            zip_label_names = set(f for f in mask_zip.namelist() if f.endswith(".npz"))
            
            missing_masks = [mask_name for mask_name in self.mask_names_list if mask_name not in zip_label_names]
            if missing_masks:
                print(f"[WARNING] The following masks listed in the CSV were not found in the ZIP file:\n{missing_masks}")

            # Retain only images present in both the CSV and ZIP file
            self.mask_names_list = [img_name for img_name in self.mask_names_list if img_name in zip_label_names]
            print(f"[INFO] Found {len(self.mask_names_list)} valid masks in the ZIP file.")

        if not self.mask_names_list:
            raise ValueError("No valid mask names found in the CSV file or ZIP file!")
        
        # Sort the mask names list for consistency
        self.mask_names_list = natsorted(self.mask_names_list)
        
        with open(os.path.join(dataset_folder, 'dataset_config.yaml'), 'r') as file:
            self.dataset_config = yaml.safe_load(file)


        # Precompute the vectorized mapping array (if mapping parameters are provided)
        if self.orig_class_names is not None and self.grouping is not None and self.new_class_mapping is not None:
            self.type_mapping_array = build_type_mapping(self.orig_class_names, self.grouping, self.new_class_mapping)
        else:
            self.type_mapping_array = None
    

    def _init_zipfile(self):
        if self.zip_file is None:  # Open the ZIP file only if not already open
            self.zip_file = zipfile.ZipFile(self.labels_zip_path, 'r')


    def load_maskfile(self, mask_name: str):

        # Load the .npz file using numpy
        data = np.load(io.BytesIO(self.zip_file.read(mask_name)), allow_pickle=True)

        # Reconstruct sparse matrices
        inst_map_sparse = csr_matrix(
            (data["inst_map_data"], data["inst_map_indices"], data["inst_map_indptr"]),
            shape=data["inst_map_shape"]
        )
        type_map_sparse = csr_matrix(
            (data["type_map_data"], data["type_map_indices"], data["type_map_indptr"]),
            shape=data["type_map_shape"]
        )

        # Convert sparse matrices to dense numpy arrays
        inst_map = inst_map_sparse.toarray().astype(np.int32)
        np_map = inst_map.copy()
        type_map = type_map_sparse.toarray().astype(np.int32)

        # Remap type_map using vectorized mapping if provided
        if self.type_mapping_array is not None:
            type_map = self.type_mapping_array[type_map]

        return np_map, inst_map, type_map
    

    def __len__(self):
        return len(self.mask_names_list)

    def __getitem__(self, idx):

        mask_name = self.mask_names_list[idx]
        
        self._init_zipfile()  # Ensure the ZIP file is open in this worker
        np_map, inst_map, type_map = self.load_maskfile(mask_name)

        # NP
        np_map[np_map > 0] = 1
        np_map = torch.Tensor(np_map).type(torch.int64)
        gt_nuclei_binary_map_onehot = (F.one_hot(np_map, num_classes=2)).type(torch.float32)
        nuclei_binary_map = gt_nuclei_binary_map_onehot.permute(2, 0, 1) # Shape: (2, H, W)

        # HV
        hv_map = PanNukeDataset.gen_instance_hv_map(inst_map)
        hv_map = torch.Tensor(hv_map).type(torch.float32) # Shape: (2, H, W) / First dimension is horizontal (horizontal gradient (-1 to 1)), last is vertical (vertical gradient (-1 to 1))
        
        # NT
        type_map = torch.tensor(type_map).type(torch.int64)
        one_hot_map = F.one_hot(type_map, num_classes=self.num_classes).type(torch.float32)
        nuclei_type_map = one_hot_map.permute(2, 0, 1)  # Shape: (num_classes, H, W)

        # Tissue
        img_id = os.path.basename(mask_name).replace(".npz", "")
        tissue_type = self.tissue_csv[self.tissue_csv["img"]==img_id+".png"]["type"].values[0]
        tissue_id = self.dataset_config["tissue_types"][tissue_type]
        tissue_id = torch.tensor(tissue_id).type(torch.float32)

        return nuclei_binary_map, hv_map, nuclei_type_map, tissue_id
    


def func(x):
    return torch.where(x < 0, x + 1, x - 1)



def generate_predictions(gt, scenario, branch, device):

    if branch == "tissue":

        if scenario == "min":
            idx = gt.long()
            pred = torch.eye(num_tissues).to(device)
            pred = pred[idx]
            pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
            pred = torch.log(pred / (1 - pred)) # Convert to logit
        
        elif scenario == "max":
            pred = torch.rand((gt.shape[0], num_tissues)).to(device)
            for i, index in enumerate(gt.long()):
                pred[i, index] = 0
            if pred.shape[1] > 1:
                pred /= pred.sum(dim=1, keepdim=True)  # Normalize to sum to 1
            pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
            pred = torch.log(pred / (1 - pred)) # Convert to logit
            
        elif scenario == "random":
            pred = torch.rand((gt.shape[0], num_tissues)).to(device)
            if pred.shape[1] > 1:
                pred /= pred.sum(dim=1, keepdim=True)  # Normalize to sum to 1
            pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
            pred = torch.log(pred / (1 - pred)) # Convert to logit
    
    else:

        if scenario == "min":
            pred = gt.clone().to(device)
        
        elif branch == "NP":
            if scenario == "max":
                pred = torch.zeros_like(gt).to(device)
                pred = 1 - gt
            elif scenario == "random":
                pred = torch.rand_like(gt).to(device)
                pred /= pred.sum(dim=1, keepdim=True)  # Normalize to sum to 1
        
        elif branch == "HV":
            if scenario == "max":
                pred = torch.zeros_like(gt).to(device)
                pred[:, 0, :, :] = func(gt[:, 0, :, :])
                pred[:, 1, :, :] = func(gt[:, 1, :, :])
            elif scenario == "random":
                pred = torch.rand_like(gt).to(device) * 2 - 1  # Random values in range [-1, 1]
        
        elif branch == "NT":
            if scenario == "max":
                pred = torch.rand_like(gt).to(device)
                gt_classes = torch.argmax(gt, dim=1, keepdim=True)
                pred.scatter_(1, gt_classes, 0.0)  # Set the true class probabilities to 0
                pred /= pred.sum(dim=1, keepdim=True)  # Normalize to sum to 1
            elif scenario == "random":
                pred = torch.rand_like(gt).to(device)
                pred /= pred.sum(dim=1, keepdim=True)  # Normalize to sum to 1
    
    return pred




def compute_losses(dataloader, device, scenario):
    """
    Compute losses for a given scenario across all batches in the dataloader.
    """
    batch_results = {
        "np_ft_loss": [],
        "np_dice_loss": [],
        "hv_mse_loss": [],
        "hv_msge_loss": [],
        "nt_bce_loss": [],
        "nt_dice_loss": [],
        "nt_ft_loss": [],
        "tissue_ce_loss": [],
        "total_loss": []
    }

    for gt_batch in tqdm(dataloader, desc=f"{scenario.capitalize()}", total=len(dataloader), leave=False):
        gt_batch = [x.to(device) for x in gt_batch]
        nuclei_binary_map, hv_map, nuclei_type_map, tissue_type = gt_batch

        pred_np = generate_predictions(nuclei_binary_map, scenario, "NP", device)
        pred_hv = generate_predictions(hv_map, scenario, "HV", device)
        pred_nt = generate_predictions(nuclei_type_map, scenario, "NT", device)
        pred_tissue = generate_predictions(tissue_type, scenario, "tissue", device)

        batch_results["np_ft_loss"].append(np_ft_loss(pred_np, nuclei_binary_map).item())
        batch_results["np_dice_loss"].append(np_dice_loss(pred_np, nuclei_binary_map).item())
        batch_results["hv_mse_loss"].append(hv_mse_loss(pred_hv, hv_map).item())
        batch_results["hv_msge_loss"].append(hv_msge_loss(pred_hv, hv_map, focus=nuclei_binary_map, device=device).item())
        batch_results["nt_bce_loss"].append(nt_bce_loss(pred_nt, nuclei_type_map).item())
        batch_results["nt_dice_loss"].append(nt_dice_loss(pred_nt, nuclei_type_map).item())
        batch_results["nt_ft_loss"].append(nt_ft_loss(pred_nt, nuclei_type_map).item())
        batch_results["tissue_ce_loss"].append(tissue_ce_loss(pred_tissue, tissue_type).item())

        total_loss_val = (weight_np_ft * batch_results["np_ft_loss"][-1] +
                          weight_np_dice * batch_results["np_dice_loss"][-1] +
                          weight_hv_mse * batch_results["hv_mse_loss"][-1] +
                          weight_hv_msge * batch_results["hv_msge_loss"][-1] +
                          weight_nt_bce * batch_results["nt_bce_loss"][-1] +
                          weight_nt_dice * batch_results["nt_dice_loss"][-1] +
                          weight_nt_ft * batch_results["nt_ft_loss"][-1] +
                          weight_tissue_ce * batch_results["tissue_ce_loss"][-1])

        batch_results["total_loss"].append(total_loss_val)

    return {key: np.mean(values) for key, values in batch_results.items()}


def main():

    print(f"\n==== Processing dataset: {dataset_id} for training: {training_id} ====")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"\nUsing device: {device}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = MaskDataset(data_zip_folder, dataset_folder, num_classes, orig_class_names, grouping, new_class_mapping)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=9)

    results = {key: {"min": None, "max": [], "random": []} for key in [
        "np_ft_loss", "np_dice_loss", "hv_mse_loss", "hv_msge_loss",
        "nt_bce_loss", "nt_dice_loss", "nt_ft_loss", "tissue_ce_loss", "total_loss"
    ]}

    # Compute the "min" scenario once
    print("\n* Computing 'min' scenario...")
    min_results = compute_losses(dataloader, device, "min")
    for key in results:
        results[key]["min"] = min_results[key]
    print('OK.')

    # Compute "max" and "random" scenarios across repetitions
    print("\n* Computing 'max' and 'random' scenarios with repetitions...")
    for _ in tqdm(range(N), leave=True, desc="Repetition", position=0):
        for scenario in ["max", "random"]:
            scenario_results = compute_losses(dataloader, device, scenario)
            for key in results:
                results[key][scenario].append(scenario_results[key])
    print('OK.')

    # Combine results
    print("\n* Results:")
    data = {"Scenario": ["Min", "Max", "Random"]}
    for key in results:
        data[f"{key}"] = [
            results[key]["min"],
            np.mean(results[key]["max"]),
            np.mean(results[key]["random"])
        ]

    df = pd.DataFrame(data)
    print(df)

    # Save results
    print(f"\n* Saving...")
    df.to_csv(save_path, index=False)
    print('Done.')
    


if __name__ == "__main__":
    main()