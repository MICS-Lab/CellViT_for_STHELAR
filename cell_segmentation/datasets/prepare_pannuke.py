# -*- coding: utf-8 -*-
# Prepare Pannuke Dataset by converting and resorting files
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
from math import ceil
import gc

from cell_segmentation.utils.metrics import remap_label



def load_sparse_3d_masks_chunked(file_path, mask_shape, chunk_size):
    """
    Load 3D masks from a sparse .npz file in chunks and process them incrementally.
    
    Args:
        file_path (str): Path to the saved .npz file.
        mask_shape (tuple): Original shape of the mask (height, width, channels).
        chunk_size (int): Number of masks to load and process per chunk.
        
    Yields:
        list: List of dense 3D masks (height, width, channels) for the current chunk.
    """

    print("Loading sparse masks...")
    sparse_matrix = load_npz(file_path)
    height, width, channels = mask_shape
    flat_size = height * width  # Number of pixels per mask slice
    total_masks = sparse_matrix.shape[0] // flat_size  # Total number of masks

    for start_idx in range(0, total_masks, chunk_size):
        end_idx = min(start_idx + chunk_size, total_masks)
        chunk_size_actual = end_idx - start_idx
        
        # Preallocate memory for dense masks in this chunk
        masks_chunk = []
        for i in range(chunk_size_actual):
            # Calculate the row range for mask `i` in this chunk
            start_row = (start_idx + i) * flat_size
            end_row = start_row + flat_size
            
            # Extract the rows corresponding to the 3D mask
            slices = sparse_matrix[start_row:end_row].toarray()
            
            # Reshape back to the original (height, width, channels)
            mask = slices.reshape(mask_shape)
            masks_chunk.append(mask)
        
        yield masks_chunk



def save_sparse_maps_single_file(inst_map, type_map, output_path, outname):
    """
    Save instance and type maps as sparse matrices in a single .npz file.

    Parameters:
    - inst_map: 2D numpy array, the instance map.
    - type_map: 2D numpy array, the type map.
    - output_path: Path object or string, the directory to save the file.
    - outname: String, the base name of the output file (without extension).
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Convert to sparse format
    inst_map_sparse = csr_matrix(inst_map)
    type_map_sparse = csr_matrix(type_map)

    # Save both sparse matrices into a single .npz file using numpy.savez
    combined_path = os.path.join(output_path, "labels", outname)
    np.savez(
        combined_path,
        inst_map_data=inst_map_sparse.data,
        inst_map_indices=inst_map_sparse.indices,
        inst_map_indptr=inst_map_sparse.indptr,
        inst_map_shape=inst_map_sparse.shape,
        type_map_data=type_map_sparse.data,
        type_map_indices=type_map_sparse.indices,
        type_map_indptr=type_map_sparse.indptr,
        type_map_shape=type_map_sparse.shape,
    )




def process_ds(input_path, output_path, list_cat) -> None:

    print(f"\n==== Processing slide {os.path.basename(input_path)} ====")

    print('\nUsing list_cat:', list_cat)
    
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)
    # os.makedirs(os.path.join(output_path, "masks_cell_ids_nuclei"), exist_ok=True)

    print("\nLoading large numpy files, this may take a while")
    print("-> Loading images.npy...")
    images = np.load(input_path / "images.npy")
    print("-> Loading types.npy...")
    types = np.load(input_path / "types.npy")
    # print("-> Loading patch_ids.npy...")
    # patch_ids = np.load(input_path / "patch_ids.npy")
    patch_ids = [f"{os.path.basename(input_path)}_{i}" for i in range(len(images))]

    print("\nProcess images")
    for i in tqdm(range(len(images)), total=len(images)):
        outname = f"{patch_ids[i]}.png"
        out_img = images[i]
        im = Image.fromarray(out_img.astype(np.uint8))
        im.save(output_path / "images" / outname)

    cell_count = {} # create a dictionary to store cell count for each patch
    save_types = {} # create a dictionary to store type for each patch

    print("\nProcess masks")
    mask_shape = (256, 256, len(list_cat) + 1)
    chunk_size = 20000
    number_of_chunks = ceil(len(patch_ids) / chunk_size)
    chunk_generator = load_sparse_3d_masks_chunked(input_path / "masks.npz", mask_shape, chunk_size=chunk_size)

    for chunk_idx, masks_chunk in enumerate(chunk_generator):
        
        for i, mask in tqdm(enumerate(masks_chunk), total=len(masks_chunk), desc=f"Chunk {chunk_idx+1}/{number_of_chunks}"):
            patch_idx = chunk_idx * chunk_size + i
            outname = f"{patch_ids[patch_idx]}.npz"
            type = types[patch_idx]

            # store cell count for each class for the given patch
            class_cell_count = {} 
            for j, class_name in enumerate(list_cat):
                class_cell_count[class_name] = len(np.unique(mask[:, :, j]))-1
            cell_count[f"{patch_ids[patch_idx]}.png"] = class_cell_count

            # store type for the given patch
            save_types[f"{patch_ids[patch_idx]}.png"] = type

            # need to create instance map and type map with shape 256x256
            inst_map = np.zeros((256, 256))
            num_nuc = 0
            for j in range(len(list_cat)):
                # copy value from new array if value is not equal 0
                layer_res = remap_label(mask[:, :, j])
                # inst_map = np.where(mask[:,:,j] != 0, mask[:,:,j], inst_map)
                inst_map = np.where(layer_res != 0, layer_res + num_nuc, inst_map)
                num_nuc = num_nuc + np.max(layer_res)
            inst_map = remap_label(inst_map)

            type_map = np.zeros((256, 256)).astype(np.int32)
            for j in range(len(list_cat)):
                layer_res = ((j + 1) * np.clip(mask[:, :, j], 0, 1)).astype(np.int32)
                type_map = np.where(layer_res != 0, layer_res, type_map)

            save_sparse_maps_single_file(inst_map, type_map, output_path, outname)

            # # Save also last layer with cell_ids as npz
            # mask_cell_ids = csr_matrix(mask[:, :, -1])
            # save_npz(output_path / "masks_cell_ids_nuclei" / outname, mask_cell_ids)

            # Delete maps after saving to free memory
            del inst_map
            del type_map
            # del mask_cell_ids
            gc.collect()  # Manually run garbage collection
        
        # Clear the current chunk's data from memory
        del masks_chunk
        gc.collect()  # Force garbage collection to free up memory
    
    # save cell count and type for each patch
    print("\nSave cell count and type for each patch")

    cell_count = pd.DataFrame(cell_count).T
    cell_count = cell_count.rename_axis("Image").reset_index()
    cell_count.to_csv(output_path / "cell_count.csv", sep=',', index=False)

    save_types = pd.DataFrame(save_types.items(), columns=["img", "type"])
    save_types.to_csv(output_path / "types.csv", sep=',', index=False)

    print("\nDone.")


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Prepare dataset by converting and resorting files",
)
parser.add_argument(
    "--input_path",
    type=str,
    #default="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/patches_xenium/heart_s0",
    #default="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/ds_slides_cat/ct_1/heart_s0",   # Comment part for images if already done during check align + comment part for masks_cell_ids_nuclei
    default="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/ds_slides_cat/pannuke/fold2",   # Comment part for masks_cell_ids_nuclei + for patch_ids use patch_ids = [f"{os.path.basename(input_path)}_{i}" for i in range(len(images))]
    help="Input path of the original PanNuke dataset"
)
parser.add_argument(
    "--output_path",
    type=str,
    #default="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/apply_cellvit/prepared_patches_xenium/heart_s0",
    #default="/Volumes/DD_FGS/MICS/data_HE2CellType/HE2CT/prepared_datasets_cat/ct_1/heart_s0",
    default="/Volumes/DD_FGS/MICS/data_HE2CellType/HE2CT/prepared_datasets_cat/pannuke/fold2",
    help="Output path to store the processed PanNuke dataset"
)

parser.add_argument(
    "--list_cat",
    nargs='+',
    type=str,
    default=['Neoplastic','Inflammatory','Connective','Dead','Epithelial'],
    #default=["T_NK", "B_Plasma", "Myeloid", "Blood_vessel", "Fibroblast_Myofibroblast", "Epithelial", "Specialized", "Melanocyte", "Dead"],
    help="List that contains the name of the categories (cell types), e.g. ['Neoplastic','Inflammatory','Connective','Dead','Epithelial']"
)

if __name__ == "__main__":
    opt = parser.parse_args()
    configuration = vars(opt)

    input_path = Path(configuration["input_path"])
    output_path = Path(configuration["output_path"])
    list_cat = configuration["list_cat"]

    process_ds(input_path, output_path, list_cat)
