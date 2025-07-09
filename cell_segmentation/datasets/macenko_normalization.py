"""Perform Macenko normalization on H&E patches for training, using authors' values for target"""

import os
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
from typing import Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys

# Add the root directory of our project to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.patch_extraction.src.utils.patch_util import macenko_normalization



def normalize_and_save_image(image_path: Path, output_folder: Path, normalization_vector_path: Union[Path, str]):
    
    # Load the image
    patch = np.array(Image.open(image_path).convert("RGB"))
    if patch is None:
        print(f"Warning: Unable to read {image_path}")
        return

    # Normalize the patch using Macenko normalization
    normalized_patch, _, _ = macenko_normalization(
        [patch],
        normalization_vector_path=normalization_vector_path,
    )
    normalized_patch = normalized_patch[0]

    # Save to output folder with the same filename
    output_path = output_folder / image_path.name
    Image.fromarray(normalized_patch).save(output_path)



def main(step: str, prepared_ds_folder: str, slide_id: str, normalization_vector_path: Union[Path, str] = None):

    print(f"\n==== Processing {slide_id} ====")

    if step=="check_align":
        input_folder = Path(f"{prepared_ds_folder}/{slide_id}/fold2/images")
        output_folder = Path(f"{prepared_ds_folder}/{slide_id}/fold2/images_macenko")
        output_folder.mkdir(parents=True, exist_ok=True)        

    elif step=="main_training":
        input_folder = Path(f"{prepared_ds_folder}/{slide_id}/images")
        output_folder = Path(f"{prepared_ds_folder}/{slide_id}/images_macenko")
        output_folder.mkdir(parents=True, exist_ok=True)
    
    else:
        raise ValueError("Invalid step argument")

    # List all .png files in the input directory
    image_paths = list(input_folder.glob("*.png"))
    print(f"Found {len(image_paths)} images in {input_folder}")

    print("Processing images...")
    # Process each image with concurrent processing for performance
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(normalize_and_save_image, img_path, output_folder, normalization_vector_path)
            for img_path in image_paths
        ]
        # Use tqdm to show progress as futures complete
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass  # Simply advance the progress bar for each completed future
    print("Done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Macenko normalization to H&E patches before training")
    
    parser.add_argument("--step", type=str, default="check_align", choices=["check_align", "main_training"], help="If the normalization is for checking alignment or main training")
    parser.add_argument("--prepared_ds_folder", type=str, default="/Volumes/DD_FGS/MICS/data_HE2CellType/CT_DS/check_align_patches/apply_cellvit/prepared_patches_xenium", help="Path to the prepared dataset folder")
    parser.add_argument("--slide_id", type=str, default="heart_s0", help="Slide ID")
    parser.add_argument("--normalization_vector_path", type=str, default=None, help="Path to the normalization vector JSON file")
    
    args = parser.parse_args()
    main(args.step, args.prepared_ds_folder, args.slide_id, args.normalization_vector_path)
