'''Zip datasets'''

import os
import zipfile
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import shutil



def handle_remove_error(func, path, exc_info):
    # Only ignore FileNotFoundError, re-raise any other exceptions
    if isinstance(exc_info[1], FileNotFoundError):
        print(f"File not found, skipping: {path}")
    else:
        raise exc_info[1]


def zip_folder(folder_path: Path, output_path: Path):
    """Zip the contents of a folder with a progress bar.

    Args:
        folder_path (Path): Path to the folder to be zipped.
        output_path (Path): Path to the output zip file.
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

        files = [Path(root) / file for root, _, files in os.walk(folder_path) for file in files]

        with tqdm(total=len(files), desc=f"Zipping {folder_path.name}", unit="file") as pbar:
            for file_path in files:
                zipf.write(file_path, file_path.relative_to(folder_path))
                pbar.update(1)


def create_zip_imgs_or_id(dataset_path: Path, slide_ids: list, done_path: Path, name_zip: str, overwrite_zip: bool = False):
    """Create zip for images or masks_cell_ids_nuclei for a given list of slide_ids"""

    print(f"List of slides: {slide_ids}\n")
    output_zip_path = dataset_path / name_zip
    
    if output_zip_path.exists():
        print(f"\nWARNING: The output file '{output_zip_path}' already exists.")
        
        if overwrite_zip or not sys.stdin.isatty():
            print("Forcing overwrite/fresh recreation as requested or running in non-interactive shell.\n")
            output_zip_path.unlink()
        else:
            action = input("Do you want to (D)elete and recreate it, (A)ppend to it, or (C)ancel the operation? [D/A/C]: ").strip().upper()
            if action == 'D':
                print("Deleting the existing file and starting fresh.\n")
                output_zip_path.unlink()
            elif action == 'A':
                print("Appending new slide IDs to the existing file.\n")
            else:
                print("Operation canceled by the user.")
                return





def create_zip_imgs_or_id(dataset_path: Path, slide_ids: list, done_path: Path, name_zip: str):
    """Create zip for images or masks_cell_ids_nuclei for a given list of slide_ids"""

    print(f"List of slides: {slide_ids}\n")
    
    # Determine output zip file path
    output_zip_path = dataset_path / name_zip
    
    # Check if the output zip file already exists
    if output_zip_path.exists():
        print(f"\nWARNING: The output file '{output_zip_path}' already exists.")
        action = input("Do you want to (D)elete and recreate it, (A)ppend to it, or (C)ancel the operation? [D/A/C]: ").strip().upper()
        
        if action == 'D':
            print("Deleting the existing file and starting fresh.\n")
            output_zip_path.unlink()  # Delete the existing file
        elif action == 'A':
            print("Appending new slide IDs to the existing file.\n")
        else:
            print("Operation canceled by the user.")
            return
    
    # Process each slide
    for i, slide_id in enumerate(slide_ids):
        
        slide_zip_path = done_path / slide_id / name_zip
        
        if not slide_zip_path.exists():
            print(f"Image zip not found for slide {slide_id}: {slide_zip_path}")
            continue
        
        try:
            # Read images from the slide's zip file
            with zipfile.ZipFile(slide_zip_path, 'r') as input_slide_zip, \
                 zipfile.ZipFile(output_zip_path, 'a', zipfile.ZIP_DEFLATED) as zip_writer:
                
                for img in tqdm(input_slide_zip.namelist(), desc=f"Slide {slide_id} ({i+1}/{len(slide_ids)})"):
                    try:
                        # Write each image to the final zip with a prefixed name
                        with input_slide_zip.open(img) as source_img:
                            with zip_writer.open(f"{slide_id}_{img}", 'w') as target_img:
                                shutil.copyfileobj(source_img, target_img)
                    except Exception as img_error:
                        print(f"Error processing image {img} in slide {slide_id}: {img_error}")
        
        except Exception as slide_error:
            print(f"Error processing slide {slide_id}: {slide_error}")








def create_zip_labels(dataset_path: Path, patch_metrics_path: Path, output_path: Path):
    """Create zip for labels for a given cell_cat_id"""

    # Check if final folder already exists, and if yes then stop everything
    if output_path.exists():
        print(f"[WARNING] Output folder already exists: {output_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    list_slides = [item.name for item in dataset_path.iterdir() if item.is_dir() and item.name != "ALL"]
    print(f"list_slides: {list_slides}")

    # Process each slide
    for i, slide_id in enumerate(list_slides):

        print(f"\nProcessing slide {slide_id} ({i+1}/{len(list_slides)})")

        # Read cell_count.csv and types.csv
        cell_count_path = dataset_path / f"{slide_id}/cell_count.csv"
        type_path = dataset_path / f"{slide_id}/types.csv"
        slide_patch_metrics_path = patch_metrics_path / f"{slide_id}/patch_metrics.csv"

        cell_count_df = pd.read_csv(cell_count_path)
        type_df = pd.read_csv(type_path)
        patch_metrics_df = pd.read_csv(slide_patch_metrics_path)

        # Update the patch id in the cell_count and types CSVs to integrate the slide_id information
        cell_count_df['Image'] = cell_count_df['Image'].map(lambda x: f"{slide_id}_{x}")
        type_df['img'] = type_df['img'].map(lambda x: f"{slide_id}_{x}")
        patch_metrics_df['patch_id'] = patch_metrics_df['patch_id'].map(lambda x: f"{slide_id}_{x}.png")

        # Save updated CSVs into the output folder
        # Keep header only for the first slide_id
        cell_count_df.to_csv(output_path / "cell_count.csv", mode='a', header=i==0, index=False)
        type_df.to_csv(output_path / "types.csv", mode='a', header=i==0, index=False)
        patch_metrics_df.to_csv(output_path / "patch_metrics.csv", mode='a', header=i==0, index=False)
        print("Updated CSVs (cell_count, types, patch_metrics) saved.")


        ### Make zip labels with renamed filenames ###

        # Path for final labels zip file
        output_labels_zip_path = output_path / "labels.zip"

        with zipfile.ZipFile(output_labels_zip_path, 'a', zipfile.ZIP_DEFLATED) as lbl_zip_writer:

            # Zip images and labels with renamed filenames
            for _, row in tqdm(cell_count_df.iterrows(), total=len(cell_count_df), desc=f"Zipping"):
                
                # img is for original images, patch_id is for renamed images / remove also .png extension
                img, patch_id = row['Image'].replace(f"{slide_id}_", ""), row['Image']
                img, patch_id = img.replace(".png", ""), patch_id.replace(".png", "")

                # Path for label input files
                lbl_path = dataset_path / f"{slide_id}/labels/{img}.npz"

                # Add label to zip with renamed filename
                if lbl_path.exists():
                    lbl_zip_writer.write(lbl_path, arcname=f"{patch_id}.npz")
                else:
                    print(f"Label not found: {lbl_path}")





def main(args):

    if args.step == "convert_only":
        print(f"\n==== Processing slide {os.path.basename(args.dataset_path)} ====")
        print("\n** Convert_only step / Creating zip files for images and labels folders **\n")
        create_zips_imgs_labels(args.dataset_path, args.macenko)
        print("\nDone")
    

    elif args.step == "zip_images_training":
        print("\n** Zip_images_training step / Grouping all images into final zip **\n")
        args.dataset_path.mkdir(parents=True, exist_ok=True)
        name_zip = "images_macenko.zip" if args.macenko else "images.zip"
        create_zip_imgs_or_id(args.dataset_path, args.slide_ids, args.done_images_path, name_zip)
        print("\nDone.")

    
    elif args.step == "zip_masks_cell_ids_nuclei_training":
        print("\n** Zip_masks_cell_ids_nuclei_training step / Grouping all masks into final zip **\n")
        args.dataset_path.mkdir(parents=True, exist_ok=True)
        name_zip = "masks_cell_ids_nuclei.zip"
        create_zip_imgs_or_id(args.dataset_path, args.slide_ids, args.done_masks_cell_ids_nuclei_path, name_zip)
        print("\nDone.")


    elif args.step == "prepare_cell_cat_id":

        print("\n** Prepare_cell_cat_id step / Grouping all labels for a given cell_cat_id into final zip **\n")
        print(f"Processing dataset: {os.path.basename(args.dataset_path)}")
        
        # # Clear previous attempts by removing output directory if it exists
        # output_path = args.dataset_path / "ALL"
        # if output_path.exists():
        #     print(f"[info] Removing previous output directory ALL")
        #     shutil.rmtree(output_path, onerror=handle_remove_error)

        output_path = args.dataset_path / "ALL"
        create_zip_labels(args.dataset_path, args.patch_metrics_path, output_path)
        print("\nDone")
    
    else:
        print(f"\n** Unknown step: {args.step} **\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and zip datasets.")

    parser.add_argument(
        "--step", 
        type=str, 
        choices=['convert_only', 'zip_images_training', 'zip_masks_cell_ids_nuclei_training', 'prepare_cell_cat_id'], 
        default="convert_only", 
        help='Step to perform'
    )
    
    parser.add_argument(
        "--dataset_path", 
        type=Path, 
        required=True, 
        help="Path to the active dataset folder (e.g., path/to/prepared_datasets_cat)."
    )

    parser.add_argument(
        "--done_images_path", 
        type=Path, 
        default=None, 
        help="Path to folder containing slide_id subdir with images zip."
    )

    parser.add_argument(
        "--done_masks_cell_ids_nuclei_path", 
        type=Path, 
        default=None, 
        help="Path to folder containing slide_id subdir with masks_cell_ids_nuclei zip."
    )
    
    parser.add_argument(
        "--slide_ids", 
        type=str, 
        nargs='+', 
        default=[], 
        help="List of slide_ids to add in the global images zip (e.g., heart_s0)."
    )

    parser.add_argument(
        "--patch_metrics_path", 
        type=Path, 
        default=None, 
        help="Path to folder containing slide_id subdir with patch_metrics.csv file."
    )

    parser.add_argument(
        "--macenko", 
        action="store_true", 
        help="Using folder with images after Macenko normalization instead of original images"
    )
    parser.add_argument(
    "--overwrite_zip", 
    action="store_true", 
    help="Force overwrite of existing zip output without asking input confirmation."
    )

    args = parser.parse_args()

    if args.step == "zip_images_training" and not args.done_images_path:
        parser.error("--done_images_path is required when --step is 'zip_images_training'")
    if args.step == "zip_masks_cell_ids_nuclei_training" and not args.done_masks_cell_ids_nuclei_path:
        parser.error("--done_masks_cell_ids_nuclei_path is required when --step is 'zip_masks_cell_ids_nuclei_training'")
    if args.step == "prepare_cell_cat_id" and not args.patch_metrics_path:
        parser.error("--patch_metrics_path is required when --step is 'prepare_cell_cat_id'")

    main(args)
