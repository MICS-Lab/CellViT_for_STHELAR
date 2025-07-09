"""Perform multi-reference Macenko normalization on H&E patches for training"""

import os
import sys
import argparse
import zipfile
import io
import math
import random
import numpy as np
import gc

from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchstain

from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.colors as mcolors

def save_bad_plot(orig_img_bytes, norm_img_bytes, name, bad_dir):
    """
    Saves a side-by-side comparison of original vs. normalized patch
    into a 'bad' folder for manual QA.
    """
    os.makedirs(bad_dir, exist_ok=True)
    orig_img = Image.open(io.BytesIO(orig_img_bytes))
    norm_img = Image.open(io.BytesIO(norm_img_bytes))
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(orig_img)
    axs[0].set_title("Original")
    axs[0].axis("off")
    axs[1].imshow(norm_img)
    axs[1].set_title("Normalized")
    axs[1].axis("off")
    fig.suptitle(f"BAD normalization: {name}")
    out_path = os.path.join(bad_dir, f"bad_{os.path.splitext(os.path.basename(name))[0]}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def compute_mean_saturation(norm_img_bytes):
    """
    Compute the mean saturation of a normalized patch (in [0,1] range).
    """
    norm_img = Image.open(io.BytesIO(norm_img_bytes))
    norm_np = np.array(norm_img).astype(np.float32) / 255.0
    hsv_img = mcolors.rgb_to_hsv(norm_np)
    return hsv_img[..., 1].mean()

def plot_saturation_distribution(saturations, out_dir):
    """
    Plots and saves the distribution (histogram) of mean saturations across all normalized patches.
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.hist(saturations, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Mean Saturation (H&E Patches)")
    plt.xlabel("Mean Saturation [0, 1]")
    plt.ylabel("Count")
    plot_path = os.path.join(out_dir, "saturation_distribution.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved saturation distribution plot to: {plot_path}")

global_zip = None
def get_zip_file(zip_path):
    global global_zip
    if global_zip is None:
        global_zip = zipfile.ZipFile(zip_path, 'r')
    return global_zip

def process_image(file_name, zip_path, normalizer_state):
    """
    Worker function to read image from zip, reconstruct the normalizer,
    and apply multi-reference Macenko normalization.
    """
    zf = get_zip_file(zip_path)
    try:
        orig_img_bytes = zf.read(file_name)
        img = Image.open(io.BytesIO(orig_img_bytes))
        img = np.array(img).astype(np.uint8)
    except Exception as e:
        print(f"Error loading image {file_name}: {e}")
        return file_name, None, None

    T_local = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])
    img_tensor = T_local(img).cpu()

    local_normalizer = torchstain.normalizers.MultiMacenkoNormalizer(norm_mode=normalizer_state['norm_mode'])
    local_normalizer.HERef = normalizer_state['HERef'].clone()
    local_normalizer.maxCRef = normalizer_state['maxCRef'].clone()

    norm, _, _ = local_normalizer.normalize(img_tensor, stains=True)
    norm = norm.clamp(0, 255).to(torch.uint8).cpu().numpy()
    norm_img = Image.fromarray(norm)

    out_buffer = io.BytesIO()
    norm_img.save(out_buffer, format='PNG')
    norm_img_bytes = out_buffer.getvalue()

    return file_name, orig_img_bytes, norm_img_bytes

def create_target_mosaic(img_zip_path, target_names):
    """
    Build one mosaic from multiple patches. Returns (mosaic PIL.Image, list[PIL.Image]).
    """
    patches = []
    with zipfile.ZipFile(img_zip_path, 'r') as zf:
        for name in target_names:
            data = zf.read(name)
            img = Image.open(io.BytesIO(data))
            img_np = np.array(img).astype(np.uint8)
            patches.append(Image.fromarray(img_np))
    if not patches:
        raise ValueError("No target images loaded.")
    # Assume all patches have the same dimensions.
    w, h = patches[0].size
    n = len(patches)
    grid_size = math.ceil(math.sqrt(n))
    mosaic_w = grid_size * w
    mosaic_h = grid_size * h
    mosaic = Image.new('RGB', (mosaic_w, mosaic_h), (255, 255, 255))
    for idx, patch in enumerate(patches):
        row = idx // grid_size
        col = idx % grid_size
        x = col * w
        y = row * h
        mosaic.paste(patch, (x, y))
    return mosaic, patches

def create_checking_plots_from_sample(sample_results, checking_dir):
    """
    Side-by-side checking plots for a random sample: (original vs. normalized).
    """
    os.makedirs(checking_dir, exist_ok=True)
    for (name, orig_bytes, norm_bytes) in tqdm(sample_results):
        try:
            orig_img = Image.open(io.BytesIO(orig_bytes))
            norm_img = Image.open(io.BytesIO(norm_bytes))
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(np.array(orig_img).astype(np.uint8))
            axs[0].set_title("Original")
            axs[0].axis("off")
            axs[1].imshow(np.array(norm_img).astype(np.uint8))
            axs[1].set_title("Normalized")
            axs[1].axis("off")
            fig.suptitle(f"Checking Plot: {name}")
            plot_path = os.path.join(checking_dir, f"check_{os.path.splitext(os.path.basename(name))[0]}.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"Error creating checking plot for {name}: {e}")

def main(args):
    """
    Main pipeline:
      - Build reference mosaics per slide.
      - Fit the MultiMacenkoNormalizer using one mosaic per slide.
      - Process images in parallel with memory-efficient reservoir sampling.
      - Record names for all failed/filtered patches.
    """
    if not args.target_img_names:
        print("Error: Must provide a comma-separated list of target image names.")
        sys.exit(1)
    all_ref_names = [n.strip() for n in args.target_img_names.split(',') if n.strip()]
    if not all_ref_names:
        print("Error: no valid reference patches provided.")
        sys.exit(1)

    img_zip_path = args.img_zip_path
    checking_dir = os.path.join(os.path.dirname(img_zip_path), "checking_macenko_normalization")

    with zipfile.ZipFile(img_zip_path, 'r') as zf:
        all_files = set(zf.namelist())
        file_list = [f for f in all_files if f.lower().endswith('.png')]
    missing = [n for n in all_ref_names if n not in all_files]
    if missing:
        print(f"Error: Some references not found in zip: {missing}")
        sys.exit(1)

    print(f"--> Found {len(file_list)} total PNG images in {img_zip_path}.")
    print(f"--> Number of reference patches from user: {len(all_ref_names)}")

    # ### SAMPLING FOR DEBUGGING / TOREMOVE ###
    # file_list = random.sample(file_list, min(500, len(file_list)))
    # print(f"--> [WARNING] Using {len(file_list)} images (sample) for debugging.")
    # #########################################

    # Group references by slide.
    slide2refs = {}
    for ref_name in all_ref_names:
        slide_id = ref_name.rsplit('_', 1)[0]
        slide2refs.setdefault(slide_id, []).append(ref_name)

    # Build one mosaic per slide and use it as the reference.
    references_tensors = []
    os.makedirs(checking_dir, exist_ok=True)
    print("\n* Building target mosaics for all slides ...")
    T_local = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])
    for slide_id, ref_names in slide2refs.items():
        print(f"Creating mosaic for slide: {slide_id} with {len(ref_names)} patches...")
        mosaic_img, patches = create_target_mosaic(img_zip_path, ref_names)
        plt.figure(figsize=(8, 8))
        plt.imshow(mosaic_img)
        plt.title(f"Target Mosaic: slide {slide_id}")
        plt.axis("off")
        mosaic_path = os.path.join(checking_dir, f"target_mosaic_{slide_id}.png")
        mosaic_img.save(mosaic_path, format='PNG')
        plt.close()
        # Convert the entire mosaic into a tensor (this becomes the unique reference for the slide)
        mosaic_tensor = T_local(np.array(mosaic_img).astype(np.uint8)).cpu()
        references_tensors.append(mosaic_tensor)

    print(f"\nNumber of different slides for reference: {len(slide2refs)} / Number of reference images: {len(references_tensors)}")
    print("All target mosaics are displayed. Please inspect them.")
    print("When you close the plots, you will be asked for final confirmation.")
    plt.show()
    answer = input("Are all target mosaics correct? (yes/no): ").strip().lower()
    if answer != "yes":
        print("Aborting as per user request.")
        sys.exit(0)

    print(f"\n* Fitting multi-reference Macenko normalizer (avg-post mode) using {len(references_tensors)} reference mosaics ...")
    multi_normalizer = torchstain.normalizers.MultiMacenkoNormalizer(norm_mode='avg-post')
    multi_normalizer.fit(references_tensors, Io=240, alpha=1, beta=0.15)
    print("Normalizer fitting complete.")
    del references_tensors
    gc.collect()

    normalizer_state = {
        'norm_mode': multi_normalizer.norm_mode,
        'HERef': multi_normalizer.HERef.cpu(),
        'maxCRef': multi_normalizer.maxCRef.cpu()
    }

    out_zip_path = os.path.join(os.path.dirname(img_zip_path), "images_macenko.zip")
    out_zip = zipfile.ZipFile(out_zip_path, 'w', compression=zipfile.ZIP_DEFLATED)
    all_saturations = []
    CHUNK_SIZE = 100000

    # Reservoir for good patch checking plots.
    reservoir = []
    sample_count = 0

    # Counters for errors and filtering.
    failed_imgs_nb = 0
    filtered_imgs_nb = 0

    # Lightweight list of patch names that failed or were filtered.
    bad_patch_names = []

    # Reservoir for bad patch examples (using reservoir sampling to limit memory usage).
    bad_reservoir = []
    bad_count = 0

    # Process images in chunks.
    for chunk_start in range(0, len(file_list), CHUNK_SIZE):
        chunk_end = chunk_start + CHUNK_SIZE
        subset = file_list[chunk_start:chunk_end]
        print(f"\n** Processing chunk {chunk_start}-{chunk_end} / {len(file_list)} (size={len(subset)}) with {args.num_workers} workers **")
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_image, fname, img_zip_path, normalizer_state): fname
                for fname in subset
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Normalizing chunk"):
                fname = futures[future]
                try:
                    out_fname, orig_bytes, norm_bytes = future.result()
                    if norm_bytes is not None and orig_bytes is not None:
                        # Compute saturation only once.
                        saturation = compute_mean_saturation(norm_bytes)
                        if saturation < 0.05:
                            filtered_imgs_nb += 1
                            bad_patch_names.append(out_fname)
                            bad_count += 1
                            if len(bad_reservoir) < args.num_bad_plots:
                                bad_reservoir.append((out_fname, orig_bytes, norm_bytes))
                            else:
                                r = random.randint(0, bad_count - 1)
                                if r < args.num_bad_plots:
                                    bad_reservoir[r] = (out_fname, orig_bytes, norm_bytes)
                            continue  # Skip writing this patch.
                        all_saturations.append(saturation)
                        out_zip.writestr(out_fname, norm_bytes)
                        sample_count += 1
                        if len(reservoir) < args.num_checking_plots:
                            reservoir.append((out_fname, orig_bytes, norm_bytes))
                        else:
                            r = random.randint(0, sample_count - 1)
                            if r < args.num_checking_plots:
                                reservoir[r] = (out_fname, orig_bytes, norm_bytes)
                    else:
                        print(f"Skipping {fname} due to processing error.")
                        failed_imgs_nb += 1
                        bad_patch_names.append(fname)
                except Exception as exc:
                    print(f"{fname} generated an exception: {exc}")
                    failed_imgs_nb += 1
                    bad_patch_names.append(fname)
        # Workers exit at the end of each chunk.
    out_zip.close()

    print(f"\nMulti-reference Macenko normalization complete. Results saved to: {out_zip_path}")
    print(f"!!!!! Total processed (written): {sample_count}, Failed: {failed_imgs_nb}, Filtered: {filtered_imgs_nb} !!!!!")

    # Save the list of failed/filtered patch names.
    bad_list_path = os.path.join(checking_dir, "bad_normalizations", "bad_patch_names.txt")
    os.makedirs(os.path.dirname(bad_list_path), exist_ok=True)
    with open(bad_list_path, "w") as f:
        for name in bad_patch_names:
            f.write(name + "\n")
    print(f"Bad/failed patch names saved to: {bad_list_path}")

    plot_saturation_distribution(all_saturations, checking_dir)

    # Plot the reservoir of bad patches.
    if args.num_bad_plots > 0 and bad_reservoir:
        bad_dir = os.path.join(checking_dir, "bad_normalizations")
        os.makedirs(bad_dir, exist_ok=True)
        print(f"\n* Creating up to {args.num_bad_plots} bad patch plots in {bad_dir} ...")
        for (name, orig_bytes, norm_bytes) in tqdm(bad_reservoir):
            save_bad_plot(orig_bytes, norm_bytes, name, bad_dir)

    # Generate checking plots for good patches.
    if args.num_checking_plots > 0 and reservoir:
        print(f"\n* Creating up to {args.num_checking_plots} checking plots in {checking_dir} ...")
        create_checking_plots_from_sample(reservoir, checking_dir)

    print("\nDone.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply multi-reference Macenko normalization to H&E patches.")
    parser.add_argument("--img_zip_path", type=str,
                        default="/Volumes/DD_FGS/MICS/data_HE2CellType/HE2CT/prepared_datasets_cat/images.zip",
                        help="Path to the zip containing H&E images")
    parser.add_argument("--target_img_names", type=str,
                        default="breast_s0_44681.png,breast_s0_37919.png,breast_s1_76960.png,breast_s1_76791.png,breast_s3_33838.png,breast_s3_33905.png,breast_s6_27698.png,breast_s6_27673.png,cervix_s0_13514.png,cervix_s0_13512.png,colon_s1_13951.png,colon_s1_13847.png,colon_s2_19293.png,colon_s2_12819.png,heart_s0_480.png,heart_s0_5774.png,kidney_s0_6356.png,kidney_s0_6345.png,kidney_s1_2766.png,kidney_s1_4478.png,liver_s0_14362.png,liver_s0_14345.png,liver_s1_2143.png,liver_s1_9482.png,lung_s1_12337.png,lung_s1_12330.png,lung_s3_13535.png,lung_s3_14408.png,lymph_node_s0_3723.png,lymph_node_s0_3421.png,ovary_s0_19561.png,ovary_s0_15869.png,ovary_s1_15114.png,ovary_s1_15077.png,pancreatic_s0_6024.png,pancreatic_s0_6031.png,pancreatic_s1_7181.png,pancreatic_s1_5847.png,pancreatic_s2_10084.png,pancreatic_s2_10554.png,prostate_s0_21513.png,prostate_s0_18554.png,skin_s1_7447.png,skin_s1_7349.png,skin_s2_2417.png,skin_s2_17544.png,skin_s3_8415.png,skin_s3_8560.png,skin_s4_15893.png,skin_s4_15891.png,tonsil_s0_12409.png,tonsil_s0_12397.png,tonsil_s1_24106.png,tonsil_s1_24105.png",
                        help="Comma-separated list of reference patches from multiple slides.")
    parser.add_argument("--num_workers", type=int, 
                        default=5,
                        help="Number of parallel workers to use.")
    parser.add_argument("--num_checking_plots", type=int, 
                        default=50,
                        help="Number of random checking plots to generate for QC (good patches).")
    parser.add_argument("--num_bad_plots", type=int,
                        default=50,
                        help="Number of random bad patch plots to generate for manual QA.")
    args = parser.parse_args()
    main(args)
