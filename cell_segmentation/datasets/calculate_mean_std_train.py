"""Calculates mean and std for the train set for standardization in training / or pannuke dataset can also be used"""


import os
import argparse
import zipfile
import io
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted



class ImgDataset(Dataset):
    
    def __init__(self, ds_folder, data_zip_folder):
        
        self.ds_folder = ds_folder
        self.img_zip_path = os.path.join(data_zip_folder, "images.zip")

        # Load the list of image names from cell_count_train.csv
        self.img_names_list = pd.read_csv(os.path.join(self.ds_folder, "cell_count_train.csv"))['Image'].tolist()
        print(f"[INFO] Found {len(self.img_names_list)} images for train set.")

        # Verify that the listed images exist in the ZIP file
        with zipfile.ZipFile(self.img_zip_path, 'r') as img_zip:
            zip_img_names = set(f for f in img_zip.namelist() if f.endswith(".png"))
            
            missing_images = [img_name for img_name in self.img_names_list if img_name not in zip_img_names]
            if missing_images:
                print(f"[WARNING] The following images listed in the CSV were not found in the ZIP file:\n{missing_images}")

            # Retain only images present in both the CSV and ZIP file
            self.img_names_list = [img_name for img_name in self.img_names_list if img_name in zip_img_names]
            print(f"[INFO] Found {len(self.img_names_list)} images ok in the ZIP file.")

        if not self.img_names_list:
            raise ValueError("No ok image names found in the CSV file or ZIP file!")
        
        # Sort the image names for consistency
        self.img_names_list = natsorted(self.img_names_list)
        
        # Initialize as None; open in the worker process
        self.zip_file = None  


    def _init_zipfile(self):
        if self.zip_file is None:  # Open the ZIP file only if not already open
            self.zip_file = zipfile.ZipFile(self.img_zip_path, 'r')


    def load_imgfile(self, img_name: str):
        img_data = self.zip_file.read(img_name)
        img = Image.open(io.BytesIO(img_data))
        return np.array(img).astype(np.uint8)


    def __del__(self):
        if self.zip_file is not None:
            self.zip_file.close()  # Ensure the ZIP file is closed properly


    def __len__(self):
        return len(self.img_names_list)


    def __getitem__(self, idx):
        self._init_zipfile()  # Ensure the ZIP file is open in this worker
        img_name = self.img_names_list[idx]
        return self.load_imgfile(img_name)





def calculate_mean_std(data_loader):
    """Calculates mean and std for each channel over the dataset."""
    
    n_pixels = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)

    for images in tqdm(data_loader):
        images = images.numpy() / 255.0  # Convert to NumPy and scale to [0,1]
        
        # Calculate total pixels per batch for each channel
        n_pixels += images.shape[0] * images.shape[1] * images.shape[2]  # batch * height * width
        
        # Sum and squared sum per channel, across batch and spatial dimensions
        channel_sum += images.sum(axis=(0, 1, 2))
        channel_squared_sum += (images ** 2).sum(axis=(0, 1, 2))

    # Mean and std calculations
    mean = channel_sum / n_pixels
    std = np.sqrt((channel_squared_sum / n_pixels) - mean ** 2)
    return mean, std




def main(args):

    print(f"\n===== Calculating mean and std for {args.dataset_id} =====")

    ds_folder = f'/Volumes/DD1_FGS/MICS/data_HE2CellType/HE2CT/training_datasets/{args.dataset_id}'
    data_zip_folder = f'/Volumes/DD1_FGS/MICS/data_HE2CellType/HE2CT/prepared_datasets_cat'
    output_csv = f'/Volumes/DD1_FGS/MICS/data_HE2CellType/HE2CT/training_datasets/{args.dataset_id}/informations/mean_std_train.csv'

    # #### Only for pannuke dataset ###
    # # Rename also temporary the real cell_count.csv file to cell_count_train.csv in folder
    # ds_folder = f'/Volumes/DD1_FGS/MICS/data_HE2CellType/HE2CT/prepared_datasets_cat/{args.dataset_id}'
    # data_zip_folder = f'/Volumes/DD1_FGS/MICS/data_HE2CellType/HE2CT/prepared_datasets_cat/{args.dataset_id}'
    # output_csv = f'/Volumes/DD1_FGS/MICS/data_HE2CellType/HE2CT/prepared_datasets_cat/{args.dataset_id}/informations/mean_std_train.csv'
    # os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    # #################################
    
    # Load dataset and DataLoader
    print("\n** Loading dataset...")
    dataset = ImgDataset(ds_folder, data_zip_folder)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=9, pin_memory=True)

    # Calculate mean and std
    print("\n** Calculating mean and std...")
    mean, std = calculate_mean_std(data_loader)

    # Save results
    print(f"** Saving to {output_csv}...")
    df = pd.DataFrame({"channel": ["R", "G", "B"], "mean": mean, "std": std})
    print(df)
    df.to_csv(output_csv, index=False)
    print("\nDone.")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate mean and std for dataset images")

    parser.add_argument("--dataset_id", type=str, default="ds_4", help="training_dataset_id")

    args = parser.parse_args()
    main(args)
