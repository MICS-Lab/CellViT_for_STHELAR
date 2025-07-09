# Make train, valid, test splits for the Pannuke dataset

import os
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import json



def apply_grouping(df, grouping):
    """
    Applies grouping to the cell count DataFrame.
    
    Parameters:
      df (pd.DataFrame): The cell count DataFrame.
      grouping (dict): Dictionary mapping new category names to lists of original columns.
                       Example: {"Immune": ["T_NK", "B_Plasma", "Myeloid"],
                                 "Stromal": ["Blood_vessel", "Fibroblast_Myofibroblast"]}
    
    Returns:
      pd.DataFrame: The modified DataFrame with new grouped columns.
    """
    df_new = df.copy()
    for new_cat, cols in grouping.items():
        # Check that all required columns exist
        missing_cols = [c for c in cols if c not in df_new.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} are missing in df_cell_count for grouping '{new_cat}'.")
        # Sum the designated columns to create the new group
        df_new[new_cat] = df_new[cols].sum(axis=1)
        # Drop the original columns that have been grouped
        df_new.drop(columns=cols, inplace=True)
    return df_new




def make_final_df(slide_ids, path_prepared_dataset, grouping=None):

    # Load data
    print("- Loading data...")
    df_cell_count = pd.read_csv(os.path.join(path_prepared_dataset, "ALL/cell_count.csv"))
    df_types = pd.read_csv(os.path.join(path_prepared_dataset, "ALL/types.csv"))
    df_patch_metrics = pd.read_csv(os.path.join(path_prepared_dataset, "ALL/patch_metrics.csv"))

    # If grouping is provided (a dict), apply it to df_cell_count
    if grouping:
        print("- Applying grouping using dict:", grouping)
        df_cell_count = apply_grouping(df_cell_count, grouping)

    # Merge df_cell_count with df_types
    print("- Merging df_cell_count with df_types...")
    df_final = pd.merge(df_cell_count, df_types, left_on='Image', right_on='img')
    df_final.drop(columns=['Image'], inplace=True)

    # Merge with df_patch_metrics
    print("- Merging with df_patch_metrics...")
    df_final = pd.merge(df_final, df_patch_metrics, left_on='img', right_on='patch_id')
    df_final.drop(columns=['patch_id'], inplace=True)

    # Add slide_id column
    df_final['slide_id'] = df_final['img'].str.rsplit('_', n=1).str[0]

    # Keep only the chosen slide_ids
    print("Slide_ids before filtering:", df_final['slide_id'].unique())
    print(len(df_final['slide_id'].unique()))
    df_final_filtered = df_final[df_final['slide_id'].isin(slide_ids)].copy()
    print("Slide_ids after filtering:", df_final_filtered['slide_id'].unique())
    print(len(df_final_filtered['slide_id'].unique()))

    # Get the cell types columns
    cell_types_cols = list(df_cell_count.columns)
    cell_types_cols.remove('Image')

    return df_final_filtered, cell_types_cols, df_cell_count


def split_dataset(df_final_filtered, cell_types_cols, force_train_list):
    """
    Split df_final_filtered into train, valid, test sets, ensuring force_train_list
    patches always go into 'train'.
    """

    forced_train_set = set(force_train_list)

    train_patch_ids = []
    valid_patch_ids = []
    test_patch_ids = []

    for tissue in tqdm(df_final_filtered['type'].unique()):

        df_tissue = df_final_filtered[df_final_filtered['type']==tissue].copy()

        # Separate forced patches from the rest
        df_tissue_forced = df_tissue[df_tissue['img'].isin(forced_train_set)]
        df_tissue_split = df_tissue[~df_tissue['img'].isin(forced_train_set)]

        df_tissue_split['is_cat_min'] = 0
        
        valid_cell_types = df_tissue_split[cell_types_cols].sum()
        valid_cell_types = valid_cell_types[valid_cell_types > 0]
        valid_cell_types = valid_cell_types[valid_cell_types.index != "Unknown"]
        min_cell_type = valid_cell_types.idxmin()
        df_tissue_split.loc[df_tissue_split[min_cell_type] > 0, 'is_cat_min'] = 1

        print(f"\nTissue: {tissue}")
        print(f"Mimimum cell type: {min_cell_type}")
        print(df_tissue_split['is_cat_min'].value_counts())

        tissue_train_valid, tissue_test = train_test_split(df_tissue_split, test_size=0.2, random_state=42, shuffle=True, stratify=df_tissue_split[['is_cat_min']])
        tissue_train, tissue_valid = train_test_split(tissue_train_valid, test_size=0.25, random_state=42, shuffle=True, stratify=tissue_train_valid[['is_cat_min']])

        # Now combine forced patches with the newly splitted train
        print(f"Forcing the following patches to be in the train set because of Macenko normalization: {df_tissue_forced['img'].tolist()}")
        train_patch_ids.extend(df_tissue_forced['img'].tolist())
        
        train_patch_ids.extend(tissue_train['img'].tolist())
        valid_patch_ids.extend(tissue_valid['img'].tolist())
        test_patch_ids.extend(tissue_test['img'].tolist())

    df_final_filtered['set'] = 'att'
    df_final_filtered.loc[df_final_filtered['img'].isin(train_patch_ids), 'set'] = 'train'
    df_final_filtered.loc[df_final_filtered['img'].isin(valid_patch_ids), 'set'] = 'valid'
    df_final_filtered.loc[df_final_filtered['img'].isin(test_patch_ids), 'set'] = 'test'

    return df_final_filtered, train_patch_ids, valid_patch_ids, test_patch_ids



def create_plots(df_final_filtered, cell_types_cols, save_dir):

    df = df_final_filtered.copy()
    df['set'] = pd.Categorical(df['set'], categories=['train', 'valid', 'test'], ordered=True)

    # FIGURE 1: Tissues
    print("- Making fig1...")
    fig1, axs = plt.subplots(1, 2, figsize=(20, 7))

    # Subplot 1: Patch count per tissue for each fold
    sns.countplot(data=df, x="type", hue="set", order=df['type'].unique(), ax=axs[0])
    axs[0].set_title("Patch count per tissue for each fold")
    axs[0].set_xlabel("Tissue type")
    axs[0].set_ylabel("Patch count")
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=3)  # Legend below plot
    
    # Subplot 2: Percentage of tissue type for each fold
    n_colors = len(df['type'].unique())
    hues = np.linspace(0, 1, n_colors, endpoint=False)  # Evenly distribute hues
    distinguishable_palette = [sns.hls_palette(1, l, s)[0] for l, s in zip(hues, [0.6] * n_colors)]  # Varying hue, constant saturation
    random.shuffle(distinguishable_palette) # Shuffle the palette

    tissue_counts = df.groupby(['set', 'type'], observed=True).size().reset_index(name='count')
    tissue_totals = tissue_counts.groupby('set', observed=True)['count'].transform('sum')
    tissue_counts['percent'] = (tissue_counts['count'] / tissue_totals) * 100
    pivot_tissue = tissue_counts.pivot(index="set", columns="type", values="percent").fillna(0)
    pivot_tissue.plot(kind="bar", stacked=True, ax=axs[1], color=distinguishable_palette)
    axs[1].set_title("Percentage of tissue type for each fold")
    axs[1].set_xlabel("Fold")
    axs[1].set_ylabel("Percentage of patches")
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)  # Legend below plot

    plt.tight_layout()
    plt.savefig(f"{save_dir}/tissues_distribution.png")

    # FIGURE 2: Slide_ids
    print("- Making fig2...")
    fig2, axs = plt.subplots(1, 2, figsize=(20, 7))

    # Subplot 1: Patch count per slide_id for each fold
    sns.countplot(data=df, x="slide_id", hue="set", order=df['slide_id'].unique(), ax=axs[0])
    axs[0].set_title("Patch count per slide_id for each fold")
    axs[0].set_xlabel("Slide ID")
    axs[0].set_ylabel("Patch Count")
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.38), ncol=3)  # Legend below plot

    # Subplot 2: Percentage for slide_id for each fold
    n_colors = len(df['slide_id'].unique())
    hues = np.linspace(0, 1, n_colors, endpoint=False)  # Evenly distribute hues
    distinguishable_palette = [sns.hls_palette(1, l, s)[0] for l, s in zip(hues, [0.6] * n_colors)]  # Varying hue, constant saturation
    random.shuffle(distinguishable_palette) # Shuffle the palette

    slide_counts = df.groupby(['set', 'slide_id'], observed=True).size().reset_index(name='count')
    slide_totals = slide_counts.groupby('set', observed=True)['count'].transform('sum')
    slide_counts['percent'] = (slide_counts['count'] / slide_totals) * 100
    pivot_slide = slide_counts.pivot(index="set", columns="slide_id", values="percent").fillna(0)  # Corrected pivot
    pivot_slide.plot(kind="bar", stacked=True, ax=axs[1], color=distinguishable_palette)
    axs[1].set_title("Percentage for slide_id for each fold")
    axs[1].set_xlabel("Fold")
    axs[1].set_ylabel("Percentage of patches")
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)  # Legend below plot

    plt.tight_layout()
    plt.savefig(f"{save_dir}/slide_ids_distribution.png")

    # FIGURE 3: Cell types
    print("- Making fig3...")
    fig3, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Cell types count for each fold
    cell_counts = df.groupby(['set'], observed=True)[cell_types_cols].sum().reset_index()
    ax1 = cell_counts.set_index("set")[cell_types_cols].plot(kind="bar", stacked=True, ax=axs[0], colormap="tab20", legend=False)
    axs[0].set_title("Cell types count for each fold")
    axs[0].set_xlabel("Fold")
    axs[0].set_ylabel("Cell count")

    # Subplot 2: Cell types percentage for each fold
    cell_totals = cell_counts[cell_types_cols].sum(axis=1)
    cell_percentages = cell_counts[cell_types_cols].div(cell_totals, axis=0) * 100
    cell_percentages['set'] = cell_counts['set']
    
    ax2 = cell_percentages.set_index("set").plot(kind="bar", stacked=True, ax=axs[1], colormap="tab20", legend=False)
    axs[1].set_title("Cell types percentage for each fold")
    axs[1].set_xlabel("Fold")
    axs[1].set_ylabel("Cell percentage")

    # Create a common legend from the first plot
    handles, labels = ax1.get_legend_handles_labels()
    fig3.legend(handles, labels, loc='lower center', ncol=4, title="Cell Types")

    # Adjust layout to leave space for the legend
    plt.tight_layout(rect=[0, 0.18, 1, 1])  # Leave space at the bottom for the legend
    plt.savefig(f"{save_dir}/cell_types_distribution.png")




def main(args):

    # Parse grouping argument if provided
    grouping_dict = None
    if args.grouping:
        grouping_dict = json.loads(args.grouping)

    # Create the output directory
    save_dir = os.path.join(args.output_dir, args.dataset_id, "informations")
    os.makedirs(save_dir, exist_ok=True)

    # Make the final DataFrame, keeping only the chosen slide_ids
    print("\n** Making the final DataFrame...")
    df_final, cell_types_cols, df_cell_count = make_final_df(args.slide_ids, args.path_prepared_dataset, grouping=grouping_dict)

    # Remove Macenko filtered images if needed
    if args.macenko:
        print("\n** Removing Macenko filtered images...")
        macenko_bad_path = os.path.join(os.path.dirname(args.path_prepared_dataset), "checking_macenko_normalization/bad_normalizations/bad_patch_names.txt")
        if os.path.exists(macenko_bad_path):
            with open(macenko_bad_path, 'r') as f:
                bad_images = set(line.strip() for line in f if line.strip())
            print(f"Found {len(bad_images)} bad patches from macenko normalization.")
            initial_count = len(df_final)
            df_final = df_final[~df_final['img'].isin(bad_images)].copy()
            df_cell_count = df_cell_count[~df_cell_count['Image'].isin(bad_images)].copy()
            print(f"Removed {initial_count - len(df_final)} patches.")
        else:
            raise FileNotFoundError(f"You chose to remove Macenko filtered images but the file {macenko_bad_path} does not exist.")

    # Filter patches based on the chosen metric and range
    print(f"\n** Filtering patches based on the metric {args.he_patches_selection[0]} and range {args.he_patches_selection[1:]}...")
    metric, min_val, max_val = args.he_patches_selection
    df_final_filtered = df_final[(df_final[metric] >= min_val) & (df_final[metric] <= max_val)].copy()
    print(f"Kept {len(df_final_filtered)} patches over {len(df_final)}.")
    print(f"\nNumber of patches per slide_id BEFORE:\n {df_final.groupby('slide_id', observed=True).size()}")
    print(f"\nNumber of patches per slide_id AFTER:\n {df_final_filtered.groupby('slide_id', observed=True).size()}")

    # Split the dataset into train, valid, test
    print("\n** Splitting the dataset into train, valid, test...")
    force_train_list = [p.strip() for p in args.force_train_patches.split(',') if p.strip()]
    df_final_filtered, train_patch_ids, valid_patch_ids, test_patch_ids = split_dataset(df_final_filtered, cell_types_cols, force_train_list)
    print(f"Splitting results:\n {df_final_filtered.groupby(['slide_id', 'set'], observed=True).size()}")

    # Create plots
    print("\n** Creating plots...")
    create_plots(df_final_filtered, cell_types_cols, save_dir)

    # Save the final DataFrame
    print("\n** Saving the final DataFrame...")
    print(df_final_filtered.head())
    df_final_filtered.to_csv(f"{save_dir}/infos_{args.dataset_id}.csv", index=False)

    # Saving the cell count for each set
    print("\n** Saving the cell count for each set...")
    df_cell_count[df_cell_count['Image'].isin(train_patch_ids)].to_csv(os.path.join(args.output_dir, args.dataset_id, 'cell_count_train.csv'), index=False)
    df_cell_count[df_cell_count['Image'].isin(valid_patch_ids)].to_csv(os.path.join(args.output_dir, args.dataset_id, 'cell_count_valid.csv'), index=False)
    df_cell_count[df_cell_count['Image'].isin(test_patch_ids)].to_csv(os.path.join(args.output_dir, args.dataset_id, 'cell_count_test.csv'), index=False)

    print("Done.")
  


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Make train, valid, test splits for the Pannuke dataset")
    
    parser.add_argument("--slide_ids", type=str, nargs='+', default=["TEST_heart_s0", "TEST_breast_s0"], help="List of slide ids to keep for the given dataset_id")
    parser.add_argument("--path_prepared_dataset", type=str, default="/Volumes/DD_FGS/MICS/data_HE2CellType/HE2CT/prepared_datasets_cat/ct_1", help="Path to the prepared_datasets directory")
    parser.add_argument("--he_patches_selection", type=tuple, default=('Jaccard', 0.45, 1.0), help="Tuple with metric name (Dice, Jaccard, bPQ) and interval for filtering he_patches (min and max values)")
    parser.add_argument("--output_dir", type=str, default="/Volumes/DD_FGS/MICS/data_HE2CellType/HE2CT/training_datasets", help="Path to the training datasets directory")
    parser.add_argument("--dataset_id", type=str, default="ds_test", help="Training dataset ID")

    parser.add_argument(
        "--grouping",
        type=str,
        default=None,
        help=("JSON string specifying grouping for cell count columns => create a new cell_cat_id dataset by grouping cat from a previous dataset. / If no grouping then put None"
          "Example: '{\"Immune\": [\"T_NK\", \"B_Plasma\", \"Myeloid\"], \"Stromal\": [\"Blood_vessel\", \"Fibroblast_Myofibroblast\"]}'")
        )

    parser.add_argument(
        "--force_train_patches", 
        type=str,
        default="breast_s0_44681.png,breast_s0_37919.png,breast_s1_76960.png,breast_s1_76791.png,breast_s3_33838.png,breast_s3_33905.png,breast_s6_27698.png,breast_s6_27673.png,cervix_s0_13514.png,cervix_s0_13512.png,colon_s1_13951.png,colon_s1_13847.png,colon_s2_19293.png,colon_s2_12819.png,heart_s0_480.png,heart_s0_5774.png,kidney_s0_6356.png,kidney_s0_6345.png,kidney_s1_2766.png,kidney_s1_4478.png,liver_s0_14362.png,liver_s0_14345.png,liver_s1_2143.png,liver_s1_9482.png,lung_s1_12337.png,lung_s1_12330.png,lung_s3_13535.png,lung_s3_14408.png,lymph_node_s0_3723.png,lymph_node_s0_3421.png,ovary_s0_19561.png,ovary_s0_15869.png,ovary_s1_15114.png,ovary_s1_15077.png,pancreatic_s0_6024.png,pancreatic_s0_6031.png,pancreatic_s1_7181.png,pancreatic_s1_5847.png,pancreatic_s2_10084.png,pancreatic_s2_10554.png,prostate_s0_21513.png,prostate_s0_18554.png,skin_s1_7447.png,skin_s1_7349.png,skin_s2_2417.png,skin_s2_17544.png,skin_s3_8415.png,skin_s3_8560.png,skin_s4_15893.png,skin_s4_15891.png,tonsil_s0_12409.png,tonsil_s0_12397.png,tonsil_s1_24106.png,tonsil_s1_24105.png",
        help="Comma-separated list of reference patches from multiple slides that have been used as references for macenko => we need to force them to be in the training set"
        )
    
    parser.add_argument("--macenko", action="store_true",
                        help="Flag to indicate that macenko filtered images should be removed from the dataset because we will use the Macenko normalization images for training => we need to remove patches where Macenko did not work well")

    args = parser.parse_args()
    main(args)



# Example command:
# python3 cell_segmentation/datasets/make_folds_pannuke.py --slide_ids breast_s0 breast_s1 breast_s3 breast_s6 lung_s1 lung_s3 skin_s1 skin_s2 skin_s3 skin_s4 pancreatic_s0 pancreatic_s1 pancreatic_s2 heart_s0 colon_s1 colon_s2 kidney_s0 kidney_s1 liver_s0 liver_s1 tonsil_s0 tonsil_s1 lymph_node_s0 ovary_s0 ovary_s1 prostate_s0 cervix_s0 --grouping '{"Immune": ["T_NK", "B_Plasma", "Myeloid"], "Stromal": ["Blood_vessel", "Fibroblast_Myofibroblast"], "Other": ["Specialized", "Dead"]}' --dataset_id ds_4
