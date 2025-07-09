# Fine-Tuning the CellViT Model with STHELAR, a Xenium-Based Spatial Transcriptomics Dataset

This repository is adapted from the original [CellViT repository](https://github.com/TIO-IKIM/CellViT) with the following reference:  
Hörst, F. et al. CellViT: Vision Transformers for precise cell segmentation and classification. Medical Image Analysis 94, 103143 (2024). https://doi.org/10.1016/j.media.2024.103143

This repository focuses on fine-tuning the CellViT model using a dataset built using publicly available Spatial Transcriptomics (ST) data from the 10x Genomics Xenium platform (**STHELAR dataset**). The dataset construction process, including H&E image patch extraction, nucleus segmentation, and cell-type classification based on RNA information, is detailed in the following github repository: [STHELAR](https://github.com/MICS-Lab/STHELAR).  

The resulting dataset includes:  
- H&E image patches
- Corresponding nucleus segmentation masks
- Cell-type annotations derived from RNA information
- Tissue provenance metadata

The goal is to fine-tune the CellViT model using a large-scale dataset with more precise cell type classes.

## CellViT Model and Codebase Modifications

All original information regarding the CellViT model, including its pre-training and authorship, is maintained in the file [`README_CellViT.md`](README_CellViT.md).

Several modifications have been made to the original codebase. For instance:
- The data format has been adapted to efficiently handle large-scale datasets.
- Dataset selection has been made more flexible to allow fine-tuning on different label levels and dataset subsets.
- Some code to extract more informations like cell features for instance.

## New Scripts and Notebooks

The following files have been added to support our dataset preparation and analysis:

- In `cell_segmentation/datasets`:
    - `convert_into_zip.py`: Converts the dataset into ZIP format.
    - `make_folds_pannuke.py`: Creates data splits based on slide selection and patch-level metrics.
    - `analyse_ds_patches.ipynb`: Analyzes the composition and distribution of patches in the dataset.
    - `get_weights_dataset.ipynb`: Computes weights for losses and dataset balancing.
    - `calculate_mean_std_train.py`: Calculates the mean and standard deviation of RGB channels in the training set.
    - `calculate_loss_extrema.py` and `analyze_loss_extrema_training.ipynb`: Estimate and analyze the range (extrema) and the random case of loss values during training.
    - `macenko_normationzation(_v2).py`: Perform Macenko normalization on the dataset.
- In `cell_segmentation/utils`:
    - `HED_augmentation.py`: Specific augmentation for H&E slides.

## Related Publication

A detailed description of the pipeline, methods, and results can be found in the following article: [link to article].