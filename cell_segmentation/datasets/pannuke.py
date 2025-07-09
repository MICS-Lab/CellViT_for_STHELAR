# -*- coding: utf-8 -*-
# PanNuke Dataset
#
# Dataset information: https://arxiv.org/abs/2003.10778
# Please Prepare Dataset as described here: docs/readmes/pannuke.md
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import logging
import sys  # remove
from pathlib import Path
from typing import Callable, Tuple, Union

sys.path.append("/homes/fhoerst/histo-projects/CellViT/")  # remove

from tqdm import tqdm
import zipfile
import io
import numpy as np
import pandas as pd
import torch
import yaml
from numba import njit
from PIL import Image
from scipy.ndimage import center_of_mass, distance_transform_edt
from scipy.sparse import csr_matrix, load_npz
import re

from cell_segmentation.datasets.base_cell import CellDataset
from cell_segmentation.utils.tools import fix_duplicates, get_bounding_box

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())

from natsort import natsorted


class PanNukeDataset(CellDataset):
    """PanNuke dataset

    Args:
        dataset_path (Union[Path, str]): Path to PanNuke dataset. Structure is described under ./docs/readmes/cell_segmentation.md
        folds (Union[str, list[str]]): Folds to use for this dataset ('train', 'valid', 'test')
        transforms (Callable, optional): PyTorch transformations. Defaults to None.
        stardist (bool, optional): Return StarDist labels. Defaults to False
        regression (bool, optional): Return Regression of cells in x and y direction. Defaults to False
        cache_dataset: If the dataset should be loaded to host memory in first epoch.
            Be careful, workers in DataLoader needs to be persistent to have speedup.
            Recommended to false, just use if you have enough RAM and your I/O operations might be limited.
            Defaults to False.
        cell_tokens (str, optional): If retrieving cell tokens, which masks to use ('nucleus' or 'cell'). Defaults to 'no'.

        If cell_cat_id dataset is built from a previous physically existing dataset, the following parameters are required:
            orig_class_names (list, optional): List of original class names (index corresponds to type_map value). Defaults to None.
            grouping (dict, optional):  Dict mapping group names to list of original names. Defaults to None.
            new_class_mapping (dict, optional): Dict mapping final category names to new indices. Defaults to None.
        
        macenko_normalization (bool, optional): If the dataset should be Macenko normalized. Defaults to False.
    """

    def __init__(
        self,
        dataset_path: Union[Path, str],
        folds: Union[str, list[str]],
        transforms: Callable = None,
        stardist: bool = False,
        regression: bool = False,
        cache_dataset: bool = False,
        cell_tokens: str = "no",
        orig_class_names=None,
        grouping=None,
        new_class_mapping=None,
        macenko_normalization: bool = False,
    ) -> None:
        if isinstance(folds, str):
            folds = [folds]

        self.dataset = Path(dataset_path).resolve()
        self.transforms = transforms
        self.images = []
        self.masks = []
        self.types = {}
        self.img_names = []
        self.folds = folds
        self.cache_dataset = cache_dataset
        self.stardist = stardist
        self.regression = regression
        self.cell_tokens = cell_tokens
        self.orig_class_names = orig_class_names
        self.grouping = grouping
        self.new_class_mapping = new_class_mapping
        self.macenko_normalization = macenko_normalization
        self.types = pd.read_csv(self.dataset / "types.csv")
        
        for fold in folds:
            
            if not self.macenko_normalization:
                self.img_zip_path = self.dataset / "images.zip"
                print("Not using Macenko normalization, img_zip_path is", self.img_zip_path)
            else:
                self.img_zip_path = self.dataset / "images_macenko.zip"
                print("Using Macenko normalization, img_zip_path is", self.img_zip_path)
            self.mask_zip_path = self.dataset / "labels.zip"
            if not self.img_zip_path.is_file() or not self.mask_zip_path.is_file():
                raise FileNotFoundError(f"Image or mask zip file not found for fold {fold}")
            
            with zipfile.ZipFile(self.img_zip_path, 'r') as img_zip, zipfile.ZipFile(self.mask_zip_path, 'r') as mask_zip:
                
                fold_images = pd.read_csv(self.dataset / f"cell_count_{fold}.csv")['Image'].tolist()
                fold_images = natsorted(fold_images)

                # # SAMPLING FOR TESTING -- TOREMOVE #
                # # Choose data randomly
                # fold_images = np.random.choice(fold_images, 100, replace=False)
                # ###################################

                # Precompute file lists as sets for O(1) lookup
                existing_img_files = set(img_zip.namelist())
                existing_mask_files = set(mask_zip.namelist())

                # sanity_check: image and mask should exist
                for fold_image in tqdm(fold_images, desc=f"{fold}"):

                    mask_path = f"{fold_image.rsplit('.', 1)[0]}.npz"

                    if fold_image in existing_img_files and mask_path in existing_mask_files:
                        self.images.append((fold, fold_image))
                        self.masks.append((fold, mask_path))
                        self.img_names.append(fold_image)
                    else:
                        logger.debug(f"[WARNING] {fold_image} not in {'images' if fold_image not in existing_img_files else 'masks'} zip file!")
                        # remove fold_image from fold_images
                        fold_images.remove(fold_image)


            # Keep in type only row with img is in fold_images
            fold_types = self.types[self.types["img"].isin(set(fold_images))]
            fold_type_dict = fold_types.set_index("img")["type"].to_dict()
            self.types = {
                **self.types,
                **fold_type_dict,
            }  # careful - should all be named differently

        logger.info(f"Created Pannuke Dataset by using fold(s) {self.folds}")
        print(f"Created Pannuke Dataset by using fold(s) {self.folds}")
        logger.info(f"Resulting dataset length: {self.__len__()}")
        print(f"Resulting dataset length: {self.__len__()}")

        if self.cache_dataset:
            self.cached_idx = []  # list of idx that should be cached
            self.cached_imgs = {}  # keys: idx, values: numpy array of imgs
            self.cached_masks = {}  # keys: idx, values: numpy array of masks
            logger.info("Using cached dataset. Cache is built up during first epoch.")
            print("Using cached dataset. Cache is built up during first epoch.")
        
        # ### CHOOSE ONLY FOR INFERENCE IN SEVERAL TIMES ###
        # # DO NOT FORGET TO REMOVE MEAN FOR CELL TOKENS IN THE INFERENCE SCRIPT (cf. to comment)
        # split_idx = len(self.images) // 2
        # self.images = self.images[split_idx:]
        # self.masks = self.masks[split_idx:]
        # self.img_names = self.img_names[split_idx:]
        # ### END CHOOSE ONLY FOR SEVERAL TIMES INFERENCE ###

        self.opened_img_zip = None # Initialize as None; open in the worker process
        self.opened_mask_zip = None # Initialize as None; open in the worker process
        self.opened_mask_zip_cell_tokens = None # Initialize as None; open in the worker process

        # Precompute the vectorized mapping array (if mapping parameters are provided)
        if self.orig_class_names is not None and self.grouping is not None and self.new_class_mapping is not None:
            self.type_mapping_array = self.build_type_mapping(self.orig_class_names, self.grouping, self.new_class_mapping)
        else:
            self.type_mapping_array = None
    
        if self.type_mapping_array is not None:
            print("Remapping type_map using the following mapping array:", self.type_mapping_array)

    def _init_zipfiles(self):
        if self.opened_img_zip is None:
            logger.info(f"Opening image zip file {self.img_zip_path}...")
            print(f"Opening image zip file {self.img_zip_path}...")
            self.opened_img_zip = zipfile.ZipFile(self.img_zip_path, 'r')
        if self.opened_mask_zip is None:
            logger.info(f"Opening mask zip file {self.mask_zip_path}...")
            print(f"Opening mask zip file {self.mask_zip_path}...")
            self.opened_mask_zip = zipfile.ZipFile(self.mask_zip_path, 'r')
    

    def _init_zipfile_cell_tokens(self):
       
        if self.cell_tokens == "nucleus":
            path_zip = self.dataset / "masks_cell_ids_nuclei.zip"
        elif self.cell_tokens == "cell":
            path_zip = self.dataset / "masks_cell_ids_cells.zip"
        else:
            raise ValueError(f"cell_tokens must be 'nucleus' or 'cell', but was {self.cell_tokens}")
        
        if self.opened_mask_zip_cell_tokens is None:
            logger.info(f"Opening mask zip file {path_zip}...")
            print(f"Opening mask zip file {path_zip}...")
            self.opened_mask_zip_cell_tokens = zipfile.ZipFile(path_zip, 'r')


    def __del__(self):
        if self.opened_img_zip is not None:
            self.opened_img_zip.close()
        if self.opened_mask_zip is not None:
            self.opened_mask_zip.close()



    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict, str, str]:
        """Get one dataset item consisting of transformed image,
        masks (instance_map, nuclei_type_map, nuclei_binary_map, hv_map) and tissue type as string

        Args:
            index (int): Index of element to retrieve

        Returns:
            Tuple[torch.Tensor, dict, str, str]:
                torch.Tensor: Image, with shape (3, H, W), in this case (3, 256, 256)
                dict:
                    "instance_map": Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (256, 256)
                    "nuclei_type_map": Nuclei-Type-Map, for each nucleus (instance) the class is indicated by an integer. Shape (256, 256)
                    "nuclei_binary_map": Binary Nuclei-Mask, Shape (256, 256)
                    "hv_map": Horizontal and vertical instance map.
                        Shape: (2 , H, W). First dimension is horizontal (horizontal gradient (-1 to 1)),
                        last is vertical (vertical gradient (-1 to 1)) Shape (2, 256, 256)
                    [Optional if stardist]
                    "dist_map": Probability distance map. Shape (256, 256)
                    "stardist_map": Stardist vector map. Shape (n_rays, 256, 256)
                    [Optional if regression]
                    "regression_map": Regression map. Shape (2, 256, 256). First is vertical, second horizontal.
                str: Tissue type
                str: Image Name
        """

        self._init_zipfiles()  # Ensure the ZIP files are open in this worker

        img_fold, img_name = self.images[index]
        mask_fold, mask_name = self.masks[index]

        # Extract slide_id from img_name
        slide_id = re.match(r"([a-zA-Z0-9_]+)_\d+\.png", img_name)
        if slide_id:
            slide_id = slide_id.group(1)
        else:
            raise ValueError(f"Unexpected filename format / not able to extract slide_id: {img_name}")

        if self.cache_dataset:
            if index in self.cached_idx:
                img = self.cached_imgs[index]
                mask = self.cached_masks[index]
            else:
                # cache file
                img = self.load_imgfile(img_name)
                mask = self.load_maskfile(mask_name)
                self.cached_imgs[index] = img
                self.cached_masks[index] = mask
                self.cached_idx.append(index)

        else:
            
            img = self.load_imgfile(img_name)
            mask = self.load_maskfile(mask_name)

            if self.cell_tokens != "no":
                self._init_zipfile_cell_tokens()
                mask_cell_tokens = self.load_maskfile_cell_tokens(mask_name)
        
        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask, slide_id=slide_id)
            img = transformed["image"]
            mask = transformed["mask"]

        tissue_type = self.types[img_name]
        inst_map = mask[:, :, 0].copy()
        type_map = mask[:, :, 1].copy()
        np_map = mask[:, :, 0].copy()
        np_map[np_map > 0] = 1
        hv_map = PanNukeDataset.gen_instance_hv_map(inst_map)

        # torch convert
        img = torch.Tensor(img).type(torch.float32)
        img = img.permute(2, 0, 1)
        if torch.max(img) >= 5:
            img = img / 255

        masks = {
            "instance_map": torch.Tensor(inst_map).type(torch.int64),
            "nuclei_type_map": torch.Tensor(type_map).type(torch.int64),
            "nuclei_binary_map": torch.Tensor(np_map).type(torch.int64),
            "hv_map": torch.Tensor(hv_map).type(torch.float32),
        }

        # load stardist transforms if neccessary
        if self.stardist:
            dist_map = PanNukeDataset.gen_distance_prob_maps(inst_map)
            stardist_map = PanNukeDataset.gen_stardist_maps(inst_map)
            masks["dist_map"] = torch.Tensor(dist_map).type(torch.float32)
            masks["stardist_map"] = torch.Tensor(stardist_map).type(torch.float32)
        if self.regression:
            masks["regression_map"] = PanNukeDataset.gen_regression_map(inst_map)

        if self.cell_tokens != "no":
            return img, masks, tissue_type, img_name, mask_cell_tokens
        else:
            return img, masks, tissue_type, img_name


    def __len__(self) -> int:
        """Length of Dataset

        Returns:
            int: Length of Dataset
        """
        return len(self.images)


    def set_transforms(self, transforms: Callable) -> None:
        """Set the transformations, can be used tp exchange transformations

        Args:
            transforms (Callable): PyTorch transformations
        """
        self.transforms = transforms


    # def load_imgfile(self, index: int) -> np.ndarray:
    #     """Load image from file (disk)

    #     Args:
    #         index (int): Index of file

    #     Returns:
    #         np.ndarray: Image as array with shape (H, W, 3)
    #     """
    #     img_path = self.images[index]
    #     return np.array(Image.open(img_path)).astype(np.uint8)

    def load_imgfile(self, img_name: str) -> np.ndarray:
        img_data = self.opened_img_zip.read(img_name)
        img = Image.open(io.BytesIO(img_data))
        return np.array(img).astype(np.uint8)
    


    def build_type_mapping(self, orig_class_names, grouping, new_class_mapping):
        """
        Build a vectorized mapping array => Useful if the cell_cat_id dataset is built using a previous physically existing dataset
        
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


    # def load_maskfile(self, index: int) -> np.ndarray:
    #     """Load mask from file (disk)

    #     Args:
    #         index (int): Index of file

    #     Returns:
    #         np.ndarray: Mask as array with shape (H, W, 2)
    #     """
    #     mask_path = self.masks[index]
    #     mask = np.load(mask_path, allow_pickle=True)
    #     inst_map = mask[()]["inst_map"].astype(np.int32)
    #     type_map = mask[()]["type_map"].astype(np.int32)
    #     mask = np.stack([inst_map, type_map], axis=-1)
    #     return mask

    def load_maskfile(self, mask_name: str) -> np.ndarray:
        
        # Load the .npz file using numpy
        data = np.load(io.BytesIO(self.opened_mask_zip.read(mask_name)), allow_pickle=True)
        
        # Reconstruct sparse matrices
        inst_map_sparse = csr_matrix(
            (data["inst_map_data"], data["inst_map_indices"], data["inst_map_indptr"]),
            shape=data["inst_map_shape"]
        )
        type_map_sparse = csr_matrix(
            (data["type_map_data"], data["type_map_indices"], data["type_map_indptr"]),
            shape=data["type_map_shape"]
        )

        # Convert sparse matrices to dense arrays
        inst_map = inst_map_sparse.toarray().astype(np.int32)
        type_map = type_map_sparse.toarray().astype(np.int32)

        # Remap type_map using vectorized mapping if provided
        if self.type_mapping_array is not None:
            type_map = self.type_mapping_array[type_map]

        # type_map[type_map != 0] = 1  # CHOOSE OR NOT / choose to do check align during inference (put everything in category 1)

        return np.stack([inst_map, type_map], axis=-1)
    

    def load_maskfile_cell_tokens(self, mask_name: str) -> np.ndarray:
                
        mask_cell_tokens = load_npz(io.BytesIO(self.opened_mask_zip_cell_tokens.read(mask_name)))
        mask_cell_tokens = mask_cell_tokens.toarray().astype(np.int32)

        return mask_cell_tokens


    def load_cell_count(self):
        """Load Cell count from cell_count_{fold}.csv file. File must be located inside the dataset folder
        and named "cell_count_{fold}.csv"

        Example file beginning:
            Image,Neoplastic,Inflammatory,Connective,Dead,Epithelial
            0_0.png,4,2,2,0,0
            0_1.png,8,1,1,0,0
            0_10.png,17,0,1,0,0
            0_100.png,10,0,11,0,0
            ...
        """
        df_placeholder = []
        for fold in self.folds:
            csv_path = self.dataset / f"cell_count_{fold}.csv"
            cell_count = pd.read_csv(csv_path, index_col=0)
            df_placeholder.append(cell_count)
        self.cell_count = pd.concat(df_placeholder)
        self.cell_count = self.cell_count.reindex(self.img_names)


    def get_sampling_weights_tissue(self, gamma: float = 1) -> torch.Tensor:
        """Get sampling weights calculated by tissue type statistics

        For this, a file named "weight_config.yaml" with the content:
            tissue:
                tissue_1: xxx
                tissue_2: xxx (name of tissue: count)
                ...
        Must exists in the dataset main folder (parent path, not inside the folds)

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        with open(
            (self.dataset / "weight_config.yaml").resolve(), "r"
        ) as run_config_file:
            yaml_config = yaml.safe_load(run_config_file)
            tissue_counts = dict(yaml_config)["tissue"]

        # calculate weight for each tissue
        weights_dict = {}
        k = np.sum(list(tissue_counts.values()))
        for tissue, count in tissue_counts.items():
            w = k / (gamma * count + (1 - gamma) * k)
            weights_dict[tissue] = w

        weights = []
        for idx in range(self.__len__()):
            img_idx = self.img_names[idx]
            type_str = self.types[img_idx]
            weights.append(weights_dict[type_str])

        return torch.Tensor(weights)


    def get_sampling_weights_cell(self, gamma: float = 1, ignore_cat: list = None) -> torch.Tensor:
        """Get sampling weights calculated by cell type statistics

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.
            ignore_cat (list): List of cell type columns to exclude or "ignore" from the
                weighting logic. Defaults to None.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        assert hasattr(self, "cell_count"), "Please run .load_cell_count() in advance!"

        if ignore_cat is None:
            ignore_cat = []

        # Determine which columns to *actually* use for weighting
        relevant_cols = [col for col in self.cell_count.columns if col not in ignore_cat]
        logger.info(f"Relevant columns for weighting cell strategy to balance the dataset: {relevant_cols}")
        print(f"Relevant columns for weighting cell strategy to balance the dataset: {relevant_cols}")
        if len(relevant_cols) == 0:
            raise ValueError("No columns left for weighting after ignoring categories")
        
        # Convert to 0/1 presence matrix, but only for the relevant columns
        cell_counts_imgs = np.clip(self.cell_count[relevant_cols].to_numpy(), 0, 1)
        binary_weight_factors = np.sum(cell_counts_imgs, axis=0)   # remove binary_weight_factors = np.array([4191, 4132, 6140, 232, 1528]) from the original code
        k = np.sum(binary_weight_factors)

        weight_vector = k / (gamma * binary_weight_factors + (1 - gamma) * k)
        img_weight = (1 - gamma) * np.max(cell_counts_imgs, axis=-1) + gamma * np.sum(
            cell_counts_imgs * weight_vector, axis=-1
        )
        img_weight[np.where(img_weight == 0)] = np.min(
            img_weight[np.nonzero(img_weight)]
        )

        return torch.Tensor(img_weight)


    def get_sampling_weights_cell_tissue(self, gamma: float = 1, ignore_cat: list = None) -> torch.Tensor:
        """Get combined sampling weights by calculating tissue and cell sampling weights,
        normalizing them and adding them up to yield one score.

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.
            ignore_cat (list): List of cell type columns to exclude or "ignore" from the
                weighting logic. Defaults to None.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        tw = self.get_sampling_weights_tissue(gamma)
        cw = self.get_sampling_weights_cell(gamma, ignore_cat=ignore_cat)
        weights = tw / torch.max(tw) + cw / torch.max(cw)

        return weights


    @staticmethod
    def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
        """Obtain the horizontal and vertical distance maps for each
        nuclear instance.

        Args:
            inst_map (np.ndarray): Instance map with each instance labelled as a unique integer
                Shape: (H, W)
        Returns:
            np.ndarray: Horizontal and vertical instance map.
                Shape: (2, H, W). First dimension is horizontal (horizontal gradient (-1 to 1)),
                last is vertical (vertical gradient (-1 to 1))
        """
        orig_inst_map = inst_map.copy()  # instance ID map

        x_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)

        inst_list = list(np.unique(orig_inst_map))
        inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(orig_inst_map == inst_id, np.uint8)
            inst_box = get_bounding_box(inst_map)

            # expand the box by 2px
            # Because we first pad the ann at line 207, the bboxes
            # will remain valid after expansion
            if inst_box[0] >= 2:
                inst_box[0] -= 2
            if inst_box[2] >= 2:
                inst_box[2] -= 2
            if inst_box[1] <= orig_inst_map.shape[0] - 2:
                inst_box[1] += 2
            if inst_box[3] <= orig_inst_map.shape[0] - 2:
                inst_box[3] += 2

            # improvement
            inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue

            # instance center of mass, rounded to nearest pixel
            inst_com = list(center_of_mass(inst_map))

            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)

            inst_x_range = np.arange(1, inst_map.shape[1] + 1)
            inst_y_range = np.arange(1, inst_map.shape[0] + 1)
            # shifting center of pixels grid to instance center of mass
            inst_x_range -= inst_com[1]
            inst_y_range -= inst_com[0]

            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

            # remove coord outside of instance
            inst_x[inst_map == 0] = 0
            inst_y[inst_map == 0] = 0
            inst_x = inst_x.astype("float32")
            inst_y = inst_y.astype("float32")

            # normalize min into -1 scale
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
            # normalize max into +1 scale
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

            ####
            x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

            y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        hv_map = np.stack([x_map, y_map])
        return hv_map


    @staticmethod
    def gen_distance_prob_maps(inst_map: np.ndarray) -> np.ndarray:
        """Generate distance probability maps

        Args:
            inst_map (np.ndarray): Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (H, W)

        Returns:
            np.ndarray: Distance probability map, shape (H, W)
        """
        inst_map = fix_duplicates(inst_map)
        dist = np.zeros_like(inst_map, dtype=np.float64)
        inst_list = list(np.unique(inst_map))
        if 0 in inst_list:
            inst_list.remove(0)

        for inst_id in inst_list:
            inst = np.array(inst_map == inst_id, np.uint8)

            y1, y2, x1, x2 = get_bounding_box(inst)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2

            inst = inst[y1:y2, x1:x2]

            if inst.shape[0] < 2 or inst.shape[1] < 2:
                continue

            # chessboard distance map generation
            # normalize distance to 0-1
            inst_dist = distance_transform_edt(inst)
            inst_dist = inst_dist.astype("float64")

            max_value = np.amax(inst_dist)
            if max_value <= 0:
                continue
            inst_dist = inst_dist / (np.max(inst_dist) + 1e-10)

            dist_map_box = dist[y1:y2, x1:x2]
            dist_map_box[inst > 0] = inst_dist[inst > 0]

        return dist


    @staticmethod
    @njit
    def gen_stardist_maps(inst_map: np.ndarray) -> np.ndarray:
        """Generate StarDist map with 32 nrays

        Args:
            inst_map (np.ndarray): Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (H, W)

        Returns:
            np.ndarray: Stardist vector map, shape (n_rays, H, W)
        """
        n_rays = 32
        # inst_map = fix_duplicates(inst_map)
        dist = np.empty(inst_map.shape + (n_rays,), np.float32)

        st_rays = np.float32((2 * np.pi) / n_rays)
        for i in range(inst_map.shape[0]):
            for j in range(inst_map.shape[1]):
                value = inst_map[i, j]
                if value == 0:
                    dist[i, j] = 0
                else:
                    for k in range(n_rays):
                        phi = np.float32(k * st_rays)
                        dy = np.cos(phi)
                        dx = np.sin(phi)
                        x, y = np.float32(0), np.float32(0)
                        while True:
                            x += dx
                            y += dy
                            ii = int(round(i + x))
                            jj = int(round(j + y))
                            if (
                                ii < 0
                                or ii >= inst_map.shape[0]
                                or jj < 0
                                or jj >= inst_map.shape[1]
                                or value != inst_map[ii, jj]
                            ):
                                # small correction as we overshoot the boundary
                                t_corr = 1 - 0.5 / max(np.abs(dx), np.abs(dy))
                                x -= t_corr * dx
                                y -= t_corr * dy
                                dst = np.sqrt(x**2 + y**2)
                                dist[i, j, k] = dst
                                break

        return dist.transpose(2, 0, 1)


    @staticmethod
    def gen_regression_map(inst_map: np.ndarray):
        n_directions = 2
        dist = np.zeros(inst_map.shape + (n_directions,), np.float32).transpose(2, 0, 1)
        inst_map = fix_duplicates(inst_map)
        inst_list = list(np.unique(inst_map))
        if 0 in inst_list:
            inst_list.remove(0)
        for inst_id in inst_list:
            inst = np.array(inst_map == inst_id, np.uint8)
            y1, y2, x1, x2 = get_bounding_box(inst)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2

            inst = inst[y1:y2, x1:x2]
            y_mass, x_mass = center_of_mass(inst)
            x_map = np.repeat(np.arange(1, x2 - x1 + 1)[None, :], y2 - y1, axis=0)
            y_map = np.repeat(np.arange(1, y2 - y1 + 1)[:, None], x2 - x1, axis=1)
            # we use a transposed coordinate system to align to HV-map, correct would be -1*x_dist_map and -1*y_dist_map
            x_dist_map = (x_map - x_mass) * np.clip(inst, 0, 1)
            y_dist_map = (y_map - y_mass) * np.clip(inst, 0, 1)
            dist[0, y1:y2, x1:x2] = x_dist_map
            dist[1, y1:y2, x1:x2] = y_dist_map

        return dist
