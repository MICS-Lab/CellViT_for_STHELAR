#!/usr/bin/env python3
"""
Convert STHELAR Hugging Face parquet data into a CellViT-ready dataset.

Output structure:

output_root/
├── images.zip
├── labels.zip
├── types.csv
├── cell_count_train.csv
├── cell_count_valid.csv
├── cell_count_test.csv
├── dataset_config.yaml
├── patch_info_with_split.csv
└── split_manifest.yaml

Supported split strategies:

- baseline:
    Random patch-level split. Useful as a reference baseline, but potentially leaky.

- spatial:
    Spatial split based on patch coordinates. Useful when only one slide/scan is available.
    Boundary regions around split borders are discarded.

- slide:
    Slide-level split. Useful when at least two slides/scans are available.
    If only two slides are available, one slide is used for test and train/valid
    are carved spatially from the other slide.

- auto:
    If >= 2 selected slides are available, use slide.
    If only 1 selected slide is available, use spatial.

Example:

python preprocessing/sthelar/convert_hf_to_cellvit.py \
  --sthelar-root /gpfs/workdir/taddeial/workspace/Datasets/STHELAR_20x \
  --output-root /gpfs/workdir/taddeial/workspace/Datasets/cellvit_ready/sthelar20x_tonsil_auto \
  --tissue tonsil \
  --strategy auto \
  --split-axis x \
  --boundary-margin 128 \
  --overwrite
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
import yaml
from PIL import Image
from scipy.sparse import csr_matrix


# ============================================================
# Class mapping
# ============================================================

FIVE_CLASS_MAP = {
    "T_NK": "Immune",
    "B_Plasma": "Immune",
    "Myeloid": "Immune",
    "Blood_vessel": "Stromal",
    "Fibroblast_Myofibroblast": "Stromal",
    "Epithelial": "Epithelial",
    "Melanocyte": "Other",
    "Specialized": "Other",
    "Other": "Other",
}

CLASS_TO_INT = {
    "Background": 0,
    "Immune": 1,
    "Stromal": 2,
    "Epithelial": 3,
    "Other": 4,
}

NUCLEI_TYPES = {
    "Background": 0,
    "Immune": 1,
    "Stromal": 2,
    "Epithelial": 3,
    "Other": 4,
}

COUNT_CLASS_NAMES = ["Immune", "Stromal", "Epithelial", "Other"]


# ============================================================
# Generic helpers
# ============================================================

def load_yaml_config(path: Optional[str]) -> dict[str, Any]:
    if path is None:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(f"Config YAML must contain a mapping/dictionary: {config_path}")

    return config


def as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def sanitize_for_filename(value: str) -> str:
    value = str(value)
    value = value.replace("/", "_").replace("\\", "_")
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def make_patch_uid(row: pd.Series) -> str:
    """
    Globally unique image name used inside CellViT zip/csv files.

    This avoids collisions such as:

        tonsil_s0 / patch_0001.png
        tonsil_s1 / patch_0001.png

    becoming the same archive entry.
    """
    original_name = Path(str(row["file_name"])).name
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix or ".png"

    slide_id = sanitize_for_filename(str(row["slide_id"]))
    xmin = int(row["xmin"])
    ymin = int(row["ymin"])
    stem = sanitize_for_filename(stem)

    return f"{slide_id}__x{xmin}_y{ymin}__{stem}{suffix}"


def image_to_label_name(image_name: str) -> str:
    return str(Path(image_name).with_suffix(".npz"))


def image_to_typemap_name(image_name: str) -> str:
    return str(Path(image_name).with_suffix(".npy"))


def make_patch_key_from_values(slide_id: str, file_name: str) -> tuple[str, str]:
    return (str(slide_id), str(file_name))


def make_patch_key(row: pd.Series) -> tuple[str, str]:
    return make_patch_key_from_values(str(row["slide_id"]), str(row["file_name"]))


def make_zip_from_entries(entries: list[tuple[Path, str]], zip_path: Path) -> None:
    """
    entries: list of (source_path, archive_name)
    archive_name must be globally unique inside the zip.
    """
    if zip_path.exists():
        zip_path.unlink()

    seen_arcnames = set()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source_path, arcname in entries:
            source_path = Path(source_path)

            if not source_path.exists():
                raise FileNotFoundError(f"Missing file while creating zip: {source_path}")

            if arcname in seen_arcnames:
                raise ValueError(f"Duplicate archive name in zip: {arcname}")

            seen_arcnames.add(arcname)
            zf.write(source_path, arcname=arcname)


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_root}\n"
                f"Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)


# ============================================================
# STHELAR HF loading helpers
# ============================================================

def find_overview_path(sthelar_root: Path, overview_path: Optional[str]) -> Path:
    if overview_path is not None:
        path = as_path(overview_path)
        if not path.exists():
            raise FileNotFoundError(f"Overview parquet not found: {path}")
        return path

    candidates = sorted(sthelar_root.glob("patches_overview_sthelar*.parquet"))
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No patches_overview_sthelar*.parquet file found under {sthelar_root}"
        )

    if len(candidates) > 1:
        names = "\n".join(f"  - {p}" for p in candidates)
        raise ValueError(
            "Multiple overview parquet files found. Please pass --overview-path explicitly:\n"
            f"{names}"
        )

    return candidates[0]


def read_parquet_columns(path: Path, requested_cols: list[str]) -> pd.DataFrame:
    schema_names = set(pq.read_schema(path).names)
    cols = [c for c in requested_cols if c in schema_names]

    if len(cols) == 0:
        raise ValueError(f"None of the requested columns exist in {path}: {requested_cols}")

    return pd.read_parquet(path, columns=cols)


def load_patch_overview(
    overview_path: Path,
    tissue: Optional[str],
    slide_ids: Optional[list[str]],
) -> pd.DataFrame:
    requested_cols = [
        "file_name",
        "slide_id",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "Dice",
        "Jaccard",
        "bPQ",
    ]

    overview = read_parquet_columns(overview_path, requested_cols)

    required = ["file_name", "slide_id", "xmin", "ymin", "xmax", "ymax"]
    missing = [c for c in required if c not in overview.columns]
    if missing:
        raise ValueError(f"Overview parquet is missing required columns: {missing}")

    overview["slide_id"] = overview["slide_id"].astype(str)
    overview["file_name"] = overview["file_name"].astype(str)

    if slide_ids:
        slide_ids_set = set(str(s) for s in slide_ids)
        overview = overview[overview["slide_id"].isin(slide_ids_set)].copy()
    elif tissue:
        tissue = str(tissue)
        overview = overview[
            overview["slide_id"].str.startswith(f"{tissue}_")
            | overview["slide_id"].str.contains(tissue, regex=False)
        ].copy()

    if len(overview) == 0:
        raise ValueError(
            "No patches found after filtering overview.\n"
            f"tissue={tissue}, slide_ids={slide_ids}"
        )

    duplicate_keys = overview.duplicated(subset=["slide_id", "file_name"])
    if duplicate_keys.any():
        examples = overview.loc[duplicate_keys, ["slide_id", "file_name"]].head(10)
        raise ValueError(
            "Duplicate (slide_id, file_name) keys found in overview. "
            "The script assumes file_name is unique within a slide.\n"
            f"{examples}"
        )

    overview = overview.sort_values(["slide_id", "file_name"]).reset_index(drop=True)
    return overview


def apply_max_patches_per_slide(
    patch_info: pd.DataFrame,
    max_patches_per_slide: Optional[int],
    random_seed: int,
) -> pd.DataFrame:
    if max_patches_per_slide is None:
        return patch_info.copy()

    parts = []
    for slide_id, df_slide in patch_info.groupby("slide_id", sort=True):
        if len(df_slide) > max_patches_per_slide:
            df_slide = df_slide.sample(n=max_patches_per_slide, random_state=random_seed)
        parts.append(df_slide)

    out = pd.concat(parts, axis=0, ignore_index=True)
    out = out.sort_values(["slide_id", "file_name"]).reset_index(drop=True)
    return out


def load_slide_meta(sthelar_root: Path, slide_id: str) -> pd.DataFrame:
    path = sthelar_root / "cell_metadata" / f"{slide_id}_cell_metadata.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing slide metadata parquet: {path}")

    requested_cols = ["cell_id_int", "cell_id", "cells_final_label_group"]
    meta = read_parquet_columns(path, requested_cols)

    required = ["cell_id_int", "cells_final_label_group"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Slide metadata {path} is missing required columns: {missing}")

    meta = meta.set_index("cell_id_int", drop=False)
    return meta


def extract_bytes(value: Any) -> bytes:
    """
    Handle common HF parquet representations:
    - {"bytes": ...}
    - raw bytes
    - bytearray
    - memoryview
    """
    if isinstance(value, dict):
        if "bytes" not in value:
            raise ValueError(f"Dictionary does not contain a 'bytes' field: keys={value.keys()}")
        value = value["bytes"]

    if isinstance(value, memoryview):
        return value.tobytes()

    if isinstance(value, bytearray):
        return bytes(value)

    if isinstance(value, bytes):
        return value

    raise TypeError(f"Cannot extract bytes from object of type {type(value)}")


def decode_png_bytes(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.asarray(img)


def decode_cell_id_map(npz_bytes: bytes) -> np.ndarray:
    return sp.load_npz(io.BytesIO(npz_bytes)).toarray().astype(np.int32, copy=False)


def map_label_to_five_class(label: Any, cell_id_int: int) -> str:
    if int(cell_id_int) == 0:
        return "Background"

    if pd.isna(label):
        return "Other"

    return FIVE_CLASS_MAP.get(str(label), "Other")


def build_type_map(cell_id_map: np.ndarray, slide_meta: pd.DataFrame) -> np.ndarray:
    """
    Build dense per-pixel type map:

        0 = Background
        1 = Immune
        2 = Stromal
        3 = Epithelial
        4 = Other
    """
    ids_all, inv = np.unique(cell_id_map, return_inverse=True)

    labels_for_ids = (
        slide_meta["cells_final_label_group"]
        .reindex(ids_all)
        .to_numpy()
    )

    class_names = [
        map_label_to_five_class(label, int(cell_id_int))
        for cell_id_int, label in zip(ids_all, labels_for_ids)
    ]

    lut = np.array([CLASS_TO_INT[name] for name in class_names], dtype=np.uint8)
    type_map = lut[inv].reshape(cell_id_map.shape)
    return type_map


def save_sparse_pair_npz(out_path: Path, inst_map: np.ndarray, type_map: np.ndarray) -> None:
    inst_sparse = csr_matrix(inst_map.astype(np.int32))
    type_sparse = csr_matrix(type_map.astype(np.int32))

    np.savez_compressed(
        out_path,
        inst_map_data=inst_sparse.data,
        inst_map_indices=inst_sparse.indices,
        inst_map_indptr=inst_sparse.indptr,
        inst_map_shape=np.array(inst_sparse.shape, dtype=np.int32),
        type_map_data=type_sparse.data,
        type_map_indices=type_sparse.indices,
        type_map_indptr=type_sparse.indptr,
        type_map_shape=np.array(type_sparse.shape, dtype=np.int32),
    )


def compute_cell_counts(inst_map: np.ndarray, type_map: np.ndarray) -> dict[str, int]:
    counts = {name: 0 for name in COUNT_CLASS_NAMES}

    ids = np.unique(inst_map)
    ids = ids[ids != 0]

    for cell_id in ids:
        vals = type_map[inst_map == cell_id]
        vals = vals[vals != 0]

        if len(vals) == 0:
            continue

        uniq, cnt = np.unique(vals, return_counts=True)
        cls = int(uniq[np.argmax(cnt)])

        if cls == 1:
            counts["Immune"] += 1
        elif cls == 2:
            counts["Stromal"] += 1
        elif cls == 3:
            counts["Epithelial"] += 1
        elif cls == 4:
            counts["Other"] += 1

    return counts


# ============================================================
# Split helpers
# ============================================================

def validate_fractions(train_frac: float, valid_frac: float, test_frac: float) -> None:
    if train_frac <= 0 or valid_frac <= 0 or test_frac <= 0:
        raise ValueError("train_frac, valid_frac and test_frac must all be > 0")

    if not np.isclose(train_frac + valid_frac + test_frac, 1.0):
        raise ValueError(
            "train_frac + valid_frac + test_frac must sum to 1. "
            f"Got {train_frac + valid_frac + test_frac}"
        )


def assign_spatial_split_single_slide(
    df_slide: pd.DataFrame,
    axis: str,
    train_frac: float,
    valid_frac: float,
    test_frac: float,
    boundary_margin: int,
) -> pd.DataFrame:
    if axis not in {"x", "y"}:
        raise ValueError("split axis must be 'x' or 'y'")

    validate_fractions(train_frac, valid_frac, test_frac)

    df = df_slide.copy()

    df["x_center"] = 0.5 * (df["xmin"] + df["xmax"])
    df["y_center"] = 0.5 * (df["ymin"] + df["ymax"])

    coord_col = "x_center" if axis == "x" else "y_center"
    coord = df[coord_col]

    cmin = float(coord.min())
    cmax = float(coord.max())
    span = cmax - cmin

    if span <= 0:
        raise ValueError(
            f"Degenerate spatial span for slide {df['slide_id'].iloc[0]} "
            f"on axis {axis}: min={cmin}, max={cmax}"
        )

    b1 = cmin + train_frac * span
    b2 = cmin + (train_frac + valid_frac) * span

    df["split"] = "discard"

    df.loc[coord <= (b1 - boundary_margin), "split"] = "train"
    df.loc[
        (coord >= (b1 + boundary_margin)) & (coord <= (b2 - boundary_margin)),
        "split",
    ] = "valid"
    df.loc[coord >= (b2 + boundary_margin), "split"] = "test"

    return df


def assign_spatial_split(
    patch_info: pd.DataFrame,
    axis: str,
    train_frac: float,
    valid_frac: float,
    test_frac: float,
    boundary_margin: int,
) -> pd.DataFrame:
    """
    Spatial split applied independently inside each slide.
    This avoids mixing coordinates from different slides.
    """
    parts = []

    for slide_id, df_slide in patch_info.groupby("slide_id", sort=True):
        split_slide = assign_spatial_split_single_slide(
            df_slide=df_slide,
            axis=axis,
            train_frac=train_frac,
            valid_frac=valid_frac,
            test_frac=test_frac,
            boundary_margin=boundary_margin,
        )
        parts.append(split_slide)

    out = pd.concat(parts, axis=0, ignore_index=True)
    return out


def assign_train_valid_inside_train_slides(
    df_train_source: pd.DataFrame,
    axis: str,
    train_frac: float,
    valid_frac: float,
    boundary_margin: int,
) -> pd.DataFrame:
    """
    Used for 2-slide slide-level setting:
    - one or more train-source slides are split into train/valid spatially
    - test slide remains entirely test
    """
    if train_frac <= 0 or valid_frac <= 0:
        raise ValueError("train_frac and valid_frac must be > 0")

    if not np.isclose(train_frac + valid_frac, 1.0):
        raise ValueError("train_frac + valid_frac must sum to 1")

    parts = []

    for slide_id, df_slide in df_train_source.groupby("slide_id", sort=True):
        df = df_slide.copy()

        df["x_center"] = 0.5 * (df["xmin"] + df["xmax"])
        df["y_center"] = 0.5 * (df["ymin"] + df["ymax"])

        coord_col = "x_center" if axis == "x" else "y_center"
        coord = df[coord_col]

        cmin = float(coord.min())
        cmax = float(coord.max())
        span = cmax - cmin

        if span <= 0:
            raise ValueError(
                f"Degenerate spatial span for slide {slide_id} on axis {axis}: "
                f"min={cmin}, max={cmax}"
            )

        boundary = cmin + train_frac * span

        df["split"] = "discard"
        df.loc[coord <= (boundary - boundary_margin), "split"] = "train"
        df.loc[coord >= (boundary + boundary_margin), "split"] = "valid"

        parts.append(df)

    return pd.concat(parts, axis=0, ignore_index=True)


def infer_slide_split_lists(
    selected_slides: list[str],
    train_slides: Optional[list[str]],
    valid_slides: Optional[list[str]],
    test_slides: Optional[list[str]],
    train_frac: float,
    valid_frac: float,
    test_frac: float,
    random_seed: int,
) -> tuple[list[str], list[str], list[str], bool]:
    """
    Returns:
        train_slides, valid_slides, test_slides, uses_spatial_valid_inside_train

    Logic:
    - explicit split: use user-provided train/valid/test slides
    - 2 slides: one train-source slide, one test slide; validation carved spatially from train
    - >=3 slides: split whole slides according to train/valid/test fractions
    """
    selected_slides = sorted(str(s) for s in selected_slides)

    if train_slides or valid_slides or test_slides:
        train_slides = list(train_slides or [])
        valid_slides = list(valid_slides or [])
        test_slides = list(test_slides or [])

        provided = set(train_slides) | set(valid_slides) | set(test_slides)

        if len(train_slides) + len(valid_slides) + len(test_slides) != len(provided):
            raise ValueError("--train-slides, --valid-slides and --test-slides must be disjoint")

        if provided != set(selected_slides):
            raise ValueError(
                "Explicit slide split does not match selected slides.\n"
                f"selected_slides={selected_slides}\n"
                f"provided={sorted(provided)}"
            )

        uses_spatial_valid = len(valid_slides) == 0
        return train_slides, valid_slides, test_slides, uses_spatial_valid

    if len(selected_slides) < 2:
        raise ValueError("Slide-level split requires at least two selected slides")

    if len(selected_slides) == 2:
        return [selected_slides[0]], [], [selected_slides[1]], True

    validate_fractions(train_frac, valid_frac, test_frac)

    rng = np.random.default_rng(random_seed)
    slides = np.array(selected_slides)
    rng.shuffle(slides)

    n = len(slides)

    n_test = max(1, int(round(test_frac * n)))
    n_valid = max(1, int(round(valid_frac * n)))

    if n_test + n_valid >= n:
        n_test = 1
        n_valid = 1

    n_train = n - n_valid - n_test

    train_slides = sorted(slides[:n_train].tolist())
    valid_slides = sorted(slides[n_train:n_train + n_valid].tolist())
    test_slides = sorted(slides[n_train + n_valid:].tolist())

    return train_slides, valid_slides, test_slides, False

def assign_slide_split(
    patch_info: pd.DataFrame,
    axis: str,
    train_frac: float,
    valid_frac: float,
    test_frac: float,
    train_frac_inside_train_slide: float,
    valid_frac_inside_train_slide: float,
    boundary_margin: int,
    train_slides: Optional[list[str]],
    valid_slides: Optional[list[str]],
    test_slides: Optional[list[str]],
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_slides = sorted(patch_info["slide_id"].astype(str).unique().tolist())

    train_slides, valid_slides, test_slides, uses_spatial_valid = infer_slide_split_lists(
        selected_slides=selected_slides,
        train_slides=train_slides,
        valid_slides=valid_slides,
        test_slides=test_slides,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        random_seed=args.random_seed,
    )

    df = patch_info.copy()
    df["split"] = "unassigned"

    if len(valid_slides) > 0:
        df.loc[df["slide_id"].isin(train_slides), "split"] = "train"
        df.loc[df["slide_id"].isin(valid_slides), "split"] = "valid"
        df.loc[df["slide_id"].isin(test_slides), "split"] = "test"

        if (df["split"] == "unassigned").any():
            raise ValueError("Some patches remained unassigned in slide split")

        manifest_extra = {
            "train_slides": train_slides,
            "valid_slides": valid_slides,
            "test_slides": test_slides,
            "uses_spatial_valid_inside_train_slide": False,
        }
        return df, manifest_extra

    # No explicit valid slide: carve valid spatially from train slides.
    df_train_source = df[df["slide_id"].isin(train_slides)].copy()
    df_test_source = df[df["slide_id"].isin(test_slides)].copy()

    df_train_valid = assign_train_valid_inside_train_slides(
        df_train_source=df_train_source,
        axis=axis,
        train_frac=train_frac_inside_train_slide,
        valid_frac=valid_frac_inside_train_slide,
        boundary_margin=boundary_margin,
    )

    df_test_source = df_test_source.copy()
    df_test_source["split"] = "test"

    out = pd.concat([df_train_valid, df_test_source], axis=0, ignore_index=True)

    manifest_extra = {
        "train_slides": train_slides,
        "valid_slides": valid_slides,
        "test_slides": test_slides,
        "uses_spatial_valid_inside_train_slide": True,
        "train_frac_inside_train_slide": train_frac_inside_train_slide,
        "valid_frac_inside_train_slide": valid_frac_inside_train_slide,
    }

    return out, manifest_extra


def assign_baseline_random_split(
    patch_info: pd.DataFrame,
    train_frac: float,
    valid_frac: float,
    test_frac: float,
    random_seed: int,
) -> pd.DataFrame:
    validate_fractions(train_frac, valid_frac, test_frac)

    df = patch_info.copy()
    rng = np.random.default_rng(random_seed)

    indices = np.arange(len(df))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(round(train_frac * n))
    n_valid = int(round(valid_frac * n))

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    df["split"] = "unassigned"
    df.loc[train_idx, "split"] = "train"
    df.loc[valid_idx, "split"] = "valid"
    df.loc[test_idx, "split"] = "test"

    return df


def cap_per_split(
    patch_info: pd.DataFrame,
    max_per_split: Optional[int],
    random_seed: int,
) -> pd.DataFrame:
    if max_per_split is None:
        return patch_info.copy()

    parts = []

    for split_name in ["train", "valid", "test"]:
        df_split = patch_info[patch_info["split"] == split_name].copy()

        if len(df_split) > max_per_split:
            df_split = df_split.sample(n=max_per_split, random_state=random_seed)

        parts.append(df_split)

    out = pd.concat(parts, axis=0, ignore_index=True)
    return out


def assign_split(args: argparse.Namespace, patch_info: pd.DataFrame) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    selected_slides = sorted(patch_info["slide_id"].astype(str).unique().tolist())
    strategy = args.strategy

    if strategy == "auto":
        if len(selected_slides) >= 2:
            strategy_actual = "slide"
        else:
            strategy_actual = "spatial"
    else:
        strategy_actual = strategy

    manifest_extra: dict[str, Any] = {}

    if strategy_actual == "spatial":
        out = assign_spatial_split(
            patch_info=patch_info,
            axis=args.split_axis,
            train_frac=args.train_frac,
            valid_frac=args.valid_frac,
            test_frac=args.test_frac,
            boundary_margin=args.boundary_margin,
        )

    elif strategy_actual == "slide":
        out, manifest_extra = assign_slide_split(
            patch_info=patch_info,
            axis=args.split_axis,
            train_frac=args.train_frac,
            valid_frac=args.valid_frac,
            test_frac=args.test_frac,
            train_frac_inside_train_slide=args.train_frac_inside_train_slide,
            valid_frac_inside_train_slide=args.valid_frac_inside_train_slide,
            boundary_margin=args.boundary_margin,
            train_slides=args.train_slides,
            valid_slides=args.valid_slides,
            test_slides=args.test_slides,
            random_seed=args.random_seed,
        )

    elif strategy_actual == "baseline":
        out = assign_baseline_random_split(
            patch_info=patch_info,
            train_frac=args.train_frac,
            valid_frac=args.valid_frac,
            test_frac=args.test_frac,
            random_seed=args.random_seed,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy_actual}")

    out = out[out["split"].isin(["train", "valid", "test", "discard"])].copy()

    print("\nSplit counts before discard:")
    print(out["split"].value_counts(dropna=False).sort_index())

    out = out[out["split"].isin(["train", "valid", "test"])].copy()

    print("\nSplit counts after discard:")
    print(out["split"].value_counts(dropna=False).sort_index())

    out = cap_per_split(out, max_per_split=args.max_per_split, random_seed=args.random_seed)

    print("\nSplit counts after optional cap:")
    print(out["split"].value_counts(dropna=False).sort_index())

    for split_name in ["train", "valid", "test"]:
        if int((out["split"] == split_name).sum()) == 0:
            raise ValueError(
                f"Split '{split_name}' is empty. "
                "Try reducing --boundary-margin, using more patches/slides, "
                "or changing --split-axis."
            )

    return out, strategy_actual, manifest_extra


# ============================================================
# Parquet row recovery
# ============================================================

def iter_shard_paths(sthelar_root: Path) -> list[Path]:
    shards_dir = sthelar_root / "data"

    if not shards_dir.exists():
        raise FileNotFoundError(f"Missing data shard directory: {shards_dir}")

    shard_paths = sorted(shards_dir.glob("train-*.parquet"))

    if len(shard_paths) == 0:
        shard_paths = sorted(shards_dir.glob("*.parquet"))

    if len(shard_paths) == 0:
        raise FileNotFoundError(f"No parquet shards found under {shards_dir}")

    return shard_paths


def read_shard_table(shard_path: Path) -> pd.DataFrame:
    requested_cols = ["file_name", "slide_id", "tissue", "image", "cell_id_map"]
    schema_names = set(pq.read_schema(shard_path).names)
    cols = [c for c in requested_cols if c in schema_names]

    required = ["file_name", "slide_id", "image", "cell_id_map"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Shard {shard_path} is missing required columns: {missing}")

    table = pq.read_table(shard_path, columns=cols)
    df = table.to_pandas()

    df["slide_id"] = df["slide_id"].astype(str)
    df["file_name"] = df["file_name"].astype(str)

    return df


def recover_rows_by_key(
    sthelar_root: Path,
    selected_keys: set[tuple[str, str]],
    selected_original_names: set[str],
) -> dict[tuple[str, str], pd.Series]:
    rows_by_key: dict[tuple[str, str], pd.Series] = {}

    shard_paths = iter_shard_paths(sthelar_root)
    print(f"\nScanning {len(shard_paths)} parquet shards to recover selected rows...")

    for shard_path in shard_paths:
        if len(rows_by_key) == len(selected_keys):
            break

        print(f"  Reading shard: {shard_path.name}")
        df = read_shard_table(shard_path)

        df = df[df["file_name"].isin(selected_original_names)].copy()
        if len(df) == 0:
            continue

        for _, row in df.iterrows():
            key = make_patch_key(row)
            if key in selected_keys:
                rows_by_key[key] = row

        print(f"    recovered so far: {len(rows_by_key)}/{len(selected_keys)}")

    missing = [key for key in selected_keys if key not in rows_by_key]
    if missing:
        raise RuntimeError(f"Missing parquet rows for these keys: {missing[:10]}")

    print("Recovered all selected original parquet rows.")
    return rows_by_key


# ============================================================
# Main conversion
# ============================================================

def convert(args: argparse.Namespace) -> None:
    print("\nEffective configuration:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")
    
    sthelar_root = as_path(args.sthelar_root)
    output_root = as_path(args.output_root)

    if not sthelar_root.exists():
        raise FileNotFoundError(f"STHELAR root does not exist: {sthelar_root}")
    
    prepare_output_root(output_root, overwrite=args.overwrite)

    tmp_root = output_root / "_tmp"
    tmp_images_dir = tmp_root / "images"
    tmp_labels_dir = tmp_root / "labels"

    tmp_images_dir.mkdir(parents=True, exist_ok=True)
    tmp_labels_dir.mkdir(parents=True, exist_ok=True)

    images_zip = output_root / "images.zip"
    labels_zip = output_root / "labels.zip"
    types_csv = output_root / "types.csv"
    train_csv = output_root / "cell_count_train.csv"
    valid_csv = output_root / "cell_count_valid.csv"
    test_csv = output_root / "cell_count_test.csv"
    dataset_config_yaml = output_root / "dataset_config.yaml"
    patch_info_with_split_csv = output_root / "patch_info_with_split.csv"
    split_manifest_yaml = output_root / "split_manifest.yaml"

    overview_path = find_overview_path(sthelar_root, args.overview_path)

    print(f"STHELAR root: {sthelar_root}")
    print(f"Overview: {overview_path}")
    print(f"Output root: {output_root}")

    patch_info = load_patch_overview(
        overview_path=overview_path,
        tissue=args.tissue,
        slide_ids=args.slide_ids,
    )

    patch_info = apply_max_patches_per_slide(
        patch_info=patch_info,
        max_patches_per_slide=args.max_patches_per_slide,
        random_seed=args.random_seed,
    )

    selected_slides = sorted(patch_info["slide_id"].unique().tolist())

    print("\nSelected slides:")
    for slide_id in selected_slides:
        n_slide = int((patch_info["slide_id"] == slide_id).sum())
        print(f"  {slide_id}: {n_slide} patches")

    # Unique CellViT archive names.
    patch_info["packed_file_name"] = patch_info.apply(make_patch_uid, axis=1)
    patch_info["packed_label_name"] = patch_info["packed_file_name"].apply(image_to_label_name)

    if patch_info["packed_file_name"].duplicated().any():
        dup = (
            patch_info.loc[patch_info["packed_file_name"].duplicated(), "packed_file_name"]
            .head(10)
            .tolist()
        )
        raise ValueError(f"Duplicate packed_file_name values detected: {dup}")

    patch_info, strategy_actual, manifest_extra = assign_split(args, patch_info)

    patch_info["key_tuple"] = list(
        zip(patch_info["slide_id"].astype(str), patch_info["file_name"].astype(str))
    )

    selected_keys = set(patch_info["key_tuple"].tolist())
    selected_original_names = set(patch_info["file_name"].astype(str).tolist())

    file_to_split = dict(zip(patch_info["key_tuple"], patch_info["split"]))
    file_to_packed_name = dict(zip(patch_info["key_tuple"], patch_info["packed_file_name"]))
    file_to_packed_label = dict(zip(patch_info["key_tuple"], patch_info["packed_label_name"]))

    rows_by_key = recover_rows_by_key(
        sthelar_root=sthelar_root,
        selected_keys=selected_keys,
        selected_original_names=selected_original_names,
    )

    slide_meta_by_id = {
        slide_id: load_slide_meta(sthelar_root, slide_id)
        for slide_id in sorted(patch_info["slide_id"].unique().tolist())
    }

    type_rows = []
    count_rows = []
    selected_image_entries: list[tuple[Path, str]] = []
    selected_label_entries: list[tuple[Path, str]] = []

    n_total = len(patch_info)

    print(f"\nPacking {n_total} selected patches...")

    for i, info_row in enumerate(patch_info.itertuples(index=False), start=1):
        slide_id = str(info_row.slide_id)
        file_name = str(info_row.file_name)
        key = (slide_id, file_name)

        row = rows_by_key[key]
        split_value = file_to_split[key]
        packed_file_name = file_to_packed_name[key]
        packed_label_name = file_to_packed_label[key]

        image_bytes = extract_bytes(row["image"])
        cell_id_map_bytes = extract_bytes(row["cell_id_map"])

        rgb = decode_png_bytes(image_bytes)
        cell_id_map = decode_cell_id_map(cell_id_map_bytes)

        slide_meta = slide_meta_by_id[slide_id]
        type_map = build_type_map(cell_id_map, slide_meta).astype(np.int32)

        if cell_id_map.shape != type_map.shape:
            raise ValueError(
                f"Shape mismatch for {key}: "
                f"inst={cell_id_map.shape}, type={type_map.shape}"
            )

        out_image_path = tmp_images_dir / packed_file_name
        out_label_path = tmp_labels_dir / packed_label_name

        Image.fromarray(rgb).save(out_image_path)
        save_sparse_pair_npz(out_label_path, cell_id_map, type_map)

        counts = compute_cell_counts(cell_id_map, type_map)

        type_rows.append(
            {
                "img": packed_file_name,
                "type": args.tissue_name,
            }
        )

        count_rows.append(
            {
                "Image": packed_file_name,
                "Immune": counts["Immune"],
                "Stromal": counts["Stromal"],
                "Epithelial": counts["Epithelial"],
                "Other": counts["Other"],
                "split": split_value,
            }
        )

        selected_image_entries.append((out_image_path, packed_file_name))
        selected_label_entries.append((out_label_path, packed_label_name))

        if i % 25 == 0 or i == 1 or i == n_total:
            print(f"  Packed {i}/{n_total}: {slide_id}/{file_name} -> {packed_file_name}")

    # Save metadata CSV files.
    df_types = pd.DataFrame(type_rows)
    df_types.to_csv(types_csv, index=False)

    df_counts = pd.DataFrame(count_rows)

    df_train = df_counts[df_counts["split"] == "train"].drop(columns=["split"]).copy()
    df_valid = df_counts[df_counts["split"] == "valid"].drop(columns=["split"]).copy()
    df_test = df_counts[df_counts["split"] == "test"].drop(columns=["split"]).copy()

    df_train.to_csv(train_csv, index=False)
    df_valid.to_csv(valid_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    # Save patch_info with enough traceability for debugging and thesis.
    cols_front = [
        "slide_id",
        "file_name",
        "packed_file_name",
        "packed_label_name",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "split",
    ]
    remaining_cols = [c for c in patch_info.columns if c not in cols_front and c != "key_tuple"]
    patch_info_out = patch_info[cols_front + remaining_cols].copy()
    patch_info_out.to_csv(patch_info_with_split_csv, index=False)

    dataset_config = {
        "tissue_types": {
            args.tissue_name: 0,
        },
        "nuclei_types": NUCLEI_TYPES,
    }

    with open(dataset_config_yaml, "w") as f:
        yaml.safe_dump(dataset_config, f, sort_keys=False)

    split_manifest = {
        "source": "sthelar_huggingface_parquet",
        "sthelar_root": str(sthelar_root),
        "overview_path": str(overview_path),
        "output_root": str(output_root),
        "tissue": args.tissue,
        "tissue_name": args.tissue_name,
        "requested_strategy": args.strategy,
        "actual_strategy": strategy_actual,
        "selected_slides": selected_slides,
        "split_axis": args.split_axis,
        "boundary_margin": args.boundary_margin,
        "train_frac": args.train_frac,
        "valid_frac": args.valid_frac,
        "test_frac": args.test_frac,
        "max_patches_per_slide": args.max_patches_per_slide,
        "max_per_split": args.max_per_split,
        "random_seed": args.random_seed,
        "counts_after_split": {
            "train": int((patch_info["split"] == "train").sum()),
            "valid": int((patch_info["split"] == "valid").sum()),
            "test": int((patch_info["split"] == "test").sum()),
            "total": int(len(patch_info)),
        },
        "slides_by_split": {
            split_name: sorted(
                patch_info.loc[
                    patch_info["split"] == split_name,
                    "slide_id"
                ].dropna().astype(str).unique().tolist()
            )
            for split_name in ["train", "valid", "test"]
        },
        **manifest_extra,
    }

    with open(split_manifest_yaml, "w") as f:
        yaml.safe_dump(split_manifest, f, sort_keys=False)

    print("\nCreating images.zip ...")
    make_zip_from_entries(selected_image_entries, images_zip)

    print("Creating labels.zip ...")
    make_zip_from_entries(selected_label_entries, labels_zip)

    if not args.keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("\nDone.")
    print(f"Created: {images_zip}")
    print(f"Created: {labels_zip}")
    print(f"Created: {types_csv}")
    print(f"Created: {train_csv}")
    print(f"Created: {valid_csv}")
    print(f"Created: {test_csv}")
    print(f"Created: {dataset_config_yaml}")
    print(f"Created: {patch_info_with_split_csv}")
    print(f"Created: {split_manifest_yaml}")


# ============================================================
# CLI
# ============================================================

def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert STHELAR Hugging Face parquet data to CellViT-ready format."
    )

    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file.")

    parser.add_argument(
        "--sthelar-root",
        type=str,
        default=defaults.get("sthelar_root"),
        help="Root directory of STHELAR HF dataset, e.g. /path/STHELAR_20x.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=defaults.get("output_root"),
        help="Output CellViT-ready dataset directory.",
    )
    parser.add_argument(
        "--overview-path",
        type=str,
        default=defaults.get("overview_path"),
        help="Optional explicit path to patches_overview_sthelar*.parquet.",
    )

    parser.add_argument(
        "--tissue",
        type=str,
        default=defaults.get("tissue"),
        help="Tissue prefix used to infer slide IDs, e.g. tonsil, breast, brain.",
    )
    parser.add_argument(
        "--tissue-name",
        type=str,
        default=defaults.get("tissue_name"),
        help="Tissue name written in types.csv and dataset_config.yaml, e.g. Tonsil.",
    )
    parser.add_argument(
        "--slide-ids",
        nargs="*",
        default=defaults.get("slide_ids"),
        help="Optional explicit list of slide IDs, e.g. tonsil_s0 tonsil_s1.",
    )

    parser.add_argument(
        "--strategy",
        choices=["auto", "baseline", "spatial", "slide"],
        default=defaults.get("strategy", "auto"),
        help="Split strategy.",
    )

    parser.add_argument(
        "--split-axis",
        choices=["x", "y"],
        default=defaults.get("split_axis", "x"),
        help="Spatial split axis.",
    )
    parser.add_argument(
        "--boundary-margin",
        type=int,
        default=defaults.get("boundary_margin", 128),
        help="Boundary margin in coordinate units. Patches close to split borders are discarded.",
    )

    parser.add_argument(
        "--train-frac",
        type=float,
        default=defaults.get("train_frac", 0.70),
        help="Train fraction for baseline/spatial split.",
    )
    parser.add_argument(
        "--valid-frac",
        type=float,
        default=defaults.get("valid_frac", 0.15),
        help="Validation fraction for baseline/spatial split.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=defaults.get("test_frac", 0.15),
        help="Test fraction for baseline/spatial split.",
    )

    parser.add_argument(
        "--train-frac-inside-train-slide",
        type=float,
        default=defaults.get("train_frac_inside_train_slide", 0.85),
        help="Used in slide strategy when validation is carved spatially from train slide.",
    )
    parser.add_argument(
        "--valid-frac-inside-train-slide",
        type=float,
        default=defaults.get("valid_frac_inside_train_slide", 0.15),
        help="Used in slide strategy when validation is carved spatially from train slide.",
    )

    parser.add_argument(
        "--train-slides",
        nargs="*",
        default=defaults.get("train_slides"),
        help="Optional explicit train slide IDs for slide strategy.",
    )
    parser.add_argument(
        "--valid-slides",
        nargs="*",
        default=defaults.get("valid_slides"),
        help="Optional explicit valid slide IDs for slide strategy.",
    )
    parser.add_argument(
        "--test-slides",
        nargs="*",
        default=defaults.get("test_slides"),
        help="Optional explicit test slide IDs for slide strategy.",
    )

    parser.add_argument(
        "--max-patches-per-slide",
        type=int,
        default=defaults.get("max_patches_per_slide"),
        help="Optional debug cap applied before split assignment.",
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=defaults.get("max_per_split"),
        help="Optional cap applied after split assignment.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=defaults.get("random_seed", 42),
        help="Random seed.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=bool(defaults.get("overwrite", False)),
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        default=bool(defaults.get("keep_tmp", False)),
        help="Keep temporary decoded images/labels under output_root/_tmp.",
    )

    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    defaults = load_yaml_config(pre_args.config)

    parser = build_parser(defaults)
    args = parser.parse_args()

    if args.sthelar_root is None:
        parser.error("--sthelar-root is required, either via CLI or YAML config.")

    if args.output_root is None:
        parser.error("--output-root is required, either via CLI or YAML config.")

    if args.tissue is None and not args.slide_ids:
        parser.error("Please provide either --tissue or --slide-ids.")

    if args.tissue_name is None:
        if args.tissue is not None:
            args.tissue_name = str(args.tissue).replace("_", " ").title().replace(" ", "")
        else:
            args.tissue_name = "Unknown"

    if args.boundary_margin < 0:
        parser.error("--boundary-margin must be >= 0")

    if args.max_patches_per_slide is not None and args.max_patches_per_slide <= 0:
        parser.error("--max-patches-per-slide must be > 0")

    if args.max_per_split is not None and args.max_per_split <= 0:
        parser.error("--max-per-split must be > 0")

    return args


def main() -> None:
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()