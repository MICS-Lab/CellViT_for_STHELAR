#!/usr/bin/env python3
"""
Inspect raw STHELAR Hugging Face parquet dataset.

This script inspects the original STHELAR dataset before conversion into
CellViT/PanNuke-like format.

It checks:
- patches_overview_sthelar{20,40}x.parquet
- data/train-*.parquet shards
- cell_metadata/{slide_id}_cell_metadata.parquet
- decoding of H&E image bytes
- decoding of cell_id_map
- join between cell_id_map cell_id_int values and biological labels
- optional visualization of one raw patch

Example on local T9:

python preprocessing/sthelar/inspect_raw_dataset.py \
  --sthelar-root /Volumes/T9/Datasets/STHELAR_20x \
  --magnification 20 \
  --slide-id tonsil_s0 \
  --row-index 0 \
  --save-figure /Volumes/T9/Datasets/raw_tonsil_s0_patch0.png

Example on Ruche:

python preprocessing/sthelar/inspect_raw_dataset.py \
  --sthelar-root /gpfs/workdir/taddeial/workspace/Datasets/STHELAR_20x \
  --magnification 20 \
  --slide-id tonsil_s0 \
  --row-index 0 \
  --save-figure /gpfs/workdir/taddeial/workspace/raw_tonsil_s0_patch0.png
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy import sparse
import pyarrow.parquet as pq


LABELS_9 = [
    "Epithelial",
    "Blood_vessel",
    "Fibroblast_Myofibroblast",
    "Myeloid",
    "B_Plasma",
    "T_NK",
    "Melanocyte",
    "Specialized",
    "Other",
]

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

LABELS_5 = ["Immune", "Stromal", "Epithelial", "Other"]


def get_clean_parquet_files(path: Path) -> list[Path]:
    """
    Return parquet files while ignoring macOS AppleDouble files such as ._xxx.parquet.
    """
    return sorted(
        p for p in path.glob("*.parquet")
        if not p.name.startswith("._")
    )


def decode_image_bytes(img_obj) -> np.ndarray:
    """
    Decode Hugging Face image bytes.

    Depending on how parquet is read, image can be:
    - dict with key "bytes"
    - raw bytes
    """
    if isinstance(img_obj, dict):
        img_bytes = img_obj["bytes"]
    else:
        img_bytes = img_obj

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.asarray(img)


def decode_cell_id_map(cell_map_obj) -> np.ndarray:
    """
    Decode STHELAR cell_id_map stored as sparse npz bytes.

    The field can be:
    - dict with key "bytes"
    - raw bytes

    Usually scipy.sparse.load_npz works directly.
    """
    if isinstance(cell_map_obj, dict):
        cell_map_bytes = cell_map_obj["bytes"]
    else:
        cell_map_bytes = cell_map_obj

    try:
        cell_map = sparse.load_npz(io.BytesIO(cell_map_bytes)).toarray()
    except Exception:
        # Fallback for manually saved csr components.
        with io.BytesIO(cell_map_bytes) as f:
            loader = np.load(f, allow_pickle=True)
            cell_map_sparse = sparse.csr_matrix(
                (loader["data"], loader["indices"], loader["indptr"]),
                shape=loader["shape"],
            )
            cell_map = cell_map_sparse.toarray()

    return cell_map.astype(np.uint32, copy=False)


def load_slide_metadata(cell_metadata_dir: Path, slide_id: str) -> pd.DataFrame:
    meta_path = cell_metadata_dir / f"{slide_id}_cell_metadata.parquet"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing cell metadata file: {meta_path}")

    meta = pd.read_parquet(meta_path)

    required = {"cell_id_int", "cells_final_label_group"}
    missing = required.difference(meta.columns)

    if missing:
        raise KeyError(
            f"Missing required columns in {meta_path}: {sorted(missing)}. "
            f"Available columns: {list(meta.columns)}"
        )

    return meta.set_index("cell_id_int", drop=False)


def find_row_in_shards(
    shards: list[Path],
    slide_id: Optional[str],
    file_name: Optional[str],
    row_index: int,
) -> pd.Series:
    """
    Find a row in parquet shards.

    Priority:
    1. If file_name is provided, search exact file_name.
    2. Else if slide_id is provided, take row_index inside that slide.
    3. Else take global row_index across all shards.
    """
    current_global = 0
    current_slide = 0

    for shard_path in shards:
        table = pq.read_table(shard_path)
        df = table.to_pandas()

        if file_name is not None:
            match = df[df["file_name"] == file_name]
            if len(match) > 0:
                return match.iloc[0]

        if slide_id is not None:
            df_slide = df[df["slide_id"] == slide_id]
            if current_slide + len(df_slide) > row_index:
                return df_slide.iloc[row_index - current_slide]
            current_slide += len(df_slide)

        if slide_id is None and file_name is None:
            if current_global + len(df) > row_index:
                return df.iloc[row_index - current_global]
            current_global += len(df)

    if file_name is not None:
        raise FileNotFoundError(f"file_name not found in shards: {file_name}")

    if slide_id is not None:
        raise IndexError(
            f"row_index={row_index} out of range for slide_id={slide_id}. "
            f"Found {current_slide} rows."
        )

    raise IndexError(
        f"row_index={row_index} out of range for dataset. "
        f"Found {current_global} rows."
    )


def build_type_map_from_metadata(
    cell_id_map: np.ndarray,
    slide_meta: pd.DataFrame,
    mode: str,
) -> tuple[np.ndarray, dict[int, str]]:
    """
    Convert cell_id_map into pixel-wise biological type map.

    cell_id_map:
        pixel -> cell_id_int

    slide_meta:
        cell_id_int -> cells_final_label_group

    output:
        type_map:
            pixel -> class integer
    """
    ids_all, inv = np.unique(cell_id_map, return_inverse=True)

    labels_for_ids = (
        slide_meta["cells_final_label_group"]
        .reindex(ids_all)
        .fillna("Background")
        .to_numpy()
    )

    if mode == "9class":
        class_names = {0: "Background"}
        class_names.update({i + 1: lab for i, lab in enumerate(LABELS_9)})

        label_to_int = {name: idx for idx, name in class_names.items()}

        mapped_labels = []
        for cid, lab in zip(ids_all, labels_for_ids):
            if cid == 0:
                mapped_labels.append("Background")
            elif lab in label_to_int:
                mapped_labels.append(lab)
            else:
                mapped_labels.append("Background")

    elif mode == "5class":
        class_names = {
            0: "Background",
            1: "Immune",
            2: "Stromal",
            3: "Epithelial",
            4: "Other",
        }

        label_to_int = {name: idx for idx, name in class_names.items()}

        mapped_labels = []
        for cid, lab in zip(ids_all, labels_for_ids):
            if cid == 0:
                mapped_labels.append("Background")
            else:
                mapped_labels.append(FIVE_CLASS_MAP.get(lab, "Other"))

    else:
        raise ValueError("mode must be either '5class' or '9class'")

    lut = np.array([label_to_int.get(label, 0) for label in mapped_labels], dtype=np.uint8)
    type_map = lut[inv].reshape(cell_id_map.shape)

    return type_map, class_names


def compute_instance_boundary(inst_map: np.ndarray) -> np.ndarray:
    boundary = np.zeros(inst_map.shape, dtype=bool)

    boundary[1:, :] |= inst_map[1:, :] != inst_map[:-1, :]
    boundary[:-1, :] |= inst_map[:-1, :] != inst_map[1:, :]
    boundary[:, 1:] |= inst_map[:, 1:] != inst_map[:, :-1]
    boundary[:, :-1] |= inst_map[:, :-1] != inst_map[:, 1:]

    boundary &= inst_map != 0
    return boundary


def make_overlay(
    rgb: np.ndarray,
    cell_id_map: np.ndarray,
    type_map: np.ndarray,
    class_names: dict[int, str],
    alpha: float,
) -> np.ndarray:
    boundary = compute_instance_boundary(cell_id_map)

    cmap = plt.get_cmap("tab10", max(class_names.keys()) + 1)
    lut = (cmap(np.arange(max(class_names.keys()) + 1))[:, :3] * 255).astype(np.uint8)

    overlay = rgb.copy().astype(np.float32)

    type_values = type_map[boundary]
    type_values = np.clip(type_values, 0, len(lut) - 1)

    overlay[boundary] = (1.0 - alpha) * overlay[boundary] + alpha * lut[type_values]

    return np.clip(overlay, 0, 255).astype(np.uint8)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def inspect_overview(overview_path: Path) -> None:
    print_header("PATCH OVERVIEW")

    if not overview_path.exists():
        print(f"Overview file not found: {overview_path}")
        return

    overview = pd.read_parquet(overview_path)

    print(f"Path: {overview_path}")
    print(f"Shape: {overview.shape}")
    print("Columns:")
    for col in overview.columns:
        print(f"  - {col}")

    print("\nHead:")
    print(overview.head())

    if "slide_id" in overview.columns:
        slides = overview["slide_id"].drop_duplicates().tolist()
        print(f"\nNumber of unique slides: {len(slides)}")
        print(f"First slide IDs: {slides[:20]}")

    label_cols = [c for c in LABELS_9 if c in overview.columns]
    if label_cols:
        print("\nTotal cell counts from overview columns:")
        print(overview[label_cols].sum().sort_values(ascending=False))


def inspect_metadata(cell_metadata_dir: Path, slide_id: str) -> pd.DataFrame:
    print_header(f"CELL METADATA: {slide_id}")

    meta = load_slide_metadata(cell_metadata_dir, slide_id)

    print(f"Shape: {meta.shape}")
    print("First columns:")
    for col in meta.columns[:40]:
        print(f"  - {col}")

    print("\nHead:")
    print(meta.head())

    print("\nLabel distribution:")
    print(meta["cells_final_label_group"].value_counts(dropna=False))

    return meta


def summarize_selected_patch(
    row: pd.Series,
    rgb: np.ndarray,
    cell_id_map: np.ndarray,
    type_map: np.ndarray,
    class_names: dict[int, str],
    slide_meta: pd.DataFrame,
) -> None:
    print_header("SELECTED RAW PATCH")

    for col in [
        "slide_id",
        "file_name",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "Dice",
        "Jaccard",
        "bPQ",
    ]:
        if col in row.index:
            print(f"{col}: {row[col]}")

    print(f"\nRGB shape: {rgb.shape}")
    print(f"cell_id_map shape: {cell_id_map.shape}")
    print(f"type_map shape: {type_map.shape}")

    ids = np.unique(cell_id_map)
    ids_no_bg = ids[ids != 0]

    print(f"\nNumber of unique cell IDs including background: {len(ids)}")
    print(f"Number of cell instances excluding background: {len(ids_no_bg)}")
    print(f"First cell IDs: {ids_no_bg[:20]}")

    meta_patch = slide_meta.loc[slide_meta.index.intersection(ids_no_bg)]

    print("\nPatch cell label distribution from metadata:")
    print(meta_patch["cells_final_label_group"].value_counts(dropna=False))

    print("\nPixel-level type counts:")
    vals, counts = np.unique(type_map, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"  {int(v)} = {class_names.get(int(v), 'Unknown')}: {int(c)} pixels")

    print("\nExample joined rows:")
    cols = [c for c in ["cell_id_int", "cell_id", "cells_final_label_group"] if c in meta_patch.columns]
    print(meta_patch[cols].head(10))


def plot_raw_patch(
    rgb: np.ndarray,
    cell_id_map: np.ndarray,
    type_map: np.ndarray,
    class_names: dict[int, str],
    row: pd.Series,
    alpha: float,
    save_figure: Optional[Path],
) -> None:
    overlay = make_overlay(
        rgb=rgb,
        cell_id_map=cell_id_map,
        type_map=type_map,
        class_names=class_names,
        alpha=alpha,
    )

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].imshow(rgb)
    ax[0].set_title("Raw H&E patch")
    ax[0].axis("off")

    ax[1].imshow(cell_id_map, cmap="nipy_spectral")
    ax[1].set_title("Raw cell_id_map")
    ax[1].axis("off")

    ax[2].imshow(type_map, cmap="tab10", vmin=0, vmax=max(class_names.keys()))
    ax[2].set_title("Type map from metadata join")
    ax[2].axis("off")

    ax[3].imshow(overlay)
    ax[3].set_title("H&E + boundaries by type")
    ax[3].axis("off")

    present_classes = sorted(int(v) for v in np.unique(type_map) if int(v) != 0)
    cmap = plt.get_cmap("tab10", max(class_names.keys()) + 1)
    lut = cmap(np.arange(max(class_names.keys()) + 1))[:, :3]

    handles = [
        mpatches.Patch(color=lut[cls], label=f"{cls}: {class_names.get(cls, 'Unknown')}")
        for cls in present_classes
    ]

    if handles:
        ax[3].legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(1.02, 0),
            borderaxespad=0.0,
        )

    title = (
        f"{row.get('slide_id', 'unknown slide')} | "
        f"{row.get('file_name', 'unknown file')} | "
        f"x={row.get('xmin', '?')}, y={row.get('ymin', '?')}"
    )
    fig.suptitle(title, fontsize=10)

    plt.tight_layout()

    if save_figure is not None:
        save_figure.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_figure, dpi=200, bbox_inches="tight")
        print(f"\nSaved figure to: {save_figure}")
        plt.close(fig)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sthelar-root",
        type=str,
        required=True,
        help="Path to raw STHELAR Hugging Face dataset root, e.g. /Volumes/T9/Datasets/STHELAR_20x.",
    )
    parser.add_argument(
        "--magnification",
        type=int,
        choices=[20, 40],
        required=True,
        help="STHELAR magnification, used to locate patches_overview_sthelar{20,40}x.parquet.",
    )
    parser.add_argument(
        "--slide-id",
        type=str,
        default=None,
        help="Optional slide ID, e.g. tonsil_s0.",
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default=None,
        help="Optional exact patch file name, e.g. tonsil_s0_1014.png.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Row index. If slide-id is given, index is inside that slide. Otherwise global index.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Boundary color opacity.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["9class", "5class"],
        default="9class",
        help=(
            "Label mode used for visualization. "
            "'9class' shows the original STHELAR label groups; "
            "'5class' shows the grouped labels used in the CellViT-ready preprocessing."
        ),
    )
    parser.add_argument(
        "--save-figure",
        type=str,
        default=None,
        help="Optional output path for figure.",
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Skip printing the full overview summary.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sthelar_root = Path(args.sthelar_root).expanduser().resolve()
    overview_path = sthelar_root / f"patches_overview_sthelar{args.magnification}x.parquet"
    shards_dir = sthelar_root / "data"
    cell_metadata_dir = sthelar_root / "cell_metadata"

    print_header("STHELAR RAW DATASET ROOT")
    print(f"Root: {sthelar_root}")
    print(f"Exists: {sthelar_root.exists()}")
    print(f"Overview path: {overview_path}")
    print(f"Data shards dir: {shards_dir}")
    print(f"Cell metadata dir: {cell_metadata_dir}")

    if not sthelar_root.exists():
        raise FileNotFoundError(f"STHELAR root does not exist: {sthelar_root}")

    if not shards_dir.exists():
        raise FileNotFoundError(f"Missing data shards directory: {shards_dir}")

    if not cell_metadata_dir.exists():
        raise FileNotFoundError(f"Missing cell_metadata directory: {cell_metadata_dir}")

    if not args.skip_overview:
        inspect_overview(overview_path)

    print_header("DATA SHARDS")
    shard_paths = get_clean_parquet_files(shards_dir)
    print(f"Number of parquet shards: {len(shard_paths)}")
    print("First shards:")
    for p in shard_paths[:10]:
        print(f"  - {p.name}")

    if len(shard_paths) == 0:
        raise FileNotFoundError(f"No parquet shards found in {shards_dir}")

    first_shard = pd.read_parquet(shard_paths[0])
    print(f"\nFirst shard shape: {first_shard.shape}")
    print("First shard columns:")
    for col in first_shard.columns:
        print(f"  - {col}")

    row = find_row_in_shards(
        shards=shard_paths,
        slide_id=args.slide_id,
        file_name=args.file_name,
        row_index=args.row_index,
    )

    slide_id = str(row["slide_id"])
    slide_meta = inspect_metadata(cell_metadata_dir, slide_id)

    rgb = decode_image_bytes(row["image"])
    cell_id_map = decode_cell_id_map(row["cell_id_map"])

    type_map, class_names = build_type_map_from_metadata(
        cell_id_map=cell_id_map,
        slide_meta=slide_meta,
        mode=args.mode,
    )

    summarize_selected_patch(
        row=row,
        rgb=rgb,
        cell_id_map=cell_id_map,
        type_map=type_map,
        class_names=class_names,
        slide_meta=slide_meta,
    )

    save_figure = Path(args.save_figure).expanduser().resolve() if args.save_figure else None

    plot_raw_patch(
        rgb=rgb,
        cell_id_map=cell_id_map,
        type_map=type_map,
        class_names=class_names,
        row=row,
        alpha=args.alpha,
        save_figure=save_figure,
    )


if __name__ == "__main__":
    main()