#!/usr/bin/env python3
"""
Visualize a CellViT-ready STHELAR patch.

Expected dataset structure:

dataset_root/
├── images.zip
├── labels.zip
├── patch_info_with_split.csv
├── dataset_config.yaml
└── split_manifest.yaml

Example:

python preprocessing/sthelar/visualize_patch.py \
  --dataset-root /path/to/cellvit_ready/sthelar20x_dataset \
  --split train \
  --index 0

Or select a specific packed image name:

python preprocessing/sthelar/visualize_patch.py \
  --dataset-root /path/to/cellvit_ready/sthelar20x_dataset \
  --image-name tonsil_s0__x123_y456__some_patch.png \
  --save logs/example_overlay.png
"""

from __future__ import annotations

import argparse
import io
import zipfile
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.sparse import csr_matrix


DEFAULT_CLASS_NAMES = {
    0: "Background",
    1: "Immune",
    2: "Stromal",
    3: "Epithelial",
    4: "Other",
}


def load_class_names(dataset_root: Path) -> dict[int, str]:
    """
    Load class names from dataset_config.yaml.

    Supports both formats:

    nuclei_types:
      Background: 0
      Immune: 1

    and:

    nuclei_types:
      0: Background
      1: Immune

    Falls back to DEFAULT_CLASS_NAMES if the file or key is missing.
    """
    config_path = dataset_root / "dataset_config.yaml"

    if not config_path.exists():
        print(f"Warning: dataset_config.yaml not found at {config_path}")
        print("Using default class names.")
        return DEFAULT_CLASS_NAMES

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        print(f"Warning: empty dataset_config.yaml at {config_path}")
        print("Using default class names.")
        return DEFAULT_CLASS_NAMES

    nuclei_types = config.get("nuclei_types", None)

    if nuclei_types is None:
        print("Warning: 'nuclei_types' not found in dataset_config.yaml")
        print("Using default class names.")
        return DEFAULT_CLASS_NAMES

    class_names = {}

    for key, value in nuclei_types.items():
        # Case 1: Background: 0
        if isinstance(value, int):
            class_names[int(value)] = str(key)

        # Case 2: 0: Background
        else:
            try:
                class_names[int(key)] = str(value)
            except ValueError:
                raise ValueError(
                    "Could not parse nuclei_types in dataset_config.yaml. "
                    f"Problematic entry: {key}: {value}"
                )

    if 0 not in class_names:
        class_names[0] = "Background"

    return dict(sorted(class_names.items()))

def resolve_zip_member(zf: zipfile.ZipFile, requested_name: str) -> str:
    """
    Resolve a file name inside a zip archive.

    This supports both:
    - exact names, e.g. "tonsil_s0_123.png"
    - nested names, e.g. "images/tonsil_s0_123.png"

    It also ignores macOS AppleDouble files such as "._xxx".
    """
    names = [n for n in zf.namelist() if not Path(n).name.startswith("._")]

    if requested_name in names:
        return requested_name

    requested_base = Path(requested_name).name
    matches = [n for n in names if Path(n).name == requested_base]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous file name in zip: {requested_name}. Matches: {matches[:10]}"
        )

    raise FileNotFoundError(
        f"File not found in zip: {requested_name}. "
        f"First available names: {names[:10]}"
    )


def read_image_from_zip(zip_path: Path, image_name: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path, "r") as zf:
        member = resolve_zip_member(zf, image_name)
        with zf.open(member) as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
    return np.asarray(img)


def read_npz_from_zip(zip_path: Path, label_name: str) -> dict:
    with zipfile.ZipFile(zip_path, "r") as zf:
        member = resolve_zip_member(zf, label_name)
        with zf.open(member) as f:
            data = np.load(io.BytesIO(f.read()), allow_pickle=True)
            return {k: data[k] for k in data.files}


def array_from_npz_dict(data: dict, key: str) -> np.ndarray:
    """
    Load a map from an npz dictionary.

    Supports two formats:
    1. Dense:
       data["inst_map"], data["type_map"]

    2. Sparse CSR:
       data["inst_map_data"], data["inst_map_indices"],
       data["inst_map_indptr"], data["inst_map_shape"]
    """
    if key in data:
        return np.asarray(data[key])

    sparse_keys = {
        f"{key}_data",
        f"{key}_indices",
        f"{key}_indptr",
        f"{key}_shape",
    }

    if sparse_keys.issubset(set(data.keys())):
        sparse = csr_matrix(
            (
                data[f"{key}_data"],
                data[f"{key}_indices"],
                data[f"{key}_indptr"],
            ),
            shape=tuple(data[f"{key}_shape"]),
        )
        return sparse.toarray()

    raise KeyError(
        f"Cannot find map '{key}' in label npz. "
        f"Available keys: {list(data.keys())}"
    )


def load_label_pair(labels_zip: Path, label_name: str) -> tuple[np.ndarray, np.ndarray]:
    data = read_npz_from_zip(labels_zip, label_name)

    inst_map = array_from_npz_dict(data, "inst_map").astype(np.int32)
    type_map = array_from_npz_dict(data, "type_map").astype(np.int32)

    if inst_map.shape != type_map.shape:
        raise ValueError(
            f"inst_map and type_map have different shapes: "
            f"{inst_map.shape} vs {type_map.shape}"
        )

    return inst_map, type_map


def compute_instance_boundary(inst_map: np.ndarray) -> np.ndarray:
    boundary = np.zeros(inst_map.shape, dtype=bool)

    boundary[1:, :] |= inst_map[1:, :] != inst_map[:-1, :]
    boundary[:-1, :] |= inst_map[:-1, :] != inst_map[1:, :]
    boundary[:, 1:] |= inst_map[:, 1:] != inst_map[:, :-1]
    boundary[:, :-1] |= inst_map[:, :-1] != inst_map[:, 1:]

    boundary &= inst_map != 0
    return boundary


def choose_patch(
    patch_info: pd.DataFrame,
    image_name: str | None,
    split: str | None,
    index: int,
) -> pd.Series:
    df = patch_info.copy()

    if image_name is not None:
        matches = df[df["packed_file_name"] == image_name]
        if len(matches) == 0:
            raise ValueError(f"Image name not found in patch_info: {image_name}")
        return matches.iloc[0]

    if split is not None:
        df = df[df["split"] == split].copy()
        if len(df) == 0:
            raise ValueError(f"No patches found for split={split}")

    if index < 0 or index >= len(df):
        raise IndexError(f"Index {index} out of range for selected dataframe of length {len(df)}")

    return df.iloc[index]


def summarize_patch(
    row: pd.Series,
    inst_map: np.ndarray,
    type_map: np.ndarray,
    class_names: dict[int, str],
) -> None:
    print("===== PATCH INFO =====")
    for col in [
        "slide_id",
        "file_name",
        "packed_file_name",
        "packed_label_name",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "split",
    ]:
        if col in row.index:
            print(f"{col}: {row[col]}")

    ids = np.unique(inst_map)
    ids = ids[ids != 0]

    print(f"Number of instances: {len(ids)}")

    print("\nPixel-level type counts:")
    vals, counts = np.unique(type_map, return_counts=True)
    for v, c in zip(vals, counts):
        print(f"  {int(v)} = {class_names.get(int(v), 'Unknown')}: {int(c)} pixels")

    print("\nInstance-level majority type counts:")
    instance_counts = {name: 0 for name in class_names.values() if name != "Background"}

    for cell_id in ids:
        vals_cell = type_map[inst_map == cell_id]
        vals_cell = vals_cell[vals_cell != 0]

        if len(vals_cell) == 0:
            continue

        uniq, cnt = np.unique(vals_cell, return_counts=True)
        cls = int(uniq[np.argmax(cnt)])
        name = class_names.get(cls, "Unknown")

        if name != "Background":
            instance_counts[name] = instance_counts.get(name, 0) + 1

    for name, count in instance_counts.items():
        print(f"  {name}: {count}")


def make_overlay(
    rgb: np.ndarray,
    inst_map: np.ndarray,
    type_map: np.ndarray,
    class_names: dict[int, str],
    alpha: float,
) -> np.ndarray:
    boundary = compute_instance_boundary(inst_map)

    num_classes = max(class_names.keys()) + 1
    cmap = plt.get_cmap("tab10", num_classes)
    lut = (cmap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)

    overlay = rgb.copy().astype(np.float32)

    type_values = type_map[boundary]
    colors = lut[np.clip(type_values, 0, num_classes - 1)]

    overlay[boundary] = (1.0 - alpha) * overlay[boundary] + alpha * colors
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_patch(
    rgb: np.ndarray,
    inst_map: np.ndarray,
    type_map: np.ndarray,
    row: pd.Series,
    class_names: dict[int, str],
    alpha: float,
    save_path: Path | None,
) -> None:
    overlay = make_overlay(
        rgb=rgb,
        inst_map=inst_map,
        type_map=type_map,
        class_names=class_names,
        alpha=alpha,
    )

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].imshow(rgb)
    ax[0].set_title("H&E patch")
    ax[0].axis("off")

    ax[1].imshow(inst_map, cmap="nipy_spectral")
    ax[1].set_title("Instance map / cell_id_map")
    ax[1].axis("off")

    ax[2].imshow(overlay)
    ax[2].set_title("Overlay: nucleus boundaries by type")
    ax[2].axis("off")

    present_classes = sorted(int(v) for v in np.unique(type_map) if int(v) != 0)
    num_classes = max(class_names.keys()) + 1
    cmap = plt.get_cmap("tab10", num_classes)
    lut = cmap(np.arange(num_classes))[:, :3]

    handles = [
        mpatches.Patch(color=lut[cls], label=f"{cls}: {class_names.get(cls, 'Unknown')}")
        for cls in present_classes
    ]

    if handles:
        ax[2].legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(1.02, 0),
            borderaxespad=0.0,
        )

    title = (
        f"{row.get('split', 'unknown split')} | "
        f"{row.get('slide_id', 'unknown slide')} | "
        f"{row.get('packed_file_name', row.get('file_name', 'unknown file'))}"
    )
    fig.suptitle(title, fontsize=10)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="CellViT-ready STHELAR dataset root.",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default=None,
        help="Optional packed image name from patch_info_with_split.csv.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "valid", "test"],
        help="Optional split to sample from.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index inside selected split or full patch_info.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Boundary color opacity.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output path for saving the figure.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    images_zip = dataset_root / "images.zip"
    labels_zip = dataset_root / "labels.zip"
    patch_info_csv = dataset_root / "patch_info_with_split.csv"

    if not images_zip.exists():
        raise FileNotFoundError(f"Missing images.zip: {images_zip}")

    if not labels_zip.exists():
        raise FileNotFoundError(f"Missing labels.zip: {labels_zip}")

    if not patch_info_csv.exists():
        raise FileNotFoundError(f"Missing patch_info_with_split.csv: {patch_info_csv}")

    patch_info = pd.read_csv(patch_info_csv)
    class_names = load_class_names(dataset_root)

    print("Class names:")
    for cls_id, cls_name in class_names.items():
        print(f"  {cls_id}: {cls_name}")

    row = choose_patch(
        patch_info=patch_info,
        image_name=args.image_name,
        split=args.split,
        index=args.index,
    )

    image_name = str(row["packed_file_name"])

    if "packed_label_name" in row.index:
        label_name = str(row["packed_label_name"])
    else:
        label_name = str(Path(image_name).with_suffix(".npz"))

    rgb = read_image_from_zip(images_zip, image_name)
    inst_map, type_map = load_label_pair(labels_zip, label_name)

    summarize_patch(row, inst_map, type_map, class_names)

    save_path = Path(args.save).expanduser().resolve() if args.save else None

    plot_patch(
        rgb=rgb,
        inst_map=inst_map,
        type_map=type_map,
        row=row,
        class_names=class_names,
        alpha=args.alpha,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()