#!/usr/bin/env python3
"""
VEDAI to YOLO Standard Format Converter.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import torch
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from src.data_pipeline import MASTER_CLASS_MAP, VedaiDataset

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VEDAI to YOLO format.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/vedai_thermal_yolo",
        help="Directory to save the converted dataset.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["thermal", "rgb"],
        default="thermal",
        help="Which image modality to use for the 'images' folder. (Default: thermal)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (e.g., 0.2 = 20%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val splits.",
    )
    return parser.parse_args()


def create_directory_structure(base_path: Path) -> Tuple[Path, Path, Path, Path]:
    """Creates the standard YOLO train/val directory structure."""
    if base_path.exists():
        log.warning(f"Output directory {base_path} already exists. Deleting...")
        shutil.rmtree(base_path)

    images_train = base_path / "images" / "train"
    images_val = base_path / "images" / "val"
    labels_train = base_path / "labels" / "train"
    labels_val = base_path / "labels" / "val"

    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    return images_train, images_val, labels_train, labels_val


def create_dataset_yaml(output_dir: Path, class_map: dict) -> None:
    """Generates the dataset.yaml file required by Ultralytics."""
    id_to_name = {v: k for k, v in class_map.items()}
    max_id = max(id_to_name.keys())
    names_list = [id_to_name.get(i, "unknown") for i in range(max_id + 1)]

    yaml_data = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(names_list)),
    }

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    log.info(f"Created dataset config: {yaml_path}")


def process_split(
    dataset: VedaiDataset,
    indices: List[int],
    split_name: str,
    img_dest_dir: Path,
    label_dest_dir: Path,
    use_thermal: bool = True,
) -> None:
    """Copies images and writes YOLO label files for a given split."""
    log.info(f"Processing {split_name} split ({len(indices)} images)...")

    for idx in tqdm(indices, desc=f"Converting {split_name}"):
        rgb_path, thermal_path, _ = dataset.samples[idx]

        try:
            _, _, targets_tensor = dataset[idx]
        except Exception as e:
            log.warning(f"Error loading sample {idx}, skipping. Details: {e}")
            continue

        src_img_path = thermal_path if use_thermal else rgb_path
        base_name = Path(src_img_path).stem

        dst_img_path = img_dest_dir / Path(src_img_path).name
        shutil.copy2(src_img_path, dst_img_path)

        label_path = label_dest_dir / f"{base_name}.txt"
        with open(label_path, "w") as f:
            for target in targets_tensor:
                cls_id = int(target[0].item())
                x, y, w, h = target[1:].tolist()
                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def main():
    args = parse_args()

    data_root = os.environ.get("TEST_DATA_PATH")
    if not data_root:
        log.error("TEST_DATA_PATH not set in .env file.")
        return

    try:
        dataset = VedaiDataset(root_dir=data_root, transform=None)
        log.info(f"Loaded VEDAI dataset with {len(dataset)} total valid samples.")
    except FileNotFoundError as e:
        log.error(e)
        return

    output_dir = Path(args.output_dir)
    img_train, img_val, lbl_train, lbl_val = create_directory_structure(output_dir)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    val_size = int(len(dataset) * args.val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    log.info(f"Split: {len(train_indices)} train, {len(val_indices)} val")

    use_thermal = args.modality == "thermal"

    process_split(dataset, train_indices, "train", img_train, lbl_train, use_thermal)
    process_split(dataset, val_indices, "val", img_val, lbl_val, use_thermal)

    create_dataset_yaml(output_dir, MASTER_CLASS_MAP)

    log.info("Conversion complete!")
    log.info(f"Training data ready at: {output_dir.absolute()}")
    log.info("To train with YOLOv8, run:")
    log.info(
        f"yolo detect train data={output_dir}/dataset.yaml model=yolov8n.pt epochs=100 imgsz=512"
    )


if __name__ == "__main__":
    main()
