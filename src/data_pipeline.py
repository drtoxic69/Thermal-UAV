"""
Robust Data Pipeline for RGB-T Object Detection.

This module contains the PyTorch Dataset classes and DataLoader logic
for handling multiple, paired RGB and Thermal (Infrared) datasets.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

log = logging.getLogger(__name__)


MASTER_CLASS_MAP: Dict[str, int] = {
    "car": 0,
    "truck": 0,
    "pickup": 0,
    "tractor": 0,
    "camping_car": 0,
    "vehicle": 0,
    "plane": 1,
    "boat": 2,
    "other": 3,
    "person": 4,
    "bicycle": 5,
}


class BaseRGBTDataset(Dataset):
    """
    Abstract base class for all RGB-T datasets.
    Provides core logic for box conversion and image loading.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
    ) -> None:
        if not os.path.isdir(root_dir):
            log.error(f"Dataset root directory not found: {root_dir}")
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, str, str]] = []

    def _box_pascal_to_yolo(
        self,
        box: Tuple[float, float, float, float],
        img_size: Tuple[int, int],
    ) -> Tuple[float, float, float, float]:
        """
        Converts PASCAL_VOC box [xmin, ymin, xmax, ymax] to YOLO [x_c, y_c, w, h].
        """
        xmin, ymin, xmax, ymax = box
        img_w, img_h = img_size

        if img_w == 0 or img_h == 0:
            log.warning(f"Image size is zero ({img_w}x{img_h}). Cannot normalize box.")
            return 0.0, 0.0, 0.0, 0.0

        dw = 1.0 / img_w
        dh = 1.0 / img_h

        x_center = (xmin + xmax) / 2.0 * dw
        y_center = (ymin + ymax) / 2.0 * dh
        width = (xmax - xmin) * dw
        height = (ymax - ymin) * dh

        # Clamp values to [0.0, 1.0] to fix potential out-of-bounds annotations
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        return (x_center, y_center, width, height)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, torch.Tensor]:
        raise NotImplementedError

    def _load_images(
        self, rgb_path: str, thermal_path: str
    ) -> Optional[Tuple[Image.Image, Image.Image]]:
        """Loads RGB and Thermal images, converting to "RGB" mode."""
        try:
            rgb_img = Image.open(rgb_path).convert("RGB")
            thermal_img = Image.open(thermal_path).convert("RGB")
            return rgb_img, thermal_img
        except FileNotFoundError as e:
            log.warning(f"Could not load image file: {e}")
            return None
        except Exception as e:
            log.warning(f"Error loading images {rgb_path}, {thermal_path}: {e}")
            return None


class VedaiDataset(BaseRGBTDataset):
    """
    VEDAI 512x512 Dataset Implementation.

    Parses the custom 14-value .txt annotation format found in the dataset.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initializes the VEDAI dataset, finding all valid paired samples.
        """
        super().__init__(root_dir, transform)

        self.annotation_dir = os.path.join(self.root_dir, "Annotations512")
        self.image_dir = os.path.join(self.root_dir, "Vehicules512")

        self.samples = self._find_valid_samples()

        if not self.samples:
            log.error(
                f"No valid samples found in {self.root_dir}. "
                "Check your file paths. Expected structure:\n"
                f"{self.annotation_dir}/*.txt\n"
                f"{self.image_dir}/*_co.png\n"
                f"{self.image_dir}/*_ir.png"
            )
            raise FileNotFoundError(f"No valid VEDAI samples found in {self.root_dir}")

    def _find_valid_samples(self) -> List[Tuple[str, str, str]]:
        """
        Scans annotation dir and keeps only samples where
        RGB, Thermal, and Annotation files all exist.
        """
        valid_samples = []
        if not os.path.isdir(self.annotation_dir):
            log.error(f"Annotation directory not found: {self.annotation_dir}")
            return []
        if not os.path.isdir(self.image_dir):
            log.error(f"Image directory not found: {self.image_dir}")
            return []

        for annot_file in os.listdir(self.annotation_dir):
            if not annot_file.endswith(".txt") or not annot_file.startswith("00"):
                continue

            base_name = annot_file.replace(".txt", "")
            txt_path = os.path.join(self.annotation_dir, annot_file)
            rgb_path = os.path.join(self.image_dir, f"{base_name}_co.png")
            thermal_path = os.path.join(self.image_dir, f"{base_name}_ir.png")

            if (
                os.path.exists(txt_path)
                and os.path.exists(rgb_path)
                and os.path.exists(thermal_path)
            ):
                valid_samples.append((rgb_path, thermal_path, txt_path))
            else:
                log.debug(f"Skipping sample {base_name}: missing one or more files.")

        log.info(f"Found {len(valid_samples)} valid VEDAI samples.")
        return valid_samples

    def _load_targets(
        self, annot_path: str, img_size: Tuple[int, int]
    ) -> List[List[float]]:
        """
        Parses the custom VEDAI .txt annotation format.

        Each line is expected to be 14 floats:
        [val, val, val, class_id, val, val, x1, y1, x2, y2, x3, y3, x4, y4]
        """
        targets = []
        try:
            with open(annot_path, "r") as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) == 14:
                        try:
                            # Extract class ID (4th element, index 3)
                            class_id = float(parts[3])

                            # Extract 8 corner coordinates (last 8 elements)
                            # This logic (parts[6:]) is correct for a 14-part line
                            corners = [float(p) for p in parts[6:]]

                            # Get x and y coordinates
                            x_coords = corners[0::2]  # [x1, x2, x3, x4]
                            y_coords = corners[1::2]  # [y1, y2, y3, y4]

                            # Create an axis-aligned bounding box
                            xmin = min(x_coords)
                            ymin = min(y_coords)
                            xmax = max(x_coords)
                            ymax = max(y_coords)

                            box_pascal = (xmin, ymin, xmax, ymax)

                            # Convert to normalized YOLO format
                            box_yolo = self._box_pascal_to_yolo(box_pascal, img_size)

                            targets.append([class_id, *box_yolo])

                        except ValueError:
                            log.warning(f"Could not parse line in {annot_path}: {line}")
                        except IndexError:
                            log.warning(
                                f"Index error parsing line in {annot_path}: {line}"
                            )
                    else:
                        log.warning(
                            f"Malformed line in {annot_path} (expected 14 parts, got {len(parts)}): {line}"
                        )

        except Exception as e:
            log.warning(f"Failed to read annotation file {annot_path}: {e}")

        return targets

    def __getitem__(self, idx: int) -> Tuple[Any, Any, torch.Tensor]:
        """
        Fetches the sample at the given index.
        """
        rgb_path, thermal_path, annot_path = self.samples[idx]

        image_pair = self._load_images(rgb_path, thermal_path)
        if image_pair is None:
            log.warning(f"Failed to load images for index {idx}, trying next sample.")
            return self.__getitem__((idx + 1) % len(self))

        rgb_img, thermal_img = image_pair

        img_size = rgb_img.size  # (width, height)

        targets_list = self._load_targets(annot_path, img_size)

        if targets_list:
            targets = torch.tensor(targets_list, dtype=torch.float32)
        else:
            targets = torch.empty((0, 5), dtype=torch.float32)

        if self.transform:
            rgb_img = self.transform(rgb_img)
            thermal_img = self.transform(thermal_img)

        return rgb_img, thermal_img, targets


def rgbt_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching RGB-T data.
    Creates a (N, 6) tensor for targets: [batch_index, class_id, x_c, y_c, w, h]
    """
    rgb_images, thermal_images, all_targets = zip(*batch)

    rgb_batch = torch.stack(rgb_images, 0)
    thermal_batch = torch.stack(thermal_images, 0)

    targets_with_batch_idx = []
    for i, targets in enumerate(all_targets):
        if targets.shape[0] > 0:
            batch_idx_col = torch.full((targets.shape[0], 1), i, dtype=torch.float32)
            targets_with_idx = torch.cat([batch_idx_col, targets], dim=1)
            targets_with_batch_idx.append(targets_with_idx)

    if targets_with_batch_idx:
        targets_batch = torch.cat(targets_with_batch_idx, 0)
    else:
        targets_batch = torch.empty((0, 6), dtype=torch.float32)

    return rgb_batch, thermal_batch, targets_batch


def create_dataloader(
    dataset_names: List[str],
    root_dirs: List[str],
    batch_size: int,
    transform: Optional[Callable] = None,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Factory function to create a DataLoader for one or more datasets.
    """
    dataset_map = {
        "vedai": VedaiDataset,
        # "flir": FlirDataset,
        # "m3fd": M3fdDataset,
    }

    if len(dataset_names) != len(root_dirs):
        raise ValueError("dataset_names and root_dirs must have the same length.")

    datasets = []
    for name, root_dir in zip(dataset_names, root_dirs):
        name_lower = name.lower()
        if name_lower not in dataset_map:
            log.warning(
                f"Unknown dataset '{name}'. "
                f"Available: {list(dataset_map.keys())}. Skipping."
            )
            continue

        DatasetClass = dataset_map[name_lower]
        try:
            dataset = DatasetClass(root_dir=root_dir, transform=transform)
            datasets.append(dataset)
        except FileNotFoundError as e:
            log.error(f"Failed to load dataset '{name}' from {root_dir}: {e}")

    if not datasets:
        raise RuntimeError("No valid datasets were loaded. Aborting.")

    combined_dataset = ConcatDataset(datasets)
    log.info(f"Created combined dataset with {len(combined_dataset)} total samples.")

    safe_num_workers = min(os.cpu_count() or 1, num_workers)

    pin_memory_enabled = True
    try:
        if torch.backends.mps.is_available():
            pin_memory_enabled = False
    except AttributeError:
        pass

    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        collate_fn=rgbt_collate_fn,
        num_workers=safe_num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory_enabled,
        drop_last=False,
    )
