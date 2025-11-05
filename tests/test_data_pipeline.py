"""
Robust tests for the src.data_pipeline module.

This test suite verifies:
1.  Core conversion logic (_box_pascal_to_yolo).
2.  Dataset integrity (correct number of samples found).
3.  __getitem__ correctness (types, shapes, and value ranges).
4.  DataLoader batching (correct shapes and value ranges).
5.  Error handling for bad inputs.
"""

import os

import pytest
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.data_pipeline import (
    BaseRGBTDataset,
    VedaiDataset,
    create_dataloader,
)

TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH")
IS_DATA_PRESENT = TEST_DATA_PATH and os.path.exists(TEST_DATA_PATH)

skip_if_no_data = pytest.mark.skipif(
    not IS_DATA_PRESENT,
    reason=f"Test data not found at {TEST_DATA_PATH}. Skipping integration tests.",
)


@pytest.fixture(scope="session")
def basic_transform() -> T.Compose:
    """A simple ToTensor transform for testing."""
    return T.Compose([T.ToTensor()])


@pytest.fixture(scope="session")
def vedai_dataset(basic_transform: T.Compose) -> VedaiDataset:
    """
    Loads the VEDAI dataset once for all tests that need it.
    """
    if not IS_DATA_PRESENT:
        pytest.skip(f"Test data not found at {TEST_DATA_PATH}.")

    assert TEST_DATA_PATH is not None, "TEST_DATA_PATH should not be None here."

    try:
        dataset = VedaiDataset(root_dir=TEST_DATA_PATH, transform=basic_transform)
        return dataset

    except FileNotFoundError as e:
        pytest.fail(f"Failed to load VEDAI dataset from {TEST_DATA_PATH}: {e}")


@pytest.mark.parametrize(
    "box_pascal, img_size, expected_yolo",
    [
        ((100, 100, 200, 200), (500, 500), (0.3, 0.3, 0.2, 0.2)),
        ((0, 0, 500, 500), (500, 500), (0.5, 0.5, 1.0, 1.0)),
        ((0, 0, 50, 100), (500, 500), (0.05, 0.1, 0.1, 0.2)),
        ((50, 100, 150, 300), (200, 400), (0.5, 0.5, 0.5, 0.5)),
        ((10, 10, 20, 20), (0, 0), (0.0, 0.0, 0.0, 0.0)),
    ],
)
def test_box_pascal_to_yolo(box_pascal, img_size, expected_yolo):
    """
    Unit tests the bounding box conversion logic.
    """
    base_data = BaseRGBTDataset(root_dir=".")
    assert base_data._box_pascal_to_yolo(box_pascal, img_size) == pytest.approx(
        expected_yolo
    )


@skip_if_no_data
def test_dataset_integrity(vedai_dataset: VedaiDataset):
    """
    Tests that the dataset was loaded correctly.
    """
    expected_samples = 1246
    assert len(vedai_dataset) == expected_samples, (
        f"Expected to find {expected_samples} valid samples, but found {len(vedai_dataset)}."
    )
    assert len(vedai_dataset.samples) == expected_samples, (
        "The internal 'samples' list does not match the dataset length."
    )


@skip_if_no_data
def test_dataset_getitem(vedai_dataset: VedaiDataset):
    """
    Tests that __getitem__ returns data in the correct format and with
    valid values.
    """
    rgb_img, thermal_img, targets = vedai_dataset[0]

    assert isinstance(rgb_img, torch.Tensor)
    assert isinstance(thermal_img, torch.Tensor)
    assert isinstance(targets, torch.Tensor)

    assert rgb_img.shape == (3, 512, 512)
    assert thermal_img.shape == (3, 512, 512)

    assert targets.dim() == 2
    assert targets.shape[1] == 5

    assert rgb_img.min() >= 0.0 and rgb_img.max() <= 1.0
    assert thermal_img.min() >= 0.0 and thermal_img.max() <= 1.0

    if targets.shape[0] > 0:
        assert torch.all(targets[:, 0] >= 0)
        assert torch.all(targets[:, 1:] >= 0.0)
        assert torch.all(targets[:, 1:] <= 1.0)


@skip_if_no_data
def test_dataloader_batching(basic_transform: T.Compose):
    """
    Tests the create_dataloader factory and the collate_fn.
    """
    BATCH_SIZE = 8

    assert TEST_DATA_PATH is not None, "TEST_DATA_PATH should not be None here."

    dataloader = create_dataloader(
        dataset_names=["vedai"],
        root_dirs=[TEST_DATA_PATH],
        batch_size=BATCH_SIZE,
        transform=basic_transform,
        shuffle=False,
        num_workers=0,
    )

    assert isinstance(dataloader, DataLoader)
    rgb_batch, thermal_batch, targets_batch = next(iter(dataloader))

    assert rgb_batch.shape[0] <= BATCH_SIZE
    assert rgb_batch.shape[1:] == (3, 512, 512)

    assert thermal_batch.shape[0] <= BATCH_SIZE
    assert thermal_batch.shape[1:] == (3, 512, 512)

    assert targets_batch.dim() == 2
    assert targets_batch.shape[1] == 6

    if targets_batch.shape[0] > 0:
        batch_indices = targets_batch[:, 0]
        class_ids = targets_batch[:, 1]
        boxes = targets_batch[:, 2:]

        assert torch.all(batch_indices >= 0)
        assert torch.all(batch_indices < BATCH_SIZE)
        assert torch.all(class_ids >= 0)
        assert torch.all(boxes >= 0.0)
        assert torch.all(boxes <= 1.0)


def test_dataloader_error_handling():
    """
    Tests that the dataloader factory fails gracefully.
    """
    with pytest.raises(RuntimeError, match="No valid datasets were loaded"):
        create_dataloader(
            dataset_names=["fake_dataset"], root_dirs=["./nonexistent"], batch_size=4
        )

    with pytest.raises(ValueError, match="dataset_names and root_dirs must have"):
        create_dataloader(
            dataset_names=["vedai"],
            root_dirs=["./data", "./another_data"],
            batch_size=4,
        )
