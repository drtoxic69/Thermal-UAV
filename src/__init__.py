"""
Main __init__.py for the src package.

This file makes 'src' a Python package and exports key components
from its modules for easier access.
"""

import logging

from .data_pipeline import (
    MASTER_CLASS_MAP,
    BaseRGBTDataset,
    VedaiDataset,
    create_dataloader,
    rgbt_collate_fn,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "BaseRGBTDataset",
    "VedaiDataset",
    "create_dataloader",
    "rgbt_collate_fn",
    "MASTER_CLASS_MAP",
]
