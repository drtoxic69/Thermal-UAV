"""
Thermal Image Augmentation Module.

This module provides specialized augmentations for simulating realistic
thermal sensor artifacts, specifically Fixed Pattern Noise (FPN) (striping)
and general sensor grain.

These augmentations help train robust models that don't overfit to
perfectly clean, pre-processed thermal datasets (like VEDAI).
"""

import torch
import torch.nn as nn


class RandomThermalNoise(nn.Module):
    """
    Applies synthetic thermal sensor noise to an image tensor.

    This is a composite augmentation that can add:
    1. Fixed Pattern Noise (FPN): Vertical and/or horizontal stripes.
    2. Grain Noise: Gaussian noise simulating sensor sensitivity limits.

    It is designed to be used within a torchvision.transforms.Compose block.
    Expects input tensors in shape (C, H, W) or (N, C, H, W) with values in [0, 1].
    """

    def __init__(
        self,
        p: float = 0.5,
        fpn_prob: float = 0.8,
        grain_prob: float = 1.0,
        vertical_stripe_intensity: float = 0.05,
        horizontal_stripe_intensity: float = 0.05,
        grain_intensity: float = 0.05,
    ) -> None:
        """
        Args:
            p (float): Probability of applying ANY thermal noise.
            fpn_prob (float): Probability of applying FPN if noise is applied.
            grain_prob (float): Probability of applying grain if noise is applied.
            vertical_stripe_intensity (float): Max std dev for vertical stripe offsets.
            horizontal_stripe_intensity (float): Max std dev for horizontal stripe offsets.
            grain_intensity (float): Max std dev for Gaussian grain noise.
        """
        super().__init__()
        self.p = p
        self.fpn_prob = fpn_prob
        self.grain_prob = grain_prob
        self.v_stripe_std = vertical_stripe_intensity
        self.h_stripe_std = horizontal_stripe_intensity
        self.grain_std = grain_intensity

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Applies augmentations to the input tensor.
        """
        if torch.rand(1).item() >= self.p:
            return img

        # Ensure we don't track gradients for augmentations
        with torch.no_grad():
            # 1. Standardize to 4D (N, C, H, W) for uniform processing
            is_3d = img.dim() == 3
            if is_3d:
                x = img.unsqueeze(0)
            else:
                x = img

            output = x.clone()

            # 2. Apply Fixed Pattern Noise (Stripes)
            if torch.rand(1).item() < self.fpn_prob:
                output = self._add_fpn(output)

            # 3. Apply Grain Noise
            if torch.rand(1).item() < self.grain_prob:
                output = self._add_grain(output)

            # 4. Clamp to ensure valid image range [0, 1]
            output = torch.clamp(output, 0.0, 1.0)

            # 5. Restore original 3D shape if necessary
            if is_3d:
                output = output.squeeze(0)

            return output

    def _add_fpn(self, img: torch.Tensor) -> torch.Tensor:
        """Adds vertical and/or horizontal stripes to a 4D tensor."""
        # We know img is (N, C, H, W) because of forward()'s standardization
        _, _, h, w = img.shape
        noise = torch.zeros_like(img)

        # Vertical stripes (offsets applied to entire columns)
        if self.v_stripe_std > 0:
            sigma = torch.rand(1).item() * self.v_stripe_std
            v_stripes = torch.randn(w, device=img.device, dtype=img.dtype) * sigma
            # Broadcast shape [W] -> [1, 1, 1, W] matches [N, C, H, W]
            noise += v_stripes.view(1, 1, 1, w)

        # Horizontal stripes (offsets applied to entire rows)
        if self.h_stripe_std > 0:
            sigma = torch.rand(1).item() * self.h_stripe_std
            h_stripes = torch.randn(h, device=img.device, dtype=img.dtype) * sigma
            # Broadcast shape [H] -> [1, 1, H, 1] matches [N, C, H, W]
            noise += h_stripes.view(1, 1, h, 1)

        return img + noise

    def _add_grain(self, img: torch.Tensor) -> torch.Tensor:
        """Adds random Gaussian noise."""
        sigma = torch.rand(1).item() * self.grain_std
        grain = torch.randn_like(img) * sigma
        return img + grain

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(p={self.p}, "
            f"fpn_prob={self.fpn_prob}, grain_prob={self.grain_prob}, "
            f"v_std={self.v_stripe_std}, grain_std={self.grain_std})"
        )
