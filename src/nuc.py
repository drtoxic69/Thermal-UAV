"""
Non-uniformity Correction (NUC) Module for Thermal Imagery.

This module provides algorithms to clean raw thermal data, specifically
targeting Fixed Pattern Noise (FPN) such as vertical striping, which is
common in uncooled thermal sensors.
"""

import logging

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _get_gaussian_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
    """
    Creates a 1D Gaussian kernel for smoothing.
    """
    kernel_size = int(sigma * 4) * 2 + 1
    if kernel_size < 3:
        kernel_size = 3

    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - (
        kernel_size // 2
    )

    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    return kernel.view(1, 1, -1)


def destripe_vertical(
    img_tensor: torch.Tensor, sigma: float = 5.0, use_median: bool = True
) -> torch.Tensor:
    """
    Applies a robust column-based destriping filter to remove vertical Fixed Pattern Noise.

    Algorithm:
    1. Calculate the aggregate value of every column (median is more robust to hotspots).
    2. Smooth these aggregates to find the "true" low-frequency background scene.
    3. The difference between the raw aggregates and smoothed background is the stripe pattern.
    4. Subtract this pattern from the original image.

    Args:
        img_tensor (torch.Tensor): Input image tensor. Supports (H,W), (C,H,W), or (N,C,H,W).
                                   Values should be roughly in [0, 1] range for best results.
        sigma (float): Strength of the smoothing filter. Higher = more aggressive destriping
                       but higher risk of removing real vertical objects (like poles).
                       Default of 5.0 is good for mild VEDAI striping.
        use_median (bool): If True, uses column median (slower but robust to hot targets).
                           If False, uses column mean (faster but can cause "ghosting" around cars).

    Returns:
        torch.Tensor: Cleaned image tensor with the same shape as input.
    """
    # 1. Standardize Input Shape to (N, C, H, W)
    original_shape = img_tensor.shape
    ndim = img_tensor.dim()

    if ndim == 2:  # (H, W) -> (1, 1, H, W)
        x = img_tensor.unsqueeze(0).unsqueeze(0)

    elif ndim == 3:  # (C, H, W) -> (1, C, H, W)
        x = img_tensor.unsqueeze(0)

    elif ndim == 4:  # (N, C, H, W) -> is already correct
        x = img_tensor

    else:
        raise ValueError(
            f"Unsupported tensor shape: {original_shape}. Expected 2D, 3D, or 4D."
        )

    n, c, h, w = x.shape

    # 2. Calculate Column Aggregates
    if use_median:
        col_aggregates = torch.median(x, dim=-2, keepdim=True)[0]  # Shape (N, C, 1, W)

    else:
        col_aggregates = torch.mean(x, dim=-2, keepdim=True)  # Shape (N, C, 1, W)

    # 3. Smooth Aggregates to find Background
    aggregates_1d = col_aggregates.view(n * c, 1, w)

    # Get kernel and move to same device/dtype as input
    kernel = _get_gaussian_kernel_1d(sigma, x.device).to(x.dtype)

    # Pad to handle edges correctly (replicate padding avoids dark borders)
    pad_size = kernel.shape[-1] // 2
    aggregates_padded = F.pad(aggregates_1d, (pad_size, pad_size), mode="replicate")

    # Apply smoothing
    low_freq_background = F.conv1d(aggregates_padded, kernel)

    # Reshape back to (N, C, 1, W)
    low_freq_background = low_freq_background.view(n, c, 1, w)

    # 4. Isolate Stripes
    stripes = col_aggregates - low_freq_background

    # 5. Subtract Stripes from Original Image
    cleaned_x = x - stripes

    # 6. Restore Original Shape
    if ndim == 2:
        cleaned_x = cleaned_x.squeeze(0).squeeze(0)
    elif ndim == 3:
        cleaned_x = cleaned_x.squeeze(0)

    # 7. Clamp output to valid range (assuming input was [0,1] or similar)
    return torch.clamp(cleaned_x, min=img_tensor.min(), max=img_tensor.max())
