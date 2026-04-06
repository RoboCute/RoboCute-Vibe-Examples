"""
SSIM (Structural Similarity Index) module for image comparison.

This module provides functions to compute the Structural Similarity Index (SSIM)
between two images. SSIM is a perceptual metric that quantifies image quality
degradation caused by processing such as data compression or by losses in data
transmission.

The SSIM index is calculated using a Gaussian window and local statistics
(mean, variance, and covariance) to compare luminance, contrast, and structure
between two images.

Reference:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
    Image quality assessment: from error visibility to structural similarity.
    IEEE transactions on image processing, 13(4), 600-612.

Example:
    >>> import numpy as np
    >>> from diff.ssim import compute_ssim
    >>> 
    >>> # Create two similar images
    >>> img1 = np.random.rand(256, 256)
    >>> img2 = img1 + np.random.randn(256, 256) * 0.01
    >>> 
    >>> # Compute SSIM
    >>> score = compute_ssim(img1, img2)
    >>> print(f"SSIM: {score:.4f}")
    SSIM: 0.9999

Attributes:
    __all__: List of exported functions and classes.

Author: RoboCute Team
Version: 1.0.0
"""

from typing import Optional, Union
import numpy as np
from scipy import ndimage

__all__ = [
    'SSIMInputError',
    'compute_ssim',
    'compute_ssim_map',
    'compare_images',
]


class SSIMInputError(ValueError):
    """Raised when SSIM input validation fails."""
    pass


def _validate_inputs(img1: np.ndarray, img2: np.ndarray) -> None:
    """
    Validate that two numpy arrays are compatible for SSIM computation.
    
    Validates:
    - Both inputs are numpy arrays
    - Arrays have the same dimensions (ndim)
    - Arrays have the same shape
    - Arrays have the same size
    - Arrays have the same dtype
    - Arrays are not empty
    - Arrays have valid dimensions (2D or 3D for images)
    
    Args:
        img1: First input array
        img2: Second input array
    
    Raises:
        SSIMInputError: If validation fails with detailed error message
        TypeError: If inputs are not numpy arrays
    """
    # Type validation
    if not isinstance(img1, np.ndarray):
        raise TypeError(f"img1 must be a numpy array, got {type(img1).__name__}")
    if not isinstance(img2, np.ndarray):
        raise TypeError(f"img2 must be a numpy array, got {type(img2).__name__}")
    
    # Dimension validation
    if img1.ndim != img2.ndim:
        raise SSIMInputError(
            f"Dimension mismatch: img1 has {img1.ndim}D, img2 has {img2.ndim}D. "
            f"Both arrays must have the same number of dimensions."
        )
    
    # Shape validation
    if img1.shape != img2.shape:
        raise SSIMInputError(
            f"Shape mismatch: img1 shape {img1.shape} != img2 shape {img2.shape}. "
            f"Both arrays must have identical shapes."
        )
    
    # Size validation (redundant with shape but explicit)
    if img1.size != img2.size:
        raise SSIMInputError(
            f"Size mismatch: img1 has {img1.size} elements, img2 has {img2.size} elements. "
            f"Both arrays must have the same number of elements."
        )
    
    # Empty array validation
    if img1.size == 0 or img2.size == 0:
        raise SSIMInputError(
            f"Empty input arrays: cannot compute SSIM on empty arrays "
            f"(img1 shape: {img1.shape}, img2 shape: {img2.shape})"
        )
    
    # Dtype validation - warn about mismatched types but don't error
    # since we convert to float64 anyway
    if img1.dtype != img2.dtype:
        raise SSIMInputError(
            f"Dtype mismatch: img1 dtype '{img1.dtype}' != img2 dtype '{img2.dtype}'. "
            f"Both arrays should have the same data type for consistent comparison."
        )
    
    # Valid dimensions check (2D or 3D for images)
    if img1.ndim not in (2, 3):
        raise SSIMInputError(
            f"Invalid dimensions: expected 2D or 3D arrays for image comparison, "
            f"got {img1.ndim}D array with shape {img1.shape}"
        )


def _gaussian_filter(size: int, sigma: float) -> np.ndarray:
    """
    Create a 2D Gaussian filter kernel.
    
    Args:
        size: Size of the filter (must be odd)
        sigma: Standard deviation of the Gaussian
    
    Returns:
        2D Gaussian filter kernel normalized to sum to 1
    """
    if size % 2 == 0:
        size += 1  # Ensure odd size
    
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    x, y = np.meshgrid(x, y)
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
    dynamic_range: Optional[float] = None
) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two images.
    
    SSIM measures the perceptual difference between two similar images.
    It compares local patterns of pixel intensities that have been 
    normalized for luminance and contrast.
    
    Args:
        img1: First image as numpy array (H, W) or (H, W, C)
        img2: Second image as numpy array (H, W) or (H, W, C)
        window_size: Size of the sliding window for local comparison (default: 11)
        k1: First stability constant (default: 0.01)
        k2: Second stability constant (default: 0.03)
        dynamic_range: Dynamic range of pixel values. If None, auto-detected
                      from data type (255 for uint8, 1.0 for float, etc.)
    
    Returns:
        SSIM value between -1 and 1, where 1 indicates perfect similarity
    
    Raises:
        SSIMInputError: If input arrays have incompatible dimensions, size, or dtype
        TypeError: If inputs are not numpy arrays
    
    Example:
        >>> import numpy as np
        >>> img1 = np.random.rand(256, 256)
        >>> img2 = img1 + np.random.randn(256, 256) * 0.01
        >>> score = compute_ssim(img1, img2)
        >>> print(f"SSIM: {score:.4f}")
    """
    # Validate inputs (dimensions, shape, size, dtype)
    _validate_inputs(img1, img2)
    
    # Handle multi-channel images (3D arrays)
    if img1.ndim == 3:
        # Compute SSIM for each channel and average
        ssim_channels = []
        for c in range(img1.shape[2]):
            ssim_c = compute_ssim(
                img1[:, :, c],
                img2[:, :, c],
                window_size,
                k1,
                k2,
                dynamic_range
            )
            ssim_channels.append(ssim_c)
        return np.mean(ssim_channels)
    
    # Auto-detect dynamic range if not specified
    if dynamic_range is None:
        if img1.dtype == np.uint8:
            dynamic_range = 255.0
        elif img1.dtype == np.uint16:
            dynamic_range = 65535.0
        else:
            # For float types, use the actual data range
            dynamic_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
            if dynamic_range == 0:
                dynamic_range = 1.0
    
    # Convert to float64 for precision
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Constants for numerical stability
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    
    # Create Gaussian window
    window = _gaussian_filter(window_size, sigma=1.5)
    
    # Compute means using convolution
    mu1 = ndimage.convolve(img1, window, mode='reflect')
    mu2 = ndimage.convolve(img2, window, mode='reflect')
    
    # Compute squares and product
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = ndimage.convolve(img1 ** 2, window, mode='reflect') - mu1_sq
    sigma2_sq = ndimage.convolve(img2 ** 2, window, mode='reflect') - mu2_sq
    sigma12 = ndimage.convolve(img1 * img2, window, mode='reflect') - mu1_mu2
    
    # SSIM formula
    # SSIM(x, y) = (2*mu_x*mu_y + c1) * (2*sigma_xy + c2) / 
    #              ((mu_x^2 + mu_y^2 + c1) * (sigma_x^2 + sigma_y^2 + c2))
    numerator1 = 2 * mu1_mu2 + c1
    numerator2 = 2 * sigma12 + c2
    denominator1 = mu1_sq + mu2_sq + c1
    denominator2 = sigma1_sq + sigma2_sq + c2
    
    ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
    
    # Return mean SSIM value
    return float(ssim_map.mean())


def compute_ssim_map(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
    dynamic_range: Optional[float] = None
) -> np.ndarray:
    """
    Compute the SSIM quality map between two images.
    
    Unlike compute_ssim() which returns a single score, this returns
    the full SSIM map showing local similarity values.
    
    Args:
        img1: First image as numpy array (H, W) or (H, W, C)
        img2: Second image as numpy array (H, W) or (H, W, C)
        window_size: Size of the sliding window (default: 11)
        k1: First stability constant (default: 0.01)
        k2: Second stability constant (default: 0.03)
        dynamic_range: Dynamic range of pixel values
    
    Returns:
        SSIM map as numpy array with same spatial dimensions as input
    
    Raises:
        SSIMInputError: If input arrays have incompatible dimensions, size, or dtype
        TypeError: If inputs are not numpy arrays
    """
    # Validate inputs (dimensions, shape, size, dtype)
    _validate_inputs(img1, img2)
    
    # For multi-channel, compute luminance SSIM
    if img1.ndim == 3:
        # Convert to luminance (simple average of channels)
        img1_gray = img1.mean(axis=2)
        img2_gray = img2.mean(axis=2)
        return compute_ssim_map(
            img1_gray, img2_gray,
            window_size, k1, k2, dynamic_range
        )
    
    # Auto-detect dynamic range
    if dynamic_range is None:
        if img1.dtype == np.uint8:
            dynamic_range = 255.0
        elif img1.dtype == np.uint16:
            dynamic_range = 65535.0
        else:
            dynamic_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
            if dynamic_range == 0:
                dynamic_range = 1.0
    
    # Convert to float64
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Constants
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    
    # Gaussian window
    window = _gaussian_filter(window_size, sigma=1.5)
    
    # Convolutions
    mu1 = ndimage.convolve(img1, window, mode='reflect')
    mu2 = ndimage.convolve(img2, window, mode='reflect')
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = ndimage.convolve(img1 ** 2, window, mode='reflect') - mu1_sq
    sigma2_sq = ndimage.convolve(img2 ** 2, window, mode='reflect') - mu2_sq
    sigma12 = ndimage.convolve(img1 * img2, window, mode='reflect') - mu1_mu2
    
    # SSIM map
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return ssim_map


def compare_images(path1: str, path2: str) -> float:
    """
    Compare two images from file paths and return SSIM score.
    
    Args:
        path1: Path to first image file
        path2: Path to second image file
    
    Returns:
        SSIM score between -1 and 1
    
    Raises:
        ImportError: If PIL is not available
        FileNotFoundError: If image files don't exist
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL (Pillow) is required for image file loading")
    
    img1 = np.array(Image.open(path1).convert('RGB'))
    img2 = np.array(Image.open(path2).convert('RGB'))
    
    return compute_ssim(img1, img2)


if __name__ == '__main__':
    # Simple test
    print("SSIM module for image comparison")
    
    # Test with identical images
    test_img = np.random.rand(64, 64)
    score = compute_ssim(test_img, test_img)
    print(f"SSIM of identical images: {score:.4f} (expected: 1.0)")
    
    # Test with different images
    test_img2 = np.random.rand(64, 64)
    score2 = compute_ssim(test_img, test_img2)
    print(f"SSIM of random images: {score2:.4f} (expected: ~0.0)")
