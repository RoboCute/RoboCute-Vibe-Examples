"""
Utility functions for noise texture generation and visualization.
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Union
import os


def save_texture(texture: np.ndarray, filepath: str, 
                 colormap: Optional[str] = None) -> None:
    """
    Save noise texture to file.
    
    Args:
        texture: 2D numpy array with noise values
        filepath: Output file path
        colormap: Optional colormap name ('grayscale', 'heatmap', 'seismic')
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', 
                exist_ok=True)
    
    if colormap and texture.ndim == 2:
        texture = apply_colormap(texture, colormap)
        img = Image.fromarray(texture, mode='RGB')
    else:
        if texture.ndim == 2:
            img = Image.fromarray(texture, mode='L')
        else:
            img = Image.fromarray(texture)
    
    img.save(filepath)


def display_texture(texture: np.ndarray, title: str = "Noise Texture",
                    colormap: Optional[str] = None) -> None:
    """
    Display noise texture using matplotlib.
    
    Args:
        texture: 2D numpy array with noise values
        title: Window title
        colormap: Optional colormap name
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 8))
        if colormap:
            plt.imshow(texture, cmap=colormap)
        else:
            plt.imshow(texture, cmap='gray')
        plt.title(title)
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not installed. Install: pip install matplotlib")
        print(f"Texture shape: {texture.shape}, range: [{texture.min():.3f}, {texture.max():.3f}]")


def apply_colormap(texture: np.ndarray, colormap: str) -> np.ndarray:
    """
    Apply colormap to grayscale texture.
    
    Args:
        texture: 2D numpy array (0-255 or 0-1)
        colormap: Colormap name
    
    Returns:
        3D numpy array (RGB)
    """
    # Normalize to 0-1 if needed
    if texture.max() > 1.0:
        texture = texture / 255.0
    
    if colormap == 'heatmap':
        # Simple heatmap: black -> red -> yellow -> white
        result = np.zeros((*texture.shape, 3))
        result[..., 0] = np.clip(texture * 3, 0, 1)  # Red
        result[..., 1] = np.clip((texture - 0.33) * 3, 0, 1)  # Green
        result[..., 2] = np.clip((texture - 0.66) * 3, 0, 1)  # Blue
        return (result * 255).astype(np.uint8)
    
    elif colormap == 'seismic':
        # Seismic colormap (good for signed data)
        result = np.zeros((*texture.shape, 3))
        # Normalize to -1 to 1
        t = texture * 2 - 1
        result[..., 0] = np.clip(1 - t, 0, 1)  # Red for negative
        result[..., 2] = np.clip(1 + t, 0, 1)  # Blue for positive
        result[..., 1] = 1 - np.abs(t)  # Green in middle
        return (result * 255).astype(np.uint8)
    
    elif colormap == 'terrain':
        # Terrain-like colors
        result = np.zeros((*texture.shape, 3))
        t = texture
        # Water (deep blue to light blue)
        mask = t < 0.3
        result[mask, 2] = 0.5 + t[mask] * 1.5
        # Land (green to brown to white)
        mask = (t >= 0.3) & (t < 0.5)
        result[mask, 1] = 0.5 + (t[mask] - 0.3) * 2
        mask = (t >= 0.5) & (t < 0.7)
        result[mask, 0] = (t[mask] - 0.5) * 3
        result[mask, 1] = 0.9 - (t[mask] - 0.5) * 1.5
        mask = t >= 0.7
        result[mask, 0] = 0.7 + (t[mask] - 0.7) * 3
        result[mask, 1] = 0.7 + (t[mask] - 0.7) * 3
        result[mask, 2] = 0.7 + (t[mask] - 0.7) * 3
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    else:  # grayscale
        return (texture * 255).astype(np.uint8)


def create_texture_grid(textures: List[np.ndarray], 
                        titles: Optional[List[str]] = None,
                        cols: int = 3,
                        figsize: Tuple[int, int] = (15, 10)) -> np.ndarray:
    """
    Create a grid image from multiple textures.
    
    Args:
        textures: List of 2D numpy arrays
        titles: Optional list of titles for each texture
        cols: Number of columns in grid
        figsize: Figure size for matplotlib display
    
    Returns:
        Combined grid image as numpy array
    """
    try:
        import matplotlib.pyplot as plt
        
        rows = (len(textures) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (texture, ax) in enumerate(zip(textures, axes.flat)):
            ax.imshow(texture, cmap='gray')
            if titles and idx < len(titles):
                ax.set_title(titles[idx])
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(len(textures), len(axes.flat)):
            axes.flat[idx].axis('off')
        
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        grid_image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        return grid_image
    
    except ImportError:
        print("matplotlib not installed. Install: pip install matplotlib")
        return np.array([])


def blend_textures(tex1: np.ndarray, tex2: np.ndarray, 
                   alpha: float = 0.5) -> np.ndarray:
    """
    Blend two textures together.
    
    Args:
        tex1: First texture (0-255)
        tex2: Second texture (0-255)
        alpha: Blend factor (0 = all tex1, 1 = all tex2)
    
    Returns:
        Blended texture
    """
    return (tex1 * (1 - alpha) + tex2 * alpha).astype(np.uint8)


def create_normal_map(height_map: np.ndarray, 
                      strength: float = 1.0) -> np.ndarray:
    """
    Convert height map to normal map.
    
    Args:
        height_map: 2D height values (0-255 or 0-1)
        strength: Normal intensity
    
    Returns:
        RGB normal map
    """
    # Normalize to 0-1
    if height_map.max() > 1:
        height_map = height_map / 255.0
    
    # Calculate gradients
    gy, gx = np.gradient(height_map)
    
    # Scale by strength
    gx *= strength
    gy *= strength
    
    # Create normal vectors
    normal = np.zeros((*height_map.shape, 3))
    normal[..., 0] = -gx  # X (red)
    normal[..., 1] = -gy  # Y (green)
    normal[..., 2] = 1.0  # Z (blue, pointing up)
    
    # Normalize vectors
    norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    normal = normal / (norm + 1e-8)
    
    # Convert to 0-255 range (normal maps are usually stored as (N+1)/2)
    normal = ((normal + 1) / 2 * 255).astype(np.uint8)
    
    return normal


def tile_texture(texture: np.ndarray, tiles_x: int, tiles_y: int) -> np.ndarray:
    """
    Create a tiled texture.
    
    Args:
        texture: Source texture
        tiles_x: Number of horizontal tiles
        tiles_y: Number of vertical tiles
    
    Returns:
        Tiled texture
    """
    return np.tile(texture, (tiles_y, tiles_x))


def add_noise_overlay(base_texture: np.ndarray, noise_texture: np.ndarray,
                      intensity: float = 0.1) -> np.ndarray:
    """
    Add noise as overlay to base texture.
    
    Args:
        base_texture: Base image (0-255)
        noise_texture: Noise to overlay (0-255)
        intensity: How much noise to add (0-1)
    
    Returns:
        Combined texture
    """
    # Normalize noise to -1 to 1
    noise_norm = (noise_texture.astype(np.float32) / 255.0 - 0.5) * 2
    
    # Apply intensity
    base_float = base_texture.astype(np.float32)
    result = base_float + noise_norm * intensity * 255
    
    return np.clip(result, 0, 255).astype(np.uint8)
