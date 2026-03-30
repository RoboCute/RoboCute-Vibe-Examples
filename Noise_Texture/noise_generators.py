"""
Noise Texture Generators

Implements various noise generation algorithms:
- Perlin Noise: Gradient noise with smooth interpolation
- Simplex Noise: Improved Perlin with simplex grid
- Value Noise: Interpolated random values
- Worley Noise: Cellular/Voronoi-based noise
- fBm: Fractal Brownian Motion
- Gabor Noise: Frequency-controllable noise
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
from PIL import Image

try:
    import noise
    HAS_NOISE = True
except ImportError:
    HAS_NOISE = False

try:
    from opensimplex import OpenSimplex
    HAS_OPENSIMPLEX = True
except ImportError:
    HAS_OPENSIMPLEX = False

try:
    from scipy.ndimage import map_coordinates
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class BaseNoiseGenerator(ABC):
    """Base class for all noise generators."""
    
    def __init__(self, width: int = 512, height: int = 512, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    @abstractmethod
    def generate(self, **kwargs) -> np.ndarray:
        """Generate noise texture and return as numpy array."""
        pass
    
    def generate_image(self, **kwargs) -> Image.Image:
        """Generate noise texture and return as PIL Image."""
        array = self.generate(**kwargs)
        return Image.fromarray(array, mode='L')
    
    def save(self, filepath: str, **kwargs) -> None:
        """Generate and save noise texture to file."""
        img = self.generate_image(**kwargs)
        img.save(filepath)


class PerlinNoiseGenerator(BaseNoiseGenerator):
    """
    Perlin Noise Generator
    
    Classic gradient noise using interpolation of random gradients.
    Good for clouds, terrain, water, and procedural textures.
    """
    
    def __init__(self, width: int = 512, height: int = 512, seed: int = 0):
        if not HAS_NOISE:
            raise ImportError("'noise' library required. Install: pip install noise")
        super().__init__(width, height, seed)
    
    def generate(self, scale: float = 100.0, octaves: int = 6,
                 persistence: float = 0.5, lacunarity: float = 2.0,
                 normalize: bool = True) -> np.ndarray:
        """
        Generate 2D Perlin noise texture.
        
        Args:
            scale: Scale factor for coordinates (higher = more zoomed in)
            octaves: Number of noise layers to combine
            persistence: Amplitude multiplier for each octave
            lacunarity: Frequency multiplier for each octave
            normalize: Whether to normalize output to 0-255
        
        Returns:
            2D numpy array with noise values
        """
        world = np.zeros((self.height, self.width))
        
        for y in range(self.height):
            for x in range(self.width):
                world[y][x] = noise.pnoise2(
                    x / scale,
                    y / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=self.width,
                    repeaty=self.height,
                    base=self.seed if self.seed else 0
                )
        
        if normalize:
            world = (world - world.min()) / (world.max() - world.min()) * 255
            return world.astype(np.uint8)
        return world


class SimplexNoiseGenerator(BaseNoiseGenerator):
    """
    Simplex Noise Generator
    
    Improved version of Perlin noise using simplex grid.
    Better performance in higher dimensions, fewer artifacts.
    """
    
    def __init__(self, width: int = 512, height: int = 512, seed: int = 42):
        if not HAS_OPENSIMPLEX:
            raise ImportError("'opensimplex' required. Install: pip install opensimplex")
        super().__init__(width, height, seed)
        self.simplex = OpenSimplex(seed=seed)
    
    def generate(self, scale: float = 100.0, octaves: int = 6,
                 persistence: float = 0.5, lacunarity: float = 2.0,
                 normalize: bool = True) -> np.ndarray:
        """
        Generate 2D Simplex noise texture with fBm.
        
        Args:
            scale: Scale factor for coordinates
            octaves: Number of noise layers
            persistence: Amplitude decay per octave
            lacunarity: Frequency increase per octave
            normalize: Whether to normalize to 0-255
        """
        world = np.zeros((self.height, self.width))
        
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            for y in range(self.height):
                for x in range(self.width):
                    nx = x / scale * frequency
                    ny = y / scale * frequency
                    world[y][x] += self.simplex.noise2(nx, ny) * amplitude
            
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        
        if normalize:
            world = (world / max_value + 1) / 2 * 255
            return world.astype(np.uint8)
        return world / max_value
    
    def generate_3d(self, depth: int, scale: float = 50.0, 
                    octaves: int = 4) -> np.ndarray:
        """
        Generate 3D Simplex noise volume (for volumetric clouds, etc.)
        
        Args:
            depth: Z dimension size
            scale: Scale factor
            octaves: Number of layers
        
        Returns:
            3D numpy array with noise values
        """
        volume = np.zeros((depth, self.height, self.width))
        
        for z in range(depth):
            for y in range(self.height):
                for x in range(self.width):
                    nx = x / scale
                    ny = y / scale
                    nz = z / scale
                    
                    value = 0.0
                    amplitude = 1.0
                    frequency = 1.0
                    max_value = 0.0
                    
                    for _ in range(octaves):
                        value += self.simplex.noise3(
                            nx * frequency, ny * frequency, nz * frequency
                        ) * amplitude
                        max_value += amplitude
                        amplitude *= 0.5
                        frequency *= 2.0
                    
                    volume[z][y][x] = value / max_value
        
        # Normalize to 0-1
        volume = (volume + 1) / 2
        return volume


class ValueNoiseGenerator(BaseNoiseGenerator):
    """
    Value Noise Generator
    
    Interpolated random values on a grid.
    Simple but lower quality than gradient noise.
    """
    
    def __init__(self, width: int = 512, height: int = 512, seed: Optional[int] = None):
        if not HAS_SCIPY:
            raise ImportError("'scipy' required. Install: pip install scipy")
        super().__init__(width, height, seed)
    
    def generate(self, grid_size: int = 16, order: int = 3,
                 normalize: bool = True) -> np.ndarray:
        """
        Generate Value noise texture.
        
        Args:
            grid_size: Number of grid cells
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)
            normalize: Whether to normalize to 0-255
        
        Returns:
            2D numpy array with noise values
        """
        # Generate random values at grid vertices
        grid = np.random.rand(grid_size + 1, grid_size + 1)
        
        # Create interpolation coordinates
        x = np.linspace(0, grid_size, self.width)
        y = np.linspace(0, grid_size, self.height)
        xx, yy = np.meshgrid(x, y)
        
        # Interpolate
        coords = np.array([yy, xx])
        noise = map_coordinates(grid, coords, order=order, mode='wrap')
        
        if normalize:
            noise = (noise * 255).astype(np.uint8)
        return noise


class WorleyNoiseGenerator(BaseNoiseGenerator):
    """
    Worley / Cellular Noise Generator
    
    Based on distance to randomly distributed feature points.
    Creates cellular, stone-like, or organic patterns.
    """
    
    def __init__(self, width: int = 512, height: int = 512, seed: Optional[int] = None):
        if not HAS_SCIPY:
            raise ImportError("'scipy' required. Install: pip install scipy")
        super().__init__(width, height, seed)
    
    def generate(self, num_points: int = 50, distance_order: int = 1,
                 normalize: bool = True) -> np.ndarray:
        """
        Generate Worley noise texture.
        
        Args:
            num_points: Number of random feature points
            distance_order: Which nearest neighbor (1=closest, 2=2nd closest, etc.)
            normalize: Whether to normalize to 0-255
        
        Returns:
            2D numpy array with noise values
        """
        # Random feature points
        points = np.random.rand(num_points, 2) * [self.width, self.height]
        tree = cKDTree(points)
        
        # Calculate distance from each pixel to nearest feature points
        grid = np.indices((self.height, self.width)).reshape(2, -1).T
        distances, _ = tree.query(grid, k=distance_order)
        
        if distance_order > 1:
            noise = distances[:, distance_order - 1]
        else:
            noise = distances
        
        noise = noise.reshape(self.height, self.width)
        
        if normalize:
            noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
            return noise.astype(np.uint8)
        return noise
    
    def generate_crack(self, num_points: int = 50, normalize: bool = True) -> np.ndarray:
        """
        Generate crack-like pattern using F2-F1 distance.
        
        Args:
            num_points: Number of feature points
            normalize: Whether to normalize to 0-255
        
        Returns:
            2D numpy array with crack pattern
        """
        points = np.random.rand(num_points, 2) * [self.width, self.height]
        tree = cKDTree(points)
        
        grid = np.indices((self.height, self.width)).reshape(2, -1).T
        distances, _ = tree.query(grid, k=2)
        
        # F2 - F1 creates crack effect
        noise = distances[:, 1] - distances[:, 0]
        noise = noise.reshape(self.height, self.width)
        
        if normalize:
            noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
            return noise.astype(np.uint8)
        return noise


class FBmNoiseGenerator(BaseNoiseGenerator):
    """
    Fractal Brownian Motion Noise Generator
    
    Combines multiple octaves of base noise for detailed fractal patterns.
    Supports different variations: fBm, Turbulence, Ridged Multifractal.
    """
    
    VARIATION_FBM = 'fbm'
    VARIATION_TURBULENCE = 'turbulence'
    VARIATION_RIDGED = 'ridged'
    
    def __init__(self, width: int = 512, height: int = 512, seed: int = 42):
        if not HAS_OPENSIMPLEX:
            raise ImportError("'opensimplex' required. Install: pip install opensimplex")
        super().__init__(width, height, seed)
        self.simplex = OpenSimplex(seed=seed)
    
    def _fbm(self, x: float, y: float, octaves: int = 6,
             persistence: float = 0.5, lacunarity: float = 2.0) -> float:
        """Calculate fBm value at given coordinates."""
        total = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            total += self.simplex.noise2(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        
        return total / max_value
    
    def generate(self, scale: float = 100.0, octaves: int = 6,
                 persistence: float = 0.5, lacunarity: float = 2.0,
                 variation: str = 'fbm', normalize: bool = True) -> np.ndarray:
        """
        Generate fBm-based texture.
        
        Args:
            scale: Scale factor for coordinates
            octaves: Number of noise layers
            persistence: Amplitude decay per octave
            lacunarity: Frequency increase per octave
            variation: 'fbm', 'turbulence', or 'ridged'
            normalize: Whether to normalize to 0-255
        
        Returns:
            2D numpy array with fractal noise
        """
        world = np.zeros((self.height, self.width))
        
        for y in range(self.height):
            for x in range(self.width):
                nx = x / scale
                ny = y / scale
                value = self._fbm(nx, ny, octaves, persistence, lacunarity)
                
                if variation == self.VARIATION_TURBULENCE:
                    value = abs(value)
                elif variation == self.VARIATION_RIDGED:
                    value = (1 - abs(value)) ** 2
                
                world[y][x] = value
        
        if normalize:
            if variation == self.VARIATION_FBM:
                world = (world + 1) / 2 * 255
            else:
                world = world / world.max() * 255
            return world.astype(np.uint8)
        return world


class GaborNoiseGenerator(BaseNoiseGenerator):
    """
    Gabor Noise Generator
    
    Uses Gabor kernels (sine modulated by Gaussian) for frequency-controllable noise.
    Good for texture synthesis and controlled patterns.
    """
    
    def __init__(self, width: int = 512, height: int = 512, seed: Optional[int] = None):
        super().__init__(width, height, seed)
    
    def _gabor_kernel(self, frequency: float, sigma: float, 
                      theta: float) -> np.ndarray:
        """Generate a Gabor kernel."""
        size = int(sigma * 4)
        if size % 2 == 0:
            size += 1
        
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        X, Y = np.meshgrid(x, y)
        
        # Rotation
        X_rot = X * np.cos(theta) + Y * np.sin(theta)
        Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
        
        # Gabor function: Gaussian * Cosine
        gaussian = np.exp(-(X_rot**2 + Y_rot**2) / (2 * sigma**2))
        sinusoid = np.cos(2 * np.pi * frequency * X_rot)
        
        return gaussian * sinusoid
    
    def generate(self, num_kernels: int = 200, frequency: float = 0.05,
                 sigma: float = 20.0, normalize: bool = True) -> np.ndarray:
        """
        Generate Gabor noise texture.
        
        Args:
            num_kernels: Number of Gabor kernels to place
            frequency: Frequency of the sinusoidal component
            sigma: Standard deviation of the Gaussian envelope
            normalize: Whether to normalize to 0-255
        
        Returns:
            2D numpy array with Gabor noise
        """
        noise = np.zeros((self.height, self.width))
        
        for _ in range(num_kernels):
            # Random position
            cx = np.random.randint(0, self.width)
            cy = np.random.randint(0, self.height)
            
            # Random orientation
            theta = np.random.uniform(0, 2 * np.pi)
            
            # Generate kernel
            kernel = self._gabor_kernel(frequency, sigma, theta)
            k_h, k_w = kernel.shape
            
            # Add to noise field with boundary handling
            y_start = max(0, cy - k_h // 2)
            y_end = min(self.height, cy + k_h // 2 + 1)
            x_start = max(0, cx - k_w // 2)
            x_end = min(self.width, cx + k_w // 2 + 1)
            
            ky_start = k_h // 2 - (cy - y_start)
            ky_end = k_h // 2 + (y_end - cy)
            kx_start = k_w // 2 - (cx - x_start)
            kx_end = k_w // 2 + (x_end - cx)
            
            noise[y_start:y_end, x_start:x_end] += kernel[ky_start:ky_end, kx_start:kx_end]
        
        if normalize:
            noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
            return noise.astype(np.uint8)
        return noise


# Factory function for easy noise generation
def create_noise(noise_type: str, **kwargs) -> np.ndarray:
    """
    Factory function to create noise textures.
    
    Args:
        noise_type: One of 'perlin', 'simplex', 'value', 'worley', 'fbm', 'gabor'
        **kwargs: Arguments passed to the specific generator
    
    Returns:
        2D numpy array with noise values
    """
    generators = {
        'perlin': PerlinNoiseGenerator,
        'simplex': SimplexNoiseGenerator,
        'value': ValueNoiseGenerator,
        'worley': WorleyNoiseGenerator,
        'fbm': FBmNoiseGenerator,
        'gabor': GaborNoiseGenerator,
    }
    
    if noise_type not in generators:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose from {list(generators.keys())}")
    
    gen = generators[noise_type](
        width=kwargs.get('width', 512),
        height=kwargs.get('height', 512),
        seed=kwargs.get('seed', 42)
    )
    
    return gen.generate(**{k: v for k, v in kwargs.items() 
                          if k not in ['width', 'height', 'seed']})
