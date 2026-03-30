# Noise Texture Generator

A comprehensive Python library for generating procedural noise textures used in computer graphics, game development, and digital art.

## Overview

This project implements six major categories of noise generation algorithms:

| Category | Algorithm | Best For |
|----------|-----------|----------|
| **Lattice-based** | Perlin Noise | Clouds, terrain, water |
| **Lattice-based** | Simplex Noise | 3D/4D noise, real-time GPU |
| **Lattice-based** | Value Noise | Fast prototyping |
| **Point-based** | Worley/Cellular Noise | Stones, tiles, organic patterns |
| **Fractal** | fBm (Fractal Brownian Motion) | Detailed terrain, natural surfaces |
| **Spectral** | Gabor Noise | Texture synthesis, frequency control |

## Installation

### Quick Install

```bash
pip install -r requirements.txt
```

### Install as Package

```bash
pip install -e .
```

### Dependencies

- **Core**: `numpy`, `Pillow`
- **Noise**: `noise` (Perlin), `opensimplex` (Simplex)
- **Geometry**: `scipy` (for Worley/Value noise)
- **Optional**: `matplotlib` (visualization), `numba` (performance)

## Quick Start

### Basic Usage

```python
from noise_generators import PerlinNoiseGenerator, SimplexNoiseGenerator

# Generate Perlin noise
perlin = PerlinNoiseGenerator(width=512, height=512, seed=42)
texture = perlin.generate(scale=100.0, octaves=6)
perlin.save("perlin.png")

# Generate Simplex noise with fBm
simplex = SimplexNoiseGenerator(width=512, height=512, seed=42)
texture = simplex.generate(scale=100.0, octaves=6, persistence=0.5)
simplex.save("simplex.png")
```

### Factory Function

```python
from noise_generators import create_noise

# Create any noise type with one function
texture = create_noise('perlin', width=512, height=512, scale=100.0, octaves=6)
texture = create_noise('worley', width=512, height=512, num_points=50)
```

## Algorithms

### 1. Perlin Noise

Classic gradient noise using random gradients and trilinear interpolation.

```python
gen = PerlinNoiseGenerator(seed=0)
texture = gen.generate(
    scale=100.0,        # Zoom level (higher = more zoomed in)
    octaves=6,          # Number of layers
    persistence=0.5,    # Amplitude decay per octave
    lacunarity=2.0      # Frequency increase per octave
)
```

**Applications**: Clouds, terrain, water, procedural textures

### 2. Simplex Noise

Improved Perlin noise using simplex grids (triangles in 2D, tetrahedrons in 3D).

```python
gen = SimplexNoiseGenerator(seed=42)
texture = gen.generate(scale=100.0, octaves=6)

# 3D volume for volumetric clouds
volume = gen.generate_3d(depth=64, scale=50.0, octaves=4)
```

**Applications**: 3D/4D noise, real-time generation, GPU shaders

### 3. Value Noise

Simple interpolated random values on a grid.

```python
gen = ValueNoiseGenerator(seed=42)
texture = gen.generate(
    grid_size=16,       # Number of grid cells
    order=3             # Interpolation: 0=nearest, 1=linear, 3=cubic
)
```

**Applications**: Fast prototyping, low-precision needs

### 4. Worley (Cellular) Noise

Based on distance to random feature points. Creates cell-like patterns.

```python
gen = WorleyNoiseGenerator(seed=42)

# Standard cellular pattern
texture = gen.generate(num_points=50, distance_order=1)

# Second-nearest for variation
texture = gen.generate(num_points=50, distance_order=2)

# Crack pattern (F2-F1)
texture = gen.generate_crack(num_points=40)
```

**Applications**: Stones, tiles, skin, biological tissue, erosion terrain

### 5. fBm (Fractal Brownian Motion)

Multi-octave combination of base noise for detailed fractal patterns.

```python
gen = FBmNoiseGenerator(seed=42)

# Standard fBm
texture = gen.generate(scale=100.0, octaves=6, variation='fbm')

# Turbulence (absolute values)
texture = gen.generate(scale=100.0, octaves=6, variation='turbulence')

# Ridged multifractal (for mountains)
texture = gen.generate(scale=100.0, octaves=6, variation='ridged')
```

**Applications**: Terrain, rocks, natural surface roughness

### 6. Gabor Noise

Uses Gabor kernels (sine modulated by Gaussian) for frequency-controllable noise.

```python
gen = GaborNoiseGenerator(seed=42)
texture = gen.generate(
    num_kernels=200,    # Number of kernels to place
    frequency=0.05,     # Sine wave frequency
    sigma=20.0          # Gaussian spread
)
```

**Applications**: Texture synthesis, material details, artistic effects

## Utilities

### Save & Display

```python
from utils import save_texture, display_texture

# Save with optional colormap
save_texture(texture, "output.png", colormap="heatmap")

# Display with matplotlib
display_texture(texture, title="My Noise", colormap="terrain")
```

### Create Normal Map

```python
from utils import create_normal_map

# Convert height map to normal map
normal_map = create_normal_map(height_texture, strength=2.0)
save_texture(normal_map, "normal.png")
```

### Create Texture Grid

```python
from utils import create_texture_grid

# Combine multiple textures into grid image
grid = create_texture_grid([tex1, tex2, tex3, tex4], 
                           titles=["A", "B", "C", "D"], 
                           cols=2)
```

## Running the Demo

```bash
python demo.py
```

This generates example textures for all noise types in the `output/` directory.

## Project Structure

```
Noise_Texture/
├── __init__.py           # Package initialization
├── noise_generators.py   # All noise generation algorithms
├── utils.py              # Helper utilities
├── demo.py               # Demonstration script
├── requirements.txt      # Dependencies
├── pyproject.toml        # Package configuration
└── README.md            # This file
```

## Algorithm Summary

| Algorithm | Type | Speed | Quality | Best For |
|-----------|------|-------|---------|----------|
| Perlin | Lattice | Medium | Good | General purpose |
| Simplex | Lattice | Fast | Better | 3D/4D, GPU |
| Value | Lattice | Fast | Basic | Prototyping |
| Worley | Point-based | Medium | High | Organic patterns |
| fBm | Fractal | Medium | High | Terrain, details |
| Gabor | Spectral | Slow | Very High | Controlled patterns |

## Performance Tips

1. **NumPy Vectorization**: All generators use NumPy arrays for efficiency
2. **Numba JIT**: Use `@jit` decorator for custom implementations
3. **GPU Acceleration**: Consider PyTorch/TensorFlow for batch generation
4. **Precomputation**: Cache noise tables for repeated use

## References

- Perlin, K. (1985). An Image Synthesizer. SIGGRAPH.
- Perlin, K. (2002). Improving Noise. SIGGRAPH.
- Worley, S. (1996). A Cellular Texture Basis Function. SIGGRAPH.
- Lagae et al. (2010). A Survey of Procedural Noise Functions. CGF.

## License

MIT License - See project repository for details.

---

**Part of the RoboCute Project** | Generated following `samples/Noise_Texture.md`
