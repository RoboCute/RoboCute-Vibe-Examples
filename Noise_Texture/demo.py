"""
Noise Texture Generator Demo

Demonstrates all noise generation algorithms with various parameters.
Run: python demo.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from noise_generators import (
    PerlinNoiseGenerator,
    SimplexNoiseGenerator,
    ValueNoiseGenerator,
    WorleyNoiseGenerator,
    FBmNoiseGenerator,
    GaborNoiseGenerator,
)
from utils import save_texture, display_texture, create_texture_grid, create_normal_map
import numpy as np


def demo_perlin_noise(output_dir: str = "output"):
    """Demonstrate Perlin noise with different scales."""
    print("Generating Perlin noise variations...")
    os.makedirs(output_dir, exist_ok=True)
    
    gen = PerlinNoiseGenerator(width=512, height=512, seed=42)
    
    variations = [
        ("perlin_scale_50.png", {"scale": 50.0, "octaves": 4}),
        ("perlin_scale_100.png", {"scale": 100.0, "octaves": 6}),
        ("perlin_scale_200.png", {"scale": 200.0, "octaves": 8}),
        ("perlin_clouds.png", {"scale": 150.0, "octaves": 8, "persistence": 0.6}),
    ]
    
    textures = []
    titles = []
    
    for filename, params in variations:
        texture = gen.generate(**params)
        save_texture(texture, os.path.join(output_dir, filename))
        textures.append(texture)
        titles.append(filename.replace('.png', '').replace('_', ' ').title())
        print(f"  Saved: {filename}")
    
    return textures, titles


def demo_simplex_noise(output_dir: str = "output"):
    """Demonstrate Simplex noise with fBm."""
    print("Generating Simplex noise variations...")
    os.makedirs(output_dir, exist_ok=True)
    
    gen = SimplexNoiseGenerator(width=512, height=512, seed=42)
    
    variations = [
        ("simplex_smooth.png", {"scale": 100.0, "octaves": 4}),
        ("simplex_detailed.png", {"scale": 80.0, "octaves": 8}),
        ("simplex_rough.png", {"scale": 50.0, "octaves": 6, "persistence": 0.7}),
    ]
    
    textures = []
    titles = []
    
    for filename, params in variations:
        texture = gen.generate(**params)
        save_texture(texture, os.path.join(output_dir, filename))
        textures.append(texture)
        titles.append(filename.replace('.png', '').replace('_', ' ').title())
        print(f"  Saved: {filename}")
    
    return textures, titles


def demo_value_noise(output_dir: str = "output"):
    """Demonstrate Value noise with different grid sizes."""
    print("Generating Value noise variations...")
    os.makedirs(output_dir, exist_ok=True)
    
    gen = ValueNoiseGenerator(width=512, height=512, seed=42)
    
    variations = [
        ("value_grid_8.png", {"grid_size": 8}),
        ("value_grid_16.png", {"grid_size": 16}),
        ("value_grid_32.png", {"grid_size": 32}),
        ("value_grid_64.png", {"grid_size": 64}),
    ]
    
    textures = []
    titles = []
    
    for filename, params in variations:
        texture = gen.generate(**params)
        save_texture(texture, os.path.join(output_dir, filename))
        textures.append(texture)
        titles.append(filename.replace('.png', '').replace('_', ' ').title())
        print(f"  Saved: {filename}")
    
    return textures, titles


def demo_worley_noise(output_dir: str = "output"):
    """Demonstrate Worley/Cellular noise."""
    print("Generating Worley noise variations...")
    os.makedirs(output_dir, exist_ok=True)
    
    gen = WorleyNoiseGenerator(width=512, height=512, seed=42)
    
    variations = [
        ("worley_cells_25.png", {"num_points": 25}),
        ("worley_cells_50.png", {"num_points": 50}),
        ("worley_cells_100.png", {"num_points": 100}),
        ("worley_f2.png", {"num_points": 50, "distance_order": 2}),
        ("worley_cracks.png", {"num_points": 40}),
    ]
    
    textures = []
    titles = []
    
    for i, (filename, params) in enumerate(variations):
        if filename == "worley_cracks.png":
            texture = gen.generate_crack(**{k: v for k, v in params.items() if k != 'filename'})
        else:
            texture = gen.generate(**params)
        save_texture(texture, os.path.join(output_dir, filename))
        textures.append(texture)
        titles.append(filename.replace('.png', '').replace('_', ' ').title())
        print(f"  Saved: {filename}")
    
    return textures, titles


def demo_fbm_noise(output_dir: str = "output"):
    """Demonstrate fBm (Fractal Brownian Motion) variations."""
    print("Generating fBm noise variations...")
    os.makedirs(output_dir, exist_ok=True)
    
    gen = FBmNoiseGenerator(width=512, height=512, seed=42)
    
    variations = [
        ("fbm_smooth.png", {"scale": 100.0, "octaves": 4, "variation": "fbm"}),
        ("fbm_terrain.png", {"scale": 120.0, "octaves": 8, "variation": "fbm", "persistence": 0.5}),
        ("fbm_turbulence.png", {"scale": 100.0, "octaves": 6, "variation": "turbulence"}),
        ("fbm_ridged.png", {"scale": 100.0, "octaves": 6, "variation": "ridged"}),
    ]
    
    textures = []
    titles = []
    
    for filename, params in variations:
        texture = gen.generate(**params)
        save_texture(texture, os.path.join(output_dir, filename))
        textures.append(texture)
        titles.append(filename.replace('.png', '').replace('_', ' ').title())
        print(f"  Saved: {filename}")
    
    return textures, titles


def demo_gabor_noise(output_dir: str = "output"):
    """Demonstrate Gabor noise."""
    print("Generating Gabor noise variations...")
    os.makedirs(output_dir, exist_ok=True)
    
    gen = GaborNoiseGenerator(width=512, height=512, seed=42)
    
    variations = [
        ("gabor_fine.png", {"num_kernels": 200, "frequency": 0.1, "sigma": 15}),
        ("gabor_medium.png", {"num_kernels": 150, "frequency": 0.05, "sigma": 25}),
        ("gabor_coarse.png", {"num_kernels": 100, "frequency": 0.02, "sigma": 40}),
    ]
    
    textures = []
    titles = []
    
    for filename, params in variations:
        texture = gen.generate(**params)
        save_texture(texture, os.path.join(output_dir, filename))
        textures.append(texture)
        titles.append(filename.replace('.png', '').replace('_', ' ').title())
        print(f"  Saved: {filename}")
    
    return textures, titles


def demo_combined_textures(output_dir: str = "output"):
    """Demonstrate combining different noise types."""
    print("Generating combined textures...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Perlin + Worley (cloud-like)
    perlin = PerlinNoiseGenerator(width=512, height=512, seed=42)
    worley = WorleyNoiseGenerator(width=512, height=512, seed=42)
    
    perlin_tex = perlin.generate(scale=150, octaves=6)
    worley_tex = worley.generate(num_points=30)
    
    # Blend them
    combined = (perlin_tex.astype(float) * 0.6 + worley_tex.astype(float) * 0.4).astype(np.uint8)
    save_texture(combined, os.path.join(output_dir, "combined_clouds.png"))
    print(f"  Saved: combined_clouds.png")
    
    # Create normal map from fBm terrain
    fbm = FBmNoiseGenerator(width=512, height=512, seed=42)
    terrain = fbm.generate(scale=100, octaves=6, variation="ridged")
    normal_map = create_normal_map(terrain, strength=2.0)
    save_texture(normal_map, os.path.join(output_dir, "terrain_normal_map.png"))
    print(f"  Saved: terrain_normal_map.png")


def demo_batch_generation(output_dir: str = "output"):
    """Generate all noise types with default parameters."""
    print("Batch generating all noise types...")
    os.makedirs(output_dir, exist_ok=True)
    
    generators = [
        ("perlin", PerlinNoiseGenerator(seed=42)),
        ("simplex", SimplexNoiseGenerator(seed=42)),
        ("value", ValueNoiseGenerator(seed=42)),
        ("worley", WorleyNoiseGenerator(seed=42)),
        ("fbm", FBmNoiseGenerator(seed=42)),
        ("gabor", GaborNoiseGenerator(seed=42)),
    ]
    
    textures = []
    titles = []
    
    for name, gen in generators:
        texture = gen.generate()
        save_texture(texture, os.path.join(output_dir, f"{name}_default.png"))
        textures.append(texture)
        titles.append(f"{name.title()} Noise")
        print(f"  Saved: {name}_default.png")
    
    return textures, titles


def main():
    """Run all demonstrations."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Noise Texture Generator Demo")
    print("=" * 60)
    print()
    
    all_textures = []
    all_titles = []
    
    try:
        textures, titles = demo_perlin_noise(output_dir)
        all_textures.extend(textures)
        all_titles.extend(titles)
    except Exception as e:
        print(f"  Skipped Perlin: {e}")
    
    print()
    
    try:
        textures, titles = demo_simplex_noise(output_dir)
        all_textures.extend(textures)
        all_titles.extend(titles)
    except Exception as e:
        print(f"  Skipped Simplex: {e}")
    
    print()
    
    try:
        textures, titles = demo_value_noise(output_dir)
        all_textures.extend(textures)
        all_titles.extend(titles)
    except Exception as e:
        print(f"  Skipped Value: {e}")
    
    print()
    
    try:
        textures, titles = demo_worley_noise(output_dir)
        all_textures.extend(textures)
        all_titles.extend(titles)
    except Exception as e:
        print(f"  Skipped Worley: {e}")
    
    print()
    
    try:
        textures, titles = demo_fbm_noise(output_dir)
        all_textures.extend(textures)
        all_titles.extend(titles)
    except Exception as e:
        print(f"  Skipped fBm: {e}")
    
    print()
    
    try:
        textures, titles = demo_gabor_noise(output_dir)
        all_textures.extend(textures)
        all_titles.extend(titles)
    except Exception as e:
        print(f"  Skipped Gabor: {e}")
    
    print()
    
    try:
        demo_combined_textures(output_dir)
    except Exception as e:
        print(f"  Skipped combined: {e}")
    
    print()
    
    try:
        textures, titles = demo_batch_generation(output_dir)
        all_textures.extend(textures)
        all_titles.extend(titles)
    except Exception as e:
        print(f"  Skipped batch: {e}")
    
    print()
    print("=" * 60)
    print(f"All textures saved to: {os.path.abspath(output_dir)}")
    print("=" * 60)
    
    # Try to create a grid visualization
    try:
        if all_textures:
            print("\nCreating texture grid visualization...")
            grid = create_texture_grid(all_textures[:6], all_titles[:6], cols=3)
            if grid.size > 0:
                from PIL import Image
                Image.fromarray(grid).save(os.path.join(output_dir, "texture_grid.png"))
                print("  Saved: texture_grid.png")
    except Exception as e:
        print(f"  Could not create grid: {e}")


if __name__ == "__main__":
    main()
