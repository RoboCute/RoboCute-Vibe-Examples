"""
RoboCute Integration Example

Shows how to use noise textures within the RoboCute framework.
This demonstrates loading noise textures as materials for 3D objects.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from noise_generators import SimplexNoiseGenerator, FBmNoiseGenerator
from utils import save_texture, create_normal_map
import numpy as np


def generate_terrain_textures(output_dir: str = "output/terrain"):
    """
    Generate a complete set of textures for terrain rendering in RoboCute.
    
    Creates:
    - Height map (displacement)
    - Albedo/diffuse map
    - Normal map
    - Roughness map
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating terrain texture set for RoboCute...")
    
    # Generate base height map using ridged fBm (good for mountains)
    fbm = FBmNoiseGenerator(width=1024, height=1024, seed=42)
    height_map = fbm.generate(
        scale=200.0,
        octaves=8,
        persistence=0.5,
        lacunarity=2.0,
        variation='ridged'
    )
    save_texture(height_map, f"{output_dir}/terrain_height.png")
    print("  Saved: terrain_height.png")
    
    # Create normal map from height
    normal_map = create_normal_map(height_map, strength=3.0)
    save_texture(normal_map, f"{output_dir}/terrain_normal.png")
    print("  Saved: terrain_normal.png")
    
    # Generate albedo with color variation
    simplex = SimplexNoiseGenerator(width=1024, height=1024, seed=43)
    
    # Base color variation
    color_var = simplex.generate(scale=150.0, octaves=4)
    
    # Create a simple RGB albedo (grayscale rock for now)
    # In practice, you'd blend multiple colors based on height
    albedo = np.stack([color_var, color_var, color_var], axis=-1)
    save_texture(albedo, f"{output_dir}/terrain_albedo.png")
    print("  Saved: terrain_albedo.png")
    
    # Roughness map (higher = rougher)
    roughness = simplex.generate(scale=100.0, octaves=3, seed=44)
    save_texture(roughness, f"{output_dir}/terrain_roughness.png")
    print("  Saved: terrain_roughness.png")
    
    print(f"\nTerrain textures saved to: {output_dir}")
    print("Use these in RoboCute material JSON like:")
    print("""
    {
        "textures": {
            "albedo": "terrain_albedo.png",
            "normal": "terrain_normal.png",
            "roughness": "terrain_roughness.png",
            "displacement": "terrain_height.png"
        }
    }
    """)


def generate_cloud_volume(output_dir: str = "output/volume"):
    """
    Generate 3D noise volume for volumetric clouds in RoboCute.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating volumetric cloud texture...")
    
    simplex = SimplexNoiseGenerator(width=128, height=128, seed=42)
    
    # Generate 3D noise
    volume = simplex.generate_3d(
        depth=128,
        scale=50.0,
        octaves=4
    )
    
    # Save as numpy for loading into RoboCute
    np.save(f"{output_dir}/cloud_volume.npy", volume)
    
    # Also save a few slices as preview images
    for i in [0, 32, 64, 96, 127]:
        slice_img = (volume[i] * 255).astype(np.uint8)
        save_texture(slice_img, f"{output_dir}/cloud_slice_{i:03d}.png")
    
    print(f"  Saved: cloud_volume.npy ({volume.shape})")
    print(f"  Preview slices saved to: {output_dir}")


def generate_procedural_materials(output_dir: str = "output/materials"):
    """
    Generate various procedural materials for RoboCute.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating procedural materials...")
    
    from noise_generators import WorleyNoiseGenerator, GaborNoiseGenerator
    
    # Stone/Granite - Worley noise
    worley = WorleyNoiseGenerator(width=512, height=512, seed=42)
    stone = worley.generate(num_points=80)
    save_texture(stone, f"{output_dir}/mat_stone.png")
    print("  Saved: mat_stone.png")
    
    # Fabric/Woven - Gabor noise
    gabor = GaborNoiseGenerator(width=512, height=512, seed=42)
    fabric = gabor.generate(num_kernels=300, frequency=0.08, sigma=15)
    save_texture(fabric, f"{output_dir}/mat_fabric.png")
    print("  Saved: mat_fabric.png")
    
    # Metal brushed - Anisotropic-like pattern
    fbm = FBmNoiseGenerator(width=512, height=512, seed=42)
    metal = fbm.generate(scale=50.0, octaves=3, variation='turbulence')
    save_texture(metal, f"{output_dir}/mat_metal.png")
    print("  Saved: mat_metal.png")
    
    print(f"\nMaterials saved to: {output_dir}")


def example_material_json():
    """
    Example of how to use these textures in RoboCute material JSON.
    """
    material_json = """
{
    "name": "ProceduralTerrain",
    "type": "PBR",
    "albedo": {
        "texture": "terrain/terrain_albedo.png",
        "color": [1.0, 1.0, 1.0]
    },
    "normal": {
        "texture": "terrain/terrain_normal.png",
        "strength": 1.0
    },
    "roughness": {
        "texture": "terrain/terrain_roughness.png",
        "value": 0.8
    },
    "displacement": {
        "texture": "terrain/terrain_height.png",
        "scale": 0.1
    },
    "ao": {
        "enabled": true,
        "strength": 1.0
    }
}
"""
    print("Example RoboCute Material JSON:")
    print(material_json)
    return material_json


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate textures for RoboCute")
    parser.add_argument("--terrain", action="store_true", help="Generate terrain textures")
    parser.add_argument("--volume", action="store_true", help="Generate volume textures")
    parser.add_argument("--materials", action="store_true", help="Generate material textures")
    parser.add_argument("--all", action="store_true", help="Generate everything")
    parser.add_argument("--json", action="store_true", help="Show example JSON")
    
    args = parser.parse_args()
    
    if args.all or not any([args.terrain, args.volume, args.materials, args.json]):
        args.terrain = args.volume = args.materials = True
    
    print("=" * 60)
    print("RoboCute Texture Generator")
    print("=" * 60)
    print()
    
    if args.terrain:
        generate_terrain_textures()
        print()
    
    if args.volume:
        generate_cloud_volume()
        print()
    
    if args.materials:
        generate_procedural_materials()
        print()
    
    if args.json:
        example_material_json()
    
    print("=" * 60)
    print("Done! Copy textures to your RoboCute project.")
    print("=" * 60)
