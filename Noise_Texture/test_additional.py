"""
Additional test cases for noise texture generators.

Tests utility functions and edge cases not covered in test_noise.py.
Run: pytest test_additional.py -v
"""

import unittest
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from noise_generators import (
    PerlinNoiseGenerator,
    SimplexNoiseGenerator,
    ValueNoiseGenerator,
    WorleyNoiseGenerator,
    FBmNoiseGenerator,
    GaborNoiseGenerator,
    create_noise,
)
from utils import (
    save_texture,
    apply_colormap,
    create_texture_grid,
    tile_texture,
    add_noise_overlay,
)
from PIL import Image


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions comprehensively."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_texture = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_save_texture_default(self):
        """Test saving texture without colormap."""
        filepath = os.path.join(self.test_dir, "test.png")
        save_texture(self.test_texture, filepath)
        self.assertTrue(os.path.exists(filepath))
        
        # Verify image can be loaded
        with Image.open(filepath) as img:
            self.assertEqual(img.size, (64, 64))
            self.assertEqual(img.mode, 'L')
    
    def test_save_texture_with_colormap(self):
        """Test saving texture with colormap."""
        filepath = os.path.join(self.test_dir, "test_colored.png")
        save_texture(self.test_texture, filepath, colormap="heatmap")
        self.assertTrue(os.path.exists(filepath))
        
        with Image.open(filepath) as img:
            self.assertEqual(img.size, (64, 64))
            self.assertEqual(img.mode, 'RGB')
    
    def test_save_texture_creates_directory(self):
        """Test that save_texture creates output directory."""
        nested_dir = os.path.join(self.test_dir, "nested", "path")
        filepath = os.path.join(nested_dir, "test.png")
        save_texture(self.test_texture, filepath)
        self.assertTrue(os.path.exists(filepath))
    
    def test_apply_colormap_heatmap(self):
        """Test heatmap colormap."""
        result = apply_colormap(self.test_texture, 'heatmap')
        self.assertEqual(result.shape, (64, 64, 3))
        self.assertEqual(result.dtype, np.uint8)
    
    def test_apply_colormap_seismic(self):
        """Test seismic colormap."""
        result = apply_colormap(self.test_texture, 'seismic')
        self.assertEqual(result.shape, (64, 64, 3))
        self.assertEqual(result.dtype, np.uint8)
    
    def test_apply_colormap_terrain(self):
        """Test terrain colormap."""
        result = apply_colormap(self.test_texture, 'terrain')
        self.assertEqual(result.shape, (64, 64, 3))
        self.assertEqual(result.dtype, np.uint8)
    
    def test_apply_colormap_grayscale(self):
        """Test grayscale colormap."""
        result = apply_colormap(self.test_texture, 'grayscale')
        self.assertEqual(result.shape, (64, 64))
        self.assertEqual(result.dtype, np.uint8)
    
    def test_apply_colormap_normalized_input(self):
        """Test colormap with normalized input (0-1)."""
        normalized = self.test_texture / 255.0
        result = apply_colormap(normalized, 'heatmap')
        self.assertEqual(result.shape, (64, 64, 3))
    
    def test_tile_texture(self):
        """Test texture tiling."""
        texture = np.ones((32, 32), dtype=np.uint8) * 128
        tiled = tile_texture(texture, tiles_x=2, tiles_y=3)
        
        self.assertEqual(tiled.shape, (96, 64))  # 32*3, 32*2
        self.assertEqual(tiled.dtype, np.uint8)
        # All values should be the same
        self.assertTrue(np.all(tiled == 128))
    
    def test_add_noise_overlay(self):
        """Test noise overlay."""
        base = np.ones((64, 64), dtype=np.uint8) * 128
        noise = np.ones((64, 64), dtype=np.uint8) * 255
        
        result = add_noise_overlay(base, noise, intensity=0.5)
        
        self.assertEqual(result.shape, (64, 64))
        self.assertEqual(result.dtype, np.uint8)
        # Result should be brighter than base
        self.assertTrue(result.mean() > 128)
    
    def test_add_noise_overlay_zero_intensity(self):
        """Test noise overlay with zero intensity."""
        base = np.ones((64, 64), dtype=np.uint8) * 128
        noise = np.ones((64, 64), dtype=np.uint8) * 255
        
        result = add_noise_overlay(base, noise, intensity=0.0)
        
        # Result should be same as base
        np.testing.assert_array_equal(result, base)
    
    def test_create_texture_grid_valid(self):
        """Test creating texture grid."""
        textures = [
            np.ones((32, 32), dtype=np.uint8) * 50,
            np.ones((32, 32), dtype=np.uint8) * 100,
            np.ones((32, 32), dtype=np.uint8) * 150,
        ]
        titles = ["Low", "Medium", "High"]
        
        grid = create_texture_grid(textures, titles=titles, cols=2)
        
        # Should return a numpy array
        self.assertIsInstance(grid, np.ndarray)
        # When matplotlib is not installed, returns empty array
        # When available, returns RGBA image with ndim=3
        if grid.size > 0:
            self.assertEqual(grid.ndim, 3)  # RGBA image


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_perlin_small_size(self):
        """Test Perlin noise with very small size."""
        gen = PerlinNoiseGenerator(width=4, height=4)
        tex = gen.generate()
        self.assertEqual(tex.shape, (4, 4))
    
    def test_perlin_single_octave(self):
        """Test Perlin noise with single octave."""
        gen = PerlinNoiseGenerator(width=64, height=64)
        tex = gen.generate(octaves=1)
        self.assertEqual(tex.shape, (64, 64))
    
    def test_simplex_single_octave(self):
        """Test Simplex noise with single octave."""
        gen = SimplexNoiseGenerator(width=64, height=64)
        tex = gen.generate(octaves=1)
        self.assertEqual(tex.shape, (64, 64))
    
    def test_worley_single_point(self):
        """Test Worley noise with single feature point."""
        gen = WorleyNoiseGenerator(width=64, height=64)
        tex = gen.generate(num_points=1)
        self.assertEqual(tex.shape, (64, 64))
    
    def test_worley_many_points(self):
        """Test Worley noise with many feature points."""
        gen = WorleyNoiseGenerator(width=64, height=64)
        tex = gen.generate(num_points=500)
        self.assertEqual(tex.shape, (64, 64))
    
    def test_gabor_few_kernels(self):
        """Test Gabor noise with very few kernels."""
        gen = GaborNoiseGenerator(width=64, height=64)
        tex = gen.generate(num_kernels=1, frequency=0.1, sigma=10.0)
        self.assertEqual(tex.shape, (64, 64))
    
    def test_fbm_variations_produce_different_results(self):
        """Test that fBm variations produce different results."""
        gen = FBmNoiseGenerator(width=64, height=64, seed=42)
        
        tex_fbm = gen.generate(variation='fbm', octaves=4)
        tex_turb = gen.generate(variation='turbulence', octaves=4)
        tex_ridged = gen.generate(variation='ridged', octaves=4)
        
        # All should be different
        self.assertFalse(np.array_equal(tex_fbm, tex_turb))
        self.assertFalse(np.array_equal(tex_fbm, tex_ridged))
        self.assertFalse(np.array_equal(tex_turb, tex_ridged))
    
    def test_value_different_orders(self):
        """Test Value noise with different interpolation orders."""
        gen = ValueNoiseGenerator(width=64, height=64, seed=42)
        
        tex0 = gen.generate(grid_size=8, order=0)
        tex1 = gen.generate(grid_size=8, order=1)
        tex3 = gen.generate(grid_size=8, order=3)
        
        # Different orders should produce different results
        self.assertFalse(np.array_equal(tex0, tex1))
        self.assertFalse(np.array_equal(tex1, tex3))
    
    def test_reproducibility_with_same_seed(self):
        """Test that same seed produces identical results across generators."""
        generators = [
            ('Perlin', PerlinNoiseGenerator),
            ('Simplex', SimplexNoiseGenerator),
            ('Value', ValueNoiseGenerator),
            ('Worley', WorleyNoiseGenerator),
            ('FBm', FBmNoiseGenerator),
            ('Gabor', GaborNoiseGenerator),
        ]
        
        for name, GenClass in generators:
            gen1 = GenClass(width=32, height=32, seed=12345)
            gen2 = GenClass(width=32, height=32, seed=12345)
            
            # Reset numpy seed to ensure reproducibility
            np.random.seed(12345)
            tex1 = gen1.generate()
            
            np.random.seed(12345)
            tex2 = gen2.generate()
            
            np.testing.assert_array_equal(tex1, tex2, 
                err_msg=f"{name} generator not reproducible with same seed")


class TestFactoryFunctionVariations(unittest.TestCase):
    """Test create_noise factory with various parameters."""
    
    def test_create_value_noise(self):
        """Test creating Value noise via factory."""
        tex = create_noise('value', width=64, height=64, grid_size=16)
        self.assertEqual(tex.shape, (64, 64))
    
    def test_create_worley_noise(self):
        """Test creating Worley noise via factory."""
        tex = create_noise('worley', width=64, height=64, num_points=30)
        self.assertEqual(tex.shape, (64, 64))
    
    def test_create_fbm_noise(self):
        """Test creating fBm noise via factory."""
        tex = create_noise('fbm', width=64, height=64, variation='ridged')
        self.assertEqual(tex.shape, (64, 64))
    
    def test_create_gabor_noise(self):
        """Test creating Gabor noise via factory."""
        tex = create_noise('gabor', width=64, height=64, num_kernels=50)
        self.assertEqual(tex.shape, (64, 64))
    
    def test_create_noise_with_explicit_seed(self):
        """Test factory with explicit seed."""
        tex1 = create_noise('perlin', width=64, height=64, seed=999)
        tex2 = create_noise('perlin', width=64, height=64, seed=999)
        np.testing.assert_array_equal(tex1, tex2)


class TestGeneratorSaveMethod(unittest.TestCase):
    """Test the save method of generators."""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_perlin_save(self):
        """Test Perlin generator save method."""
        gen = PerlinNoiseGenerator(width=64, height=64)
        filepath = os.path.join(self.test_dir, "perlin.png")
        gen.save(filepath)
        self.assertTrue(os.path.exists(filepath))
    
    def test_simplex_save(self):
        """Test Simplex generator save method."""
        gen = SimplexNoiseGenerator(width=64, height=64)
        filepath = os.path.join(self.test_dir, "simplex.png")
        gen.save(filepath)
        self.assertTrue(os.path.exists(filepath))
    
    def test_generate_image_method(self):
        """Test generate_image method returns PIL Image."""
        gen = GaborNoiseGenerator(width=64, height=64)
        img = gen.generate_image(num_kernels=10)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (64, 64))


if __name__ == '__main__':
    unittest.main()
