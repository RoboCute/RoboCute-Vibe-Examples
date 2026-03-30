"""
Unit tests for noise texture generators.

Run: pytest test_noise.py
"""

import unittest
import numpy as np
import sys
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
from utils import save_texture, create_normal_map, blend_textures


class TestBaseNoiseGenerator(unittest.TestCase):
    """Test base generator functionality."""
    
    def test_init_with_seed(self):
        """Test initialization with seed sets reproducible results."""
        gen1 = PerlinNoiseGenerator(width=64, height=64, seed=42)
        gen2 = PerlinNoiseGenerator(width=64, height=64, seed=42)
        
        tex1 = gen1.generate()
        tex2 = gen2.generate()
        
        np.testing.assert_array_equal(tex1, tex2)
    
    def test_different_seeds(self):
        """Test different seeds produce different results."""
        gen1 = PerlinNoiseGenerator(width=64, height=64, seed=42)
        gen2 = PerlinNoiseGenerator(width=64, height=64, seed=43)
        
        tex1 = gen1.generate()
        tex2 = gen2.generate()
        
        self.assertFalse(np.array_equal(tex1, tex2))
    
    def test_output_shape(self):
        """Test output has correct shape."""
        gen = PerlinNoiseGenerator(width=128, height=64)
        tex = gen.generate()
        
        self.assertEqual(tex.shape, (64, 128))


class TestPerlinNoise(unittest.TestCase):
    """Test Perlin noise generator."""
    
    def test_generate_default(self):
        """Test basic generation."""
        gen = PerlinNoiseGenerator(width=64, height=64)
        tex = gen.generate()
        
        self.assertEqual(tex.dtype, np.uint8)
        self.assertTrue(0 <= tex.min() <= 255)
        self.assertTrue(0 <= tex.max() <= 255)
    
    def test_generate_no_normalize(self):
        """Test generation without normalization."""
        gen = PerlinNoiseGenerator(width=64, height=64)
        tex = gen.generate(normalize=False)
        
        self.assertEqual(tex.dtype, np.float64)


class TestSimplexNoise(unittest.TestCase):
    """Test Simplex noise generator."""
    
    def test_generate_2d(self):
        """Test 2D generation."""
        gen = SimplexNoiseGenerator(width=64, height=64)
        tex = gen.generate()
        
        self.assertEqual(tex.shape, (64, 64))
        self.assertEqual(tex.dtype, np.uint8)
    
    def test_generate_3d(self):
        """Test 3D generation."""
        gen = SimplexNoiseGenerator(width=32, height=32)
        vol = gen.generate_3d(depth=16, scale=10.0, octaves=2)
        
        self.assertEqual(vol.shape, (16, 32, 32))
        self.assertTrue(0 <= vol.min() <= 1)
        self.assertTrue(0 <= vol.max() <= 1)


class TestValueNoise(unittest.TestCase):
    """Test Value noise generator."""
    
    def test_generate(self):
        """Test basic generation."""
        gen = ValueNoiseGenerator(width=64, height=64)
        tex = gen.generate(grid_size=8)
        
        self.assertEqual(tex.shape, (64, 64))
        self.assertEqual(tex.dtype, np.uint8)
    
    def test_different_grids(self):
        """Test different grid sizes."""
        gen = ValueNoiseGenerator(width=64, height=64)
        
        tex1 = gen.generate(grid_size=4)
        tex2 = gen.generate(grid_size=16)
        
        # Different grid sizes should produce different patterns
        self.assertFalse(np.array_equal(tex1, tex2))


class TestWorleyNoise(unittest.TestCase):
    """Test Worley noise generator."""
    
    def test_generate_f1(self):
        """Test F1 (closest point) generation."""
        gen = WorleyNoiseGenerator(width=64, height=64)
        tex = gen.generate(num_points=10, distance_order=1)
        
        self.assertEqual(tex.shape, (64, 64))
        self.assertEqual(tex.dtype, np.uint8)
    
    def test_generate_f2(self):
        """Test F2 (second closest) generation."""
        gen = WorleyNoiseGenerator(width=64, height=64)
        tex = gen.generate(num_points=10, distance_order=2)
        
        self.assertEqual(tex.shape, (64, 64))
    
    def test_generate_crack(self):
        """Test crack pattern generation."""
        gen = WorleyNoiseGenerator(width=64, height=64)
        tex = gen.generate_crack(num_points=10)
        
        self.assertEqual(tex.shape, (64, 64))


class TestFBmNoise(unittest.TestCase):
    """Test fBm noise generator."""
    
    def test_fbm(self):
        """Test standard fBm."""
        gen = FBmNoiseGenerator(width=64, height=64)
        tex = gen.generate(variation='fbm', octaves=4)
        
        self.assertEqual(tex.shape, (64, 64))
    
    def test_turbulence(self):
        """Test turbulence variation."""
        gen = FBmNoiseGenerator(width=64, height=64)
        tex = gen.generate(variation='turbulence', octaves=4)
        
        self.assertEqual(tex.shape, (64, 64))
    
    def test_ridged(self):
        """Test ridged variation."""
        gen = FBmNoiseGenerator(width=64, height=64)
        tex = gen.generate(variation='ridged', octaves=4)
        
        self.assertEqual(tex.shape, (64, 64))


class TestGaborNoise(unittest.TestCase):
    """Test Gabor noise generator."""
    
    def test_generate(self):
        """Test basic generation."""
        gen = GaborNoiseGenerator(width=64, height=64)
        tex = gen.generate(num_kernels=20, frequency=0.1, sigma=10.0)
        
        self.assertEqual(tex.shape, (64, 64))
        self.assertEqual(tex.dtype, np.uint8)


class TestFactoryFunction(unittest.TestCase):
    """Test create_noise factory function."""
    
    def test_create_perlin(self):
        """Test creating Perlin noise."""
        tex = create_noise('perlin', width=64, height=64)
        self.assertEqual(tex.shape, (64, 64))
    
    def test_create_simplex(self):
        """Test creating Simplex noise."""
        tex = create_noise('simplex', width=64, height=64)
        self.assertEqual(tex.shape, (64, 64))
    
    def test_create_invalid(self):
        """Test invalid noise type raises error."""
        with self.assertRaises(ValueError):
            create_noise('invalid', width=64, height=64)


class TestUtilities(unittest.TestCase):
    """Test utility functions."""
    
    def test_blend_textures(self):
        """Test texture blending."""
        tex1 = np.ones((64, 64), dtype=np.uint8) * 100
        tex2 = np.ones((64, 64), dtype=np.uint8) * 200
        
        blended = blend_textures(tex1, tex2, alpha=0.5)
        
        self.assertEqual(blended.shape, (64, 64))
        self.assertEqual(blended.dtype, np.uint8)
        # Blended value should be between 100 and 200
        self.assertTrue(140 <= blended.mean() <= 160)
    
    def test_create_normal_map(self):
        """Test normal map creation."""
        height = np.random.rand(64, 64) * 255
        normal = create_normal_map(height, strength=1.0)
        
        self.assertEqual(normal.shape, (64, 64, 3))
        self.assertEqual(normal.dtype, np.uint8)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_all_generators_run(self):
        """Test that all generators can produce output."""
        generators = [
            ('Perlin', PerlinNoiseGenerator(width=32, height=32)),
            ('Simplex', SimplexNoiseGenerator(width=32, height=32)),
            ('Value', ValueNoiseGenerator(width=32, height=32)),
            ('Worley', WorleyNoiseGenerator(width=32, height=32)),
            ('fBm', FBmNoiseGenerator(width=32, height=32)),
            ('Gabor', GaborNoiseGenerator(width=32, height=32)),
        ]
        
        for name, gen in generators:
            try:
                tex = gen.generate()
                self.assertIsNotNone(tex)
                self.assertEqual(tex.shape, (32, 32))
            except Exception as e:
                self.fail(f"{name} generator failed: {e}")


if __name__ == '__main__':
    unittest.main()
