"""
Tests for LOD (Level of Detail) module.
"""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# Check for dependencies
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False

try:
    from core.lod import (
        LODConfig,
        LODGenerator,
        generate_lod_chain,
        generate_preset_lod,
        LOD_PRESETS
    )
    from core.utils import create_test_mesh
    HAS_CORE = True
except ImportError:
    HAS_CORE = False


# Need both open3d and pymeshlab for most tests
HAS_DEPS = HAS_OPEN3D and HAS_PYMESHLAB and HAS_CORE


def test_lod_config():
    """Test LODConfig dataclass"""
    config = LODConfig('LOD0', 10000, 0)
    assert config.name == 'LOD0'
    assert config.target_faces == 10000
    assert config.subdiv_iterations == 0
    assert 'LOD0' in repr(config)


def test_lod_presets():
    """Test LOD preset configurations"""
    assert 'game_character' in LOD_PRESETS
    assert 'game_prop' in LOD_PRESETS
    assert 'archviz' in LOD_PRESETS
    assert 'mobile' in LOD_PRESETS
    
    # Check game_character preset has expected configs
    game_configs = LOD_PRESETS['game_character']
    assert len(game_configs) >= 3
    assert all(isinstance(c, LODConfig) for c in game_configs)


@pytest.mark.skipif(not HAS_DEPS, reason="Dependencies not available")
def test_lod_generator_init():
    """Test LODGenerator initialization"""
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = LODGenerator(tmpdir)
        assert generator.output_dir == Path(tmpdir)
        assert generator.results == []


@pytest.mark.skipif(not HAS_DEPS, reason="Dependencies not available")
def test_lod_generator_single_level():
    """Test generating single LOD level"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test mesh
        mesh = create_test_mesh('sphere', size=2.0)
        mesh = mesh.subdivide_loop(number_of_iterations=2)
        input_path = Path(tmpdir) / 'input.ply'
        o3d.io.write_triangle_mesh(str(input_path), mesh)
        
        original_faces = len(mesh.triangles)
        
        # Generate single LOD
        configs = [LODConfig('LOD0', target_faces=min(1000, original_faces // 2), subdiv_iterations=0)]
        generator = LODGenerator(tmpdir)
        results = generator.generate(input_path, configs)
        
        assert len(results) == 1
        assert results[0][0] == 'LOD0'


@pytest.mark.skipif(not HAS_DEPS, reason="Dependencies not available")
def test_lod_generator_multiple_levels():
    """Test generating multiple LOD levels"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create high-poly test mesh
        mesh = create_test_mesh('sphere', size=2.0)
        mesh = mesh.subdivide_loop(number_of_iterations=3)
        input_path = Path(tmpdir) / 'input.ply'
        o3d.io.write_triangle_mesh(str(input_path), mesh)
        
        original_faces = len(mesh.triangles)
        
        # Generate LOD chain
        configs = [
            LODConfig('LOD0', min(5000, original_faces // 2), 0),
            LODConfig('LOD1', min(2000, original_faces // 4), 0),
            LODConfig('LOD2', min(500, original_faces // 8), 0),
        ]
        
        generator = LODGenerator(tmpdir)
        results = generator.generate(input_path, configs)
        
        assert len(results) == 3
        
        # Check each LOD has decreasing face count
        prev_faces = original_faces
        for name, lod_mesh in results:
            if hasattr(lod_mesh, 'face_number'):
                faces = lod_mesh.face_number()
            else:
                faces = len(lod_mesh.triangles)
            assert faces < prev_faces or faces <= configs[0].target_faces
            prev_faces = faces


@pytest.mark.skipif(not HAS_DEPS, reason="Dependencies not available")
def test_lod_get_stats():
    """Test LODGenerator.get_stats()"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh = create_test_mesh('cube', size=2.0)
        mesh = mesh.subdivide_loop(number_of_iterations=2)
        input_path = Path(tmpdir) / 'input.ply'
        o3d.io.write_triangle_mesh(str(input_path), mesh)
        
        configs = [LODConfig('LOD0', 500, 0)]
        generator = LODGenerator(tmpdir)
        generator.generate(input_path, configs)
        
        stats = generator.get_stats()
        
        assert 'LOD0' in stats
        assert 'vertices' in stats['LOD0']
        assert 'faces' in stats['LOD0']
        assert stats['LOD0']['faces'] > 0


@pytest.mark.skipif(not HAS_DEPS, reason="Dependencies not available")
def test_generate_preset_lod():
    """Test generate_preset_lod function"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh = create_test_mesh('sphere', size=2.0)
        mesh = mesh.subdivide_loop(number_of_iterations=3)
        input_path = Path(tmpdir) / 'input.ply'
        o3d.io.write_triangle_mesh(str(input_path), mesh)
        
        output_dir = Path(tmpdir) / 'lods'
        
        results = generate_preset_lod(input_path, output_dir, preset='mobile')
        
        assert len(results) == len(LOD_PRESETS['mobile'])


@pytest.mark.skipif(not HAS_CORE, reason="Core module not available")
def test_invalid_preset():
    """Test invalid preset raises error"""
    with pytest.raises(ValueError):
        generate_preset_lod('dummy', 'dummy', preset='invalid_preset')


@pytest.mark.skipif(not HAS_DEPS, reason="Dependencies not available")
def test_generate_lod_chain():
    """Test generate_lod_chain convenience function"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh = create_test_mesh('cube', size=2.0)
        mesh = mesh.subdivide_loop(number_of_iterations=2)
        input_path = Path(tmpdir) / 'input.ply'
        o3d.io.write_triangle_mesh(str(input_path), mesh)
        
        lod_configs = [
            ('High', 1000, 0),
            ('Medium', 500, 0),
            ('Low', 200, 0),
        ]
        
        output_dir = Path(tmpdir) / 'chain_output'
        results = generate_lod_chain(input_path, output_dir, lod_configs)
        
        assert len(results) == 3
        assert results[0][0] == 'High'
        assert results[1][0] == 'Medium'
        assert results[2][0] == 'Low'


@pytest.mark.skipif(not HAS_CORE, reason="Core module not available")
def test_lod_with_nonexistent_input():
    """Test LOD generation with non-existent input file"""
    generator = LODGenerator(tempfile.gettempdir())
    
    with pytest.raises((FileNotFoundError, ImportError)):
        generator.generate('nonexistent_file.ply', [LODConfig('LOD0', 1000, 0)])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
