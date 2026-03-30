"""
Tests for decimation module.
"""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

try:
    from core.decimation import (
        decimate_mesh_pymeshlab,
        decimate_mesh_open3d,
        decimate_mesh_percentage
    )
    from core.utils import create_test_mesh
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


@pytest.mark.skipif(not HAS_OPEN3D, reason="Open3D not available")
def test_open3d_decimation():
    """Test Open3D decimation"""
    import open3d as o3d
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test mesh
        mesh = create_test_mesh('sphere', size=2.0)
        input_path = Path(tmpdir) / 'input.ply'
        o3d.io.write_triangle_mesh(str(input_path), mesh)
        
        original_faces = len(mesh.triangles)
        
        # Decimate
        output_path = Path(tmpdir) / 'output.ply'
        result = decimate_mesh_open3d(input_path, output_path, target_faces=100)
        
        # Check result has fewer faces
        assert len(result.triangles) < original_faces
        assert len(result.triangles) <= 100


@pytest.mark.skipif(not HAS_OPEN3D, reason="Open3D not available")
def test_percentage_decimation():
    """Test percentage-based decimation"""
    import open3d as o3d
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh = create_test_mesh('cube', size=2.0)
        input_path = Path(tmpdir) / 'input.ply'
        o3d.io.write_triangle_mesh(str(input_path), mesh)
        
        original_faces = len(mesh.triangles)
        
        output_path = Path(tmpdir) / 'output.ply'
        result = decimate_mesh_percentage(
            input_path, output_path,
            target_percentage=0.5,
            method='open3d'
        )
        
        new_faces = len(result.triangles)
        ratio = new_faces / original_faces
        
        # Should be around 50% (with some tolerance)
        assert 0.3 <= ratio <= 0.7


def test_invalid_percentage():
    """Test invalid percentage raises error"""
    with pytest.raises(ValueError):
        decimate_mesh_percentage('dummy', 'dummy', target_percentage=0)
    
    with pytest.raises(ValueError):
        decimate_mesh_percentage('dummy', 'dummy', target_percentage=1.5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
