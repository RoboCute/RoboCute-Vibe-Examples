"""
Tests for repair module.
"""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

try:
    import open3d as o3d
    from core.repair import diagnose_mesh
    from core.utils import create_test_mesh
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@pytest.mark.skipif(not HAS_DEPS, reason="Dependencies not available")
def test_diagnose_healthy_mesh():
    """Test diagnosing a healthy mesh"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh = create_test_mesh('sphere', size=2.0)
        path = Path(tmpdir) / 'healthy.ply'
        o3d.io.write_triangle_mesh(str(path), mesh)
        
        result = diagnose_mesh(path)
        
        assert result['healthy'] is True
        assert result['edge_manifold'] is True
        assert result['watertight'] is True


@pytest.mark.skipif(not HAS_DEPS, reason="Dependencies not available")
def test_diagnose_mesh_with_holes():
    """Test diagnosing mesh with holes"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh = create_test_mesh('sphere', size=2.0)
        
        # Remove some triangles to create holes
        triangles = np.asarray(mesh.triangles)
        mesh.triangles = o3d.utility.Vector3iVector(triangles[100:])
        
        path = Path(tmpdir) / 'with_holes.ply'
        o3d.io.write_triangle_mesh(str(path), mesh)
        
        result = diagnose_mesh(path)
        
        assert result['watertight'] is False


@pytest.mark.skipif(not HAS_DEPS, reason="Dependencies not available")
def test_mesh_info():
    """Test getting mesh info"""
    from core.utils import get_mesh_info
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh = create_test_mesh('cube', size=2.0)
        path = Path(tmpdir) / 'cube.ply'
        o3d.io.write_triangle_mesh(str(path), mesh)
        
        info = get_mesh_info(path)
        
        assert 'vertices' in info
        assert 'faces' in info
        assert 'bounds' in info
        assert info['vertices'] > 0
        assert info['faces'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
