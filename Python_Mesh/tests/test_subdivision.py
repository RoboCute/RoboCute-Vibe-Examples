"""
Tests for subdivision module.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

try:
    from core.subdivision import (
        catmull_clark_subdivision,
        check_manifold,
        save_catmull_clark_mesh
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


def test_catmull_clark_basic():
    """Test basic Catmull-Clark subdivision"""
    # Simple quad
    points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    faces = [[0, 1, 2, 3]]
    
    new_points, new_faces = catmull_clark_subdivision(points, faces, iterations=1)
    
    # After one iteration: 1 quad -> 4 quads
    assert len(new_faces) == 4
    assert len(new_points) > len(points)


def test_catmull_clark_cube():
    """Test Catmull-Clark on cube"""
    # Cube vertices
    points = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ]
    
    # Cube faces (quads)
    faces = [
        [0, 1, 2, 3], [1, 5, 6, 2], [5, 4, 7, 6],
        [4, 0, 3, 7], [3, 2, 6, 7], [4, 5, 1, 0]
    ]
    
    original_faces = len(faces)
    
    # Subdivide twice
    new_points, new_faces = catmull_clark_subdivision(points, faces, iterations=2)
    
    # After 2 iterations: each quad becomes 16 quads
    expected_faces = original_faces * (4 ** 2)
    assert len(new_faces) == expected_faces


@pytest.mark.skipif(not HAS_OPEN3D, reason="Open3D not available")
def test_check_manifold():
    """Test manifold checking"""
    import open3d as o3d
    
    mesh = create_test_mesh('sphere', size=2.0)
    
    result = check_manifold(mesh)
    
    # Sphere should be manifold
    assert result is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
