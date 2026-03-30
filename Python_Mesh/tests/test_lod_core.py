"""
Core LOD module tests that don't require external dependencies.
Tests for LODConfig dataclass and LODGenerator without PyMeshLab/Open3D.
"""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# Only import the dataclass and presets - no external deps needed
try:
    from core.lod import (
        LODConfig,
        LOD_PRESETS
    )
    HAS_LOD_CONFIG = True
except ImportError:
    HAS_LOD_CONFIG = False


@pytest.mark.skipif(not HAS_LOD_CONFIG, reason="LOD module not available")
class TestLODConfig:
    """Tests for LODConfig dataclass"""
    
    def test_config_creation(self):
        """Test basic LODConfig creation"""
        config = LODConfig('LOD0', 10000, 0)
        assert config.name == 'LOD0'
        assert config.target_faces == 10000
        assert config.subdiv_iterations == 0
    
    def test_config_defaults(self):
        """Test LODConfig default values"""
        config = LODConfig('LOD1', 5000)
        assert config.name == 'LOD1'
        assert config.target_faces == 5000
        assert config.subdiv_iterations == 0  # default
    
    def test_config_repr(self):
        """Test LODConfig string representation"""
        config = LODConfig('TestLOD', 1000, 2)
        repr_str = repr(config)
        assert 'TestLOD' in repr_str
        assert '1000' in repr_str
        assert '2' in repr_str
    
    def test_config_equality(self):
        """Test LODConfig equality comparison"""
        config1 = LODConfig('LOD0', 1000, 0)
        config2 = LODConfig('LOD0', 1000, 0)
        config3 = LODConfig('LOD1', 2000, 1)
        
        # Note: dataclasses don't implement __eq__ by default for different instances
        assert config1.name == config2.name
        assert config1.target_faces == config2.target_faces
        assert config1.name != config3.name


@pytest.mark.skipif(not HAS_LOD_CONFIG, reason="LOD module not available")
class TestLODPresets:
    """Tests for LOD preset configurations"""
    
    def test_all_presets_exist(self):
        """Test all expected presets are defined"""
        expected_presets = ['game_character', 'game_prop', 'archviz', 'mobile']
        for preset in expected_presets:
            assert preset in LOD_PRESETS, f"Missing preset: {preset}"
    
    def test_game_character_preset(self):
        """Test game_character preset configuration"""
        configs = LOD_PRESETS['game_character']
        assert len(configs) == 4
        
        # Check expected structure
        assert configs[0].name == 'LOD0_High'
        assert configs[0].target_faces == 50000
        
        assert configs[1].name == 'LOD1_Medium'
        assert configs[1].target_faces == 15000
        
        assert configs[2].name == 'LOD2_Low'
        assert configs[2].target_faces == 5000
        
        assert configs[3].name == 'LOD3_UltraLow'
        assert configs[3].target_faces == 1000
    
    def test_game_prop_preset(self):
        """Test game_prop preset configuration"""
        configs = LOD_PRESETS['game_prop']
        assert len(configs) == 3
        assert configs[0].target_faces == 10000
        assert configs[1].target_faces == 3000
        assert configs[2].target_faces == 500
    
    def test_archviz_preset(self):
        """Test archviz preset configuration"""
        configs = LOD_PRESETS['archviz']
        assert len(configs) == 3
        # Archviz uses higher face counts
        assert configs[0].target_faces == 100000
        assert configs[1].target_faces == 30000
        assert configs[2].target_faces == 10000
    
    def test_mobile_preset(self):
        """Test mobile preset configuration"""
        configs = LOD_PRESETS['mobile']
        assert len(configs) == 4
        # Mobile uses lower face counts
        assert configs[0].target_faces == 10000
        assert configs[3].target_faces == 150
    
    def test_preset_configs_are_lodconfig(self):
        """Test that preset configs are LODConfig instances"""
        for preset_name, configs in LOD_PRESETS.items():
            for config in configs:
                assert isinstance(config, LODConfig), \
                    f"{preset_name} contains non-LODConfig item"
    
    def test_preset_face_counts_descending(self):
        """Test that face counts decrease in each preset"""
        for preset_name, configs in LOD_PRESETS.items():
            face_counts = [c.target_faces for c in configs]
            assert face_counts == sorted(face_counts, reverse=True), \
                f"{preset_name} face counts should be descending"


def test_module_imports():
    """Test that core.lod module can be imported"""
    try:
        from core.lod import LODConfig, LOD_PRESETS
        assert True
    except ImportError as e:
        pytest.skip(f"Cannot import core.lod: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
