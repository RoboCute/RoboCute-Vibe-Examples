# Python 3D Mesh Processing Toolkit

A comprehensive Python implementation for 3D mesh processing including decimation, subdivision, topology repair, and LOD generation.

基于 Python_Mesh.md 实现的全功能 3D 网格处理工具包，包含减面、细分、拓扑修复和 LOD 生成功能。

---

## Features / 功能特性

| Module | Description | Key Features |
|--------|-------------|--------------|
| **Decimation** / 减面 | QEM algorithm mesh simplification | PyMeshLab (production) & Open3D (lightweight) |
| **Subdivision** / 细分 | Increase mesh resolution | Loop, Midpoint, Catmull-Clark |
| **Repair** / 修复 | Topology fixing | Duplicate removal, hole filling, normal fixing |
| **LOD** / 多细节层次 | Level-of-Detail generation | Batch LOD chain with presets |
| **Utils** / 工具 | Helper functions | Mesh info, comparison, batch processing |

---

## Quick Start / 快速开始

### Installation / 安装

```bash
# Clone or navigate to the project
cd samples/Python_Mesh

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage / 基本用法

```bash
# Decimate a mesh to 1000 faces
python mesh_tool.py decimate input.ply output.ply -t 1000

# Subdivide mesh using Loop algorithm (2 iterations = 16x faces)
python mesh_tool.py subdivide input.ply output.ply -a loop -i 2

# Repair mesh topology
python mesh_tool.py repair input.ply output.ply

# Diagnose mesh issues
python mesh_tool.py diagnose input.ply

# Generate LOD chain with preset
python mesh_tool.py lod input.ply -o ./lods --preset game_character
```

---

## Project Structure / 项目结构

```
Python_Mesh/
├── core/                      # Core processing modules
│   ├── decimation.py         # QEM decimation algorithms
│   ├── subdivision.py        # Loop, Midpoint, Catmull-Clark
│   ├── repair.py             # Topology repair functions
│   ├── lod.py                # LOD generation workflow
│   └── utils.py              # Utility functions
├── examples/                  # Example scripts
│   ├── example_decimation.py
│   ├── example_subdivision.py
│   ├── example_repair.py
│   ├── example_lod.py
│   └── example_complete_workflow.py
├── tests/                     # Unit tests
│   ├── test_decimation.py
│   ├── test_repair.py
│   └── test_subdivision.py
├── output/                    # Output directory (generated)
├── mesh_tool.py              # CLI tool
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## API Reference / API 参考

### Decimation / 减面

```python
from core.decimation import decimate_mesh_pymeshlab, decimate_mesh_open3d

# PyMeshLab (recommended for production)
result = decimate_mesh_pymeshlab(
    'input.obj',
    'output.obj',
    target_faces=10000,
    preservenormal=True,
    preservetopology=True
)

# Open3D (lightweight)
result = decimate_mesh_open3d(
    'input.obj',
    'output.obj',
    target_faces=5000
)
```

### Subdivision / 细分

```python
from core.subdivision import loop_subdivide, catmull_clark_subdivision

# Loop subdivision (triangles only, 4x faces per iteration)
mesh = loop_subdivide('input.ply', 'output.ply', iterations=1)

# Catmull-Clark (pure Python, quads output)
points = [[x1,y1,z1], ...]
faces = [[v0,v1,v2,v3], ...]  # quads
new_points, new_faces = catmull_clark_subdivision(points, faces, iterations=1)
```

### Repair / 修复

```python
from core.repair import repair_topology_pymeshlab, diagnose_mesh

# Diagnose issues
diagnose_mesh('input.ply')

# Full repair workflow
repair_topology_pymeshlab(
    'input.ply',
    'output.ply',
    remove_duplicates=True,
    close_holes=True,
    max_hole_size=100
)
```

### LOD Generation / LOD 生成

```python
from core.lod import LODGenerator, LODConfig, generate_preset_lod

# Custom LOD configs
configs = [
    LODConfig('LOD0_High', 50000, 0),
    LODConfig('LOD1_Medium', 10000, 0),
    LODConfig('LOD2_Low', 2000, 0),
]

generator = LODGenerator('./output')
results = generator.generate('input.ply', configs)
generator.print_report()

# Use presets
generate_preset_lod('input.ply', './lods', preset='game_character')
# Available presets: game_character, game_prop, archviz, mobile
```

---

## Examples / 示例

Run examples to see the toolkit in action:

```bash
# Run all examples
cd examples
python example_decimation.py
python example_subdivision.py
python example_repair.py
python example_lod.py
python example_complete_workflow.py
```

### Example: Complete Workflow

```python
from core.repair import repair_topology_pymeshlab
from core.lod import LODGenerator, LODConfig

# 1. Repair input mesh
repair_topology_pymeshlab('raw.obj', 'cleaned.obj')

# 2. Generate LOD chain
configs = [
    LODConfig('LOD0', 10000, 0),
    LODConfig('LOD1', 3000, 0),
    LODConfig('LOD2', 1000, 0),
]

generator = LODGenerator('./lods')
generator.generate('cleaned.obj', configs)
```

---

## Performance Reference / 性能参考

| Operation | 10K Faces | 100K Faces | 1M Faces |
|-----------|-----------|------------|----------|
| QEM Decimation (50%) | 0.05s | 0.3s | 2.5s |
| Loop Subdivision (1x) | 0.19s | 1.72s | ~15s |
| Topology Repair | 0.02s | 0.15s | 1.2s |

**Memory Tips:**
- For meshes > 1M faces, decimate to < 100K first
- Subdivision increases memory usage quadratically
- Use Open3D for lighter operations, PyMeshLab for quality

---

## Troubleshooting / 常见问题

### ImportError: No module named 'pymeshlab'
```bash
pip install pymeshlab
```

### Mesh subdivision fails
- Check if mesh is manifold: `diagnose_mesh('input.ply')`
- Repair first: `repair_topology_pymeshlab('input.ply', 'fixed.ply')`

### Out of memory on large meshes
- Decimate first before subdivision
- Process in chunks for batch operations

---

## Dependencies / 依赖

- **PyMeshLab** ≥ 2022.2 - Production mesh processing
- **Open3D** ≥ 0.17.0 - Lightweight operations & visualization
- **Trimesh** ≥ 3.20.0 - Additional utilities (optional)
- **NumPy** ≥ 1.24.0 - Array operations

---

## Testing / 测试

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_decimation.py -v
```

---

## License / 许可证

This project is part of the RoboCute samples collection.

---

## Conclusion / 总结

This project implements a complete 3D mesh processing pipeline based on the Python_Mesh.md guide:

1. **Decimation** - QEM algorithm for polygon reduction with quality preservation
2. **Subdivision** - Loop, Midpoint, and Catmull-Clark algorithms for detail enhancement
3. **Repair** - Comprehensive topology fixing including hole filling and normal correction
4. **LOD** - Automated Level-of-Detail generation with preset configurations

The modular design allows each component to be used independently or combined into complete workflows for game assets, architectural visualization, and real-time rendering applications.

本项目完整实现了 Python_Mesh.md 中描述的所有网格处理功能：

1. **减面** - 基于 QEM 算法的多边形简化，保持模型质量
2. **细分** - Loop、中点和 Catmull-Clark 算法增加模型细节
3. **修复** - 完整的拓扑修复流程，包括孔洞填补和法线修正
4. **LOD** - 自动化多细节层次生成，支持预设配置

模块化设计允许各组件独立使用或组合成完整工作流，适用于游戏资源、建筑可视化和实时渲染等应用场景。
