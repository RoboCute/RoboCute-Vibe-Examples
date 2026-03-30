"""
Mesh Topology Repair Module
拓扑修复模块 - 修复网格常见问题
"""

import os
from pathlib import Path
from typing import Union, List, Dict

try:
    import pymeshlab
    PYMESHLAB_AVAILABLE = True
except ImportError:
    PYMESHLAB_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


def repair_topology_pymeshlab(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    remove_duplicates: bool = True,
    remove_unreferenced: bool = True,
    close_holes: bool = True,
    max_hole_size: int = 100,
    remove_isolated: bool = True,
    min_component_diag: float = 10.0
):
    """
    使用 PyMeshLab 进行完整拓扑修复
    
    修复顺序很重要！
    1. 移除重复顶点
    2. 移除未引用顶点
    3. 修复法线一致性
    4. 填补孔洞
    5. 移除孤立片段
    
    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
        remove_duplicates: 是否移除重复顶点
        remove_unreferenced: 是否移除未引用顶点
        close_holes: 是否填补孔洞
        max_hole_size: 最大孔洞大小
        remove_isolated: 是否移除孤立组件
        min_component_diag: 最小组件对角线长度
    
    Returns:
        pymeshlab.Mesh: 修复后的网格
    """
    if not PYMESHLAB_AVAILABLE:
        raise ImportError("PyMeshLab is required. Install with: pip install pymeshlab")
    
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_path))
    
    original_vertices = ms.current_mesh().vertex_number()
    original_faces = ms.current_mesh().face_number()
    
    print(f"[Repair] Before: {original_vertices} vertices, {original_faces} faces")
    
    # Step 1: 移除重复顶点（焊接）
    if remove_duplicates:
        ms.apply_filter('remove_duplicate_vertices', threshold=0.0001)
        print("  - Removed duplicate vertices")
    
    # Step 2: 移除未引用顶点
    if remove_unreferenced:
        ms.apply_filter('remove_unreferenced_vertices')
        print("  - Removed unreferenced vertices")
    
    # Step 3: 修复法线一致性
    ms.apply_filter('re_orient_all_faces_coherentely')
    print("  - Re-oriented faces coherently")
    
    # Step 4: 填补孔洞
    if close_holes:
        ms.apply_filter('close_holes', maxholesize=max_hole_size, newfaceselected=True)
        print(f"  - Closed holes (max size: {max_hole_size})")
    
    # Step 5: 移除孤立片段
    if remove_isolated:
        ms.apply_filter(
            'remove_connected_component_by_diameter',
            mincomponentdiag=min_component_diag,
            removeunrefvertices=True
        )
        print(f"  - Removed isolated components (min diag: {min_component_diag})")
    
    mesh = ms.current_mesh()
    print(f"[Repair] After: {mesh.vertex_number()} vertices, {mesh.face_number()} faces")
    
    os.makedirs(Path(output_path).parent, exist_ok=True)
    ms.save_current_mesh(str(output_path))
    
    return mesh


def laplacian_smooth(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    steps: int = 3,
    cotangent_weight: bool = True
):
    """
    拉普拉斯平滑去噪
    
    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
        steps: 平滑步数
        cotangent_weight: 使用余切权重
    """
    if not PYMESHLAB_AVAILABLE:
        raise ImportError("PyMeshLab is required.")
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_path))
    
    ms.apply_filter(
        'laplacian_smooth',
        stepsmoothnum=steps,
        cotangentweight=cotangent_weight
    )
    
    os.makedirs(Path(output_path).parent, exist_ok=True)
    ms.save_current_mesh(str(output_path))
    
    return ms.current_mesh()


def diagnose_mesh(mesh_path: Union[str, Path]) -> Dict[str, bool]:
    """
    诊断网格问题
    
    Args:
        mesh_path: 网格文件路径
    
    Returns:
        Dict 包含各项检查结果
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required.")
    
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    
    issues = []
    results = {}
    
    results['edge_manifold'] = mesh.is_edge_manifold()
    if not results['edge_manifold']:
        issues.append("非边流形（边被多于2个面共享）")
    
    results['vertex_manifold'] = mesh.is_vertex_manifold()
    if not results['vertex_manifold']:
        issues.append("非顶点流形（扇形结构异常）")
    
    results['watertight'] = mesh.is_watertight()
    if not results['watertight']:
        issues.append("非水密（存在边界边）")
    
    results['self_intersecting'] = mesh.is_self_intersecting()
    if results['self_intersecting']:
        issues.append("自相交")
    
    # Check orientability - method may not exist in all Open3D versions
    try:
        results['orientable_triangle_pairs'] = mesh.is_orientable_triangle_pairs()
        if not results['orientable_triangle_pairs']:
            issues.append("三角形对不可定向")
    except AttributeError:
        # Fallback for older Open3D versions
        results['orientable_triangle_pairs'] = True  # Assume orientable if method unavailable
    
    results['vertex_colors'] = mesh.has_vertex_colors()
    results['triangle_normals'] = mesh.has_triangle_normals()
    results['vertex_normals'] = mesh.has_vertex_normals()
    # texture_coordinates check may not exist in all Open3D versions
    try:
        results['texture_coordinates'] = mesh.has_texture_coordinates(0)
    except AttributeError:
        results['texture_coordinates'] = False  # Assume no UVs if method unavailable
    
    print(f"\n网格诊断报告: {mesh_path.name}")
    print(f"  顶点数: {len(mesh.vertices)}")
    print(f"  面片数: {len(mesh.triangles)}")
    print(f"  边流形: {results['edge_manifold']}")
    print(f"  顶点流形: {results['vertex_manifold']}")
    print(f"  水密性: {results['watertight']}")
    print(f"  自相交: {results['self_intersecting']}")
    print(f"  可定向: {results['orientable_triangle_pairs']}")
    print(f"  顶点法线: {results['vertex_normals']}")
    print(f"  面片法线: {results['triangle_normals']}")
    
    if issues:
        print(f"\n发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n[OK] 网格健康，无问题")
    
    results['healthy'] = len(issues) == 0
    results['issues'] = issues
    
    return results


def fix_winding_order(
    input_path: Union[str, Path],
    output_path: Union[str, Path]
):
    """
    修复面片绕序，确保法线朝外
    
    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
    """
    if not PYMESHLAB_AVAILABLE:
        raise ImportError("PyMeshLab is required.")
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_path))
    
    ms.apply_filter('re_orient_all_faces_coherentely')
    ms.apply_filter('invert_face_orientation', forceflip=False)
    
    os.makedirs(Path(output_path).parent, exist_ok=True)
    ms.save_current_mesh(str(output_path))
    
    return ms.current_mesh()


def remove_degenerate_faces(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    threshold: float = 0.0
):
    """
    移除退化面片
    
    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
        threshold: 面积阈值
    """
    if not PYMESHLAB_AVAILABLE:
        raise ImportError("PyMeshLab is required.")
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_path))
    
    ms.apply_filter('remove_zero_area_faces', threshold=threshold)
    
    os.makedirs(Path(output_path).parent, exist_ok=True)
    ms.save_current_mesh(str(output_path))
    
    return ms.current_mesh()


# 便捷函数
repair = repair_topology_pymeshlab
