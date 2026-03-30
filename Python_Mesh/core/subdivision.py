"""
Mesh Subdivision Implementation
细分模块 - 增加网格细节
"""

import numpy as np
from typing import List, Tuple, Union
from pathlib import Path
import os

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


def check_manifold(mesh) -> bool:
    """
    检查网格流形状态
    
    Args:
        mesh: Open3D TriangleMesh
    
    Returns:
        bool: 是否为流形网格
    """
    is_edge_mf = mesh.is_edge_manifold()
    is_vertex_mf = mesh.is_vertex_manifold()
    is_watertight = mesh.is_watertight()
    
    print(f"  边流形 (Edge Manifold): {is_edge_mf}")
    print(f"  顶点流形 (Vertex Manifold): {is_vertex_mf}")
    print(f"  水密性 (Watertight): {is_watertight}")
    
    return is_edge_mf and is_vertex_mf


def loop_subdivide(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    iterations: int = 1
) -> 'o3d.geometry.TriangleMesh':
    """
    Loop 细分 - 每次迭代面数增长 4 倍
    
    ⚠️ 细分前必须检查流形性！
    
    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
        iterations: 迭代次数，1次=4倍面数，2次=16倍面数
    
    Returns:
        Open3D TriangleMesh
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required. Install with: pip install open3d")
    
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    
    print("检查网格流形性...")
    if not check_manifold(mesh):
        raise ValueError("网格非流形，需要先修复拓扑")
    
    original_faces = len(mesh.triangles)
    
    subdivided = mesh.subdivide_loop(number_of_iterations=iterations)
    subdivided.compute_vertex_normals()
    
    new_faces = len(subdivided.triangles)
    multiplier = 4 ** iterations
    
    print(f"Loop Subdivision: {original_faces} -> {new_faces} faces (x{multiplier})")
    
    os.makedirs(Path(output_path).parent, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), subdivided)
    
    return subdivided


def midpoint_subdivide(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    iterations: int = 1
) -> 'o3d.geometry.TriangleMesh':
    """
    中点细分 - 简单快速，每次迭代面数增长 4 倍
    
    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
        iterations: 迭代次数
    
    Returns:
        Open3D TriangleMesh
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required.")
    
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    
    original_faces = len(mesh.triangles)
    
    subdivided = mesh.subdivide_midpoint(number_of_iterations=iterations)
    subdivided.compute_vertex_normals()
    
    new_faces = len(subdivided.triangles)
    print(f"Midpoint Subdivision: {original_faces} -> {new_faces} faces")
    
    os.makedirs(Path(output_path).parent, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), subdivided)
    
    return subdivided


def catmull_clark_subdivision(
    points: List[List[float]],
    faces: List[List[int]],
    iterations: int = 1
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Catmull-Clark 细分 - 纯 Python 实现
    支持任意多边形输入，输出全四边形网格
    
    Args:
        points: [[x,y,z], ...] 顶点坐标
        faces: [[v0,v1,v2,v3], ...] 面片索引（支持多边形）
        iterations: 迭代次数
    
    Returns:
        new_points, new_faces: 细分后的顶点和面片
    """
    points = [list(p) for p in points]  # 复制
    faces = [list(f) for f in faces]
    
    for iteration in range(iterations):
        # 1. 计算面点 (面片中心)
        face_points = []
        for face in faces:
            fp = [sum(points[v][i] for v in face) / len(face) for i in range(3)]
            face_points.append(fp)
        
        # 2. 收集边信息
        edges = {}
        for fid, face in enumerate(faces):
            n = len(face)
            for i in range(n):
                v1, v2 = face[i], face[(i+1)%n]
                key = (min(v1,v2), max(v1,v2))
                if key not in edges:
                    p1, p2 = points[v1], points[v2]
                    edges[key] = {
                        'center': [(p1[i]+p2[i])/2 for i in range(3)],
                        'faces': []
                    }
                edges[key]['faces'].append(fid)
        
        # 3. 计算边点
        edge_points = {}
        for key, edata in edges.items():
            if len(edata['faces']) == 2:
                # 内部边：边中点 + 相邻两个面点的平均
                fp1 = face_points[edata['faces'][0]]
                fp2 = face_points[edata['faces'][1]]
                edge_points[key] = [
                    (edata['center'][i] + (fp1[i]+fp2[i])/2) / 2 
                    for i in range(3)
                ]
            else:
                # 边界边：就是边中点
                edge_points[key] = edata['center']
        
        # 4. 更新原顶点位置
        vertex_faces = [[] for _ in points]
        vertex_edges = [[] for _ in points]
        
        for fid, face in enumerate(faces):
            for v in face:
                vertex_faces[v].append(fid)
        
        for (v1, v2), edata in edges.items():
            vertex_edges[v1].append(edata['center'])
            vertex_edges[v2].append(edata['center'])
        
        new_vertex_points = []
        for vid, old_pos in enumerate(points):
            n = len(vertex_faces[vid])
            if n < 2:
                new_vertex_points.append(old_pos)
                continue
            
            # 平均相邻面点
            avg_face = [
                sum(face_points[fid][i] for fid in vertex_faces[vid]) / n 
                for i in range(3)
            ]
            
            # 平均相邻边点
            avg_edge = [
                sum(ep[i] for ep in vertex_edges[vid]) / len(vertex_edges[vid])
                for i in range(3)
            ]
            
            # Catmull-Clark 公式: (n-3)/n * P + 1/n * F + 2/n * R
            m1, m2, m3 = (n-3)/n, 1/n, 2/n
            new_pos = [
                m1*old_pos[i] + m2*avg_face[i] + m3*avg_edge[i] 
                for i in range(3)
            ]
            new_vertex_points.append(new_pos)
        
        # 5. 重建拓扑（生成四边形）
        edge_offset = {}
        offset = len(points)
        for i, key in enumerate(edges.keys()):
            edge_offset[key] = offset + i
        
        face_offset = offset + len(edges)
        
        new_points = new_vertex_points + list(edge_points.values()) + face_points
        new_faces = []
        
        for fid, face in enumerate(faces):
            fidx = face_offset + fid
            n = len(face)
            for i in range(n):
                v = face[i]
                e1 = edge_offset[(min(v, face[(i+1)%n]), max(v, face[(i+1)%n]))]
                e2 = edge_offset[(min(v, face[(i-1)%n]), max(v, face[(i-1)%n]))]
                new_faces.append([v, e1, fidx, e2])
        
        points, faces = new_points, new_faces
        print(f"  Iteration {iteration+1}: {len(points)} vertices, {len(faces)} faces")
    
    return points, faces


def save_catmull_clark_mesh(
    points: List[List[float]],
    faces: List[List[int]],
    output_path: Union[str, Path]
):
    """
    将 Catmull-Clark 结果保存为 OBJ 文件
    
    Args:
        points: 顶点列表
        faces: 面片列表（四边形）
        output_path: 输出路径
    """
    os.makedirs(Path(output_path).parent, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Catmull-Clark Subdivision Result\n")
        for p in points:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        
        for face in faces:
            # OBJ 使用 1-based indexing
            indices = ' '.join(str(v+1) for v in face)
            f.write(f"f {indices}\n")
    
    print(f"Saved to: {output_path}")
