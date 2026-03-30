"""
Houdini-Style Procedural Scene Generation for RoboCute
======================================================

This module implements Houdini's procedural scene generation nodes using RoboCute Python API.

Equivalent Houdini Nodes:
- Scatter / Scatter::2.0 → scatter_on_surface, poisson_disk_sampling
- Copy to Points → create_instances_at_points
- Copy and Transform → create_grid_instances
- Points from Volume → generate_volume_points
- Hexagonal Packing → hexagonal_packing
- Relax → relax_points
- Group by Bounding Box → ScenePartitioner.group_by_bounding_box
- Attribute Wrangle → ScenePartitioner.apply_condition
- L-System → l_system_fractal
- For-Each → foreach_batch_process
- PolyExtrude → extrude_building
- Variant LOP → VariantSet
- Component Builder → ComponentBuilder
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
from dataclasses import dataclass


# =============================================================================
# 1. 基础散布与复制系统 (Scatter & Copy to Points)
# =============================================================================

@dataclass
class ScatterPoint:
    """Represents a scatter point with position, scale, and rotation."""
    position: Tuple[float, float, float]
    scale: float = 1.0
    rotation: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)  # quaternion
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


def scatter_on_surface(
    bounds: Tuple[float, float] = (-50, 50),
    count: int = 100,
    y_height: float = 0.0,
    scale_range: Tuple[float, float] = (0.5, 1.5),
    seed: int = 0
) -> List[ScatterPoint]:
    """
    Houdini: Scatter / Scatter::2.0
    
    Randomly scatter points on a surface with random scale and rotation.
    
    Args:
        bounds: (min, max) bounds for x and z coordinates
        count: Number of points to scatter
        y_height: Height of the surface (y coordinate)
        scale_range: (min, max) range for random scale
        seed: Random seed for reproducibility
        
    Returns:
        List of ScatterPoint objects
    """
    np.random.seed(seed)
    points = []
    
    for i in range(count):
        x = np.random.uniform(bounds[0], bounds[1])
        z = np.random.uniform(bounds[0], bounds[1])
        
        scale = np.random.uniform(*scale_range)
        angle = np.random.uniform(0, 2 * np.pi)
        # Quaternion for Y-axis rotation
        rotation = (0.0, 1.0, 0.0, angle)
        
        point = ScatterPoint(
            position=(x, y_height, z),
            scale=scale,
            rotation=rotation,
            attributes={"id": i, "pscale": scale}
        )
        points.append(point)
    
    return points


def poisson_disk_sampling(
    radius: float = 2.0,
    width: float = 100.0,
    height: float = 100.0,
    max_attempts: int = 30,
    seed: int = 0
) -> List[Tuple[float, float]]:
    """
    Houdini: Poisson Disk Sampling (via VEX or Labs tools)
    
    Blue noise sampling that generates natural point distribution without overlap.
    
    Args:
        radius: Minimum distance between points
        width: Width of the sampling area
        height: Height of the sampling area
        max_attempts: Maximum attempts to place each point
        seed: Random seed
        
    Returns:
        List of (x, y) point coordinates
    """
    np.random.seed(seed)
    
    cell_size = radius / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))
    
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
    points = []
    active = []
    
    # Initial point at center
    x, y = width / 2, height / 2
    points.append((x, y))
    active.append((x, y))
    grid[int(x / cell_size)][int(y / cell_size)] = (x, y)
    
    while active:
        idx = np.random.randint(len(active))
        center = active[idx]
        found = False
        
        for _ in range(max_attempts):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(radius, 2 * radius)
            new_x = center[0] + dist * np.cos(angle)
            new_y = center[1] + dist * np.sin(angle)
            
            if 0 <= new_x < width and 0 <= new_y < height:
                grid_x = int(new_x / cell_size)
                grid_y = int(new_y / cell_size)
                
                if grid[grid_x][grid_y] is None:
                    # Check neighbors
                    valid = True
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = grid_x + dx, grid_y + dy
                            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                                neighbor = grid[nx][ny]
                                if neighbor:
                                    d = np.sqrt((new_x - neighbor[0])**2 + (new_y - neighbor[1])**2)
                                    if d < radius:
                                        valid = False
                                        break
                        if not valid:
                            break
                    
                    if valid:
                        points.append((new_x, new_y))
                        active.append((new_x, new_y))
                        grid[grid_x][grid_y] = (new_x, new_y)
                        found = True
                        break
        
        if not found:
            active.pop(idx)
    
    return points


def hexagonal_packing(
    spacing: float = 2.0,
    rows: int = 10,
    cols: int = 10,
    center_origin: bool = True
) -> List[Tuple[float, float, float]]:
    """
    Houdini: Hexagonal Packing
    
    Creates hexagonally packed points for regular arrangements.
    
    Args:
        spacing: Distance between adjacent points
        rows: Number of rows
        cols: Number of columns
        center_origin: Whether to center the grid at origin
        
    Returns:
        List of (x, y, z) positions
    """
    points = []
    offset_x = 0.0
    offset_z = 0.0
    
    # Calculate base offsets for centering
    for row in range(rows):
        for col in range(cols):
            x = col * spacing
            z = row * spacing * np.sqrt(3) / 2
            if row % 2 == 1:
                x += spacing / 2
            points.append((x, 0.0, z))
    
    # Center the entire grid at origin if requested
    if center_origin and points:
        xs = [p[0] for p in points]
        zs = [p[2] for p in points]
        center_x = (min(xs) + max(xs)) / 2
        center_z = (min(zs) + max(zs)) / 2
        points = [(x - center_x, 0.0, z - center_z) for x, _, z in points]
    
    return points


def rectangular_packing(
    spacing: float = 2.0,
    rows: int = 10,
    cols: int = 10,
    center_origin: bool = True
) -> List[Tuple[float, float, float]]:
    """
    Houdini: Rectangular Packing
    
    Creates rectangular grid of points.
    
    Args:
        spacing: Distance between adjacent points
        rows: Number of rows
        cols: Number of columns
        center_origin: Whether to center the grid at origin
        
    Returns:
        List of (x, y, z) positions
    """
    points = []
    offset_x = 0.0
    offset_z = 0.0
    
    if center_origin:
        offset_x = -(cols - 1) * spacing / 2
        offset_z = -(rows - 1) * spacing / 2
    
    for row in range(rows):
        for col in range(cols):
            x = col * spacing + offset_x
            z = row * spacing + offset_z
            points.append((x, 0.0, z))
    
    return points


def relax_points(
    points: List[Tuple[float, float]],
    iterations: int = 50,
    repulsion_radius: float = 2.0,
    damping: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Houdini: Relax
    
    Applies point relaxation to avoid overlaps and create natural distribution.
    
    Args:
        points: Initial point positions
        iterations: Number of relaxation iterations
        repulsion_radius: Distance for point repulsion
        damping: Movement damping factor
        
    Returns:
        Relaxed point positions
    """
    points = np.array(points, dtype=np.float32)
    
    for _ in range(iterations):
        displacements = np.zeros_like(points)
        
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i != j:
                    diff = p1 - p2
                    dist = np.linalg.norm(diff)
                    if dist < repulsion_radius and dist > 0.001:
                        force = (repulsion_radius - dist) / repulsion_radius
                        displacements[i] += (diff / dist) * force
        
        points += displacements * damping
    
    return [(float(p[0]), float(p[1])) for p in points]


# =============================================================================
# 2. 约束与条件系统 (Group, Blast, Attribute Wrangle)
# =============================================================================

class ScenePartitioner:
    """
    Houdini: Group + Blast + Attribute Wrangle
    
    Scene partitioning and conditional selection system.
    """
    
    def __init__(self):
        self.groups: Dict[str, List[Any]] = {}
    
    def group_by_bounding_box(
        self,
        points: List[ScatterPoint],
        min_point: Tuple[float, float, float],
        max_point: Tuple[float, float, float],
        group_name: str
    ) -> List[ScatterPoint]:
        """
        Houdini: Group by Bounding Box
        
        Creates a selection group based on spatial bounds.
        """
        selected = []
        for point in points:
            pos = point.position
            if (min_point[0] <= pos[0] <= max_point[0] and
                min_point[1] <= pos[1] <= max_point[1] and
                min_point[2] <= pos[2] <= max_point[2]):
                selected.append(point)
        
        self.groups[group_name] = selected
        return selected
    
    def filter_by_attribute(
        self,
        points: List[ScatterPoint],
        attribute_name: str,
        min_value: float,
        max_value: float
    ) -> List[ScatterPoint]:
        """
        Houdini: Attribute-based filtering
        
        Filters points by attribute value range.
        """
        return [
            p for p in points
            if attribute_name in p.attributes
            and min_value <= p.attributes[attribute_name] <= max_value
        ]
    
    def apply_condition(
        self,
        items: List[Any],
        condition_func: Callable[[Any], bool],
        action_func: Callable[[Any], None]
    ) -> None:
        """
        Houdini: Attribute Wrangle style conditional execution
        
        Applies action to items that satisfy the condition.
        
        Args:
            items: List of items to process
            condition_func: Function returning True/False for each item
            action_func: Action to perform on items passing condition
        """
        for item in items:
            if condition_func(item):
                action_func(item)
    
    def partition(
        self,
        points: List[ScatterPoint],
        condition_func: Callable[[ScatterPoint], bool]
    ) -> Tuple[List[ScatterPoint], List[ScatterPoint]]:
        """
        Houdini: Partition (Blast)
        
        Splits points into two groups based on condition.
        """
        group_a = []
        group_b = []
        
        for point in points:
            if condition_func(point):
                group_a.append(point)
            else:
                group_b.append(point)
        
        return group_a, group_b


# =============================================================================
# 3. 程序化建模 (L-System, PolyExtrude, For-Each)
# =============================================================================

class LSystem:
    """
    Houdini: L-System
    
    L-System fractal generator for plants, roads, and branching structures.
    """
    
    def __init__(
        self,
        axiom: str = "F",
        rules: Optional[Dict[str, str]] = None,
        angle: float = 25.7,
        step_size: float = 1.0
    ):
        self.axiom = axiom
        self.rules = rules or {"F": "F[+F]F[-F]F"}
        self.angle = np.radians(angle)
        self.step_size = step_size
    
    def generate(self, iterations: int = 4) -> str:
        """Generate L-System string after n iterations."""
        result = self.axiom
        for _ in range(iterations):
            new_result = ""
            for char in result:
                new_result += self.rules.get(char, char)
            result = new_result
        return result
    
    def interpret(self, lstring: str) -> Dict[str, Any]:
        """
        Interpret L-System string and generate geometry data.
        
        Returns:
            Dictionary with 'branches' and 'leaves' lists
        """
        branches = []  # List of (start, end) tuples
        leaves = []    # List of positions
        
        stack = []
        current_pos = np.array([0.0, 0.0, 0.0])
        current_dir = np.array([0.0, 1.0, 0.0])  # Y-up
        
        for cmd in lstring:
            if cmd == "F":  # Forward
                new_pos = current_pos + current_dir * self.step_size
                branches.append((current_pos.copy(), new_pos.copy()))
                current_pos = new_pos
            elif cmd == "+":  # Turn right
                rotation = np.array([
                    [np.cos(self.angle), -np.sin(self.angle), 0],
                    [np.sin(self.angle), np.cos(self.angle), 0],
                    [0, 0, 1]
                ])
                current_dir = rotation @ current_dir
            elif cmd == "-":  # Turn left
                rotation = np.array([
                    [np.cos(-self.angle), -np.sin(-self.angle), 0],
                    [np.sin(-self.angle), np.cos(-self.angle), 0],
                    [0, 0, 1]
                ])
                current_dir = rotation @ current_dir
            elif cmd == "[":  # Save state
                stack.append((current_pos.copy(), current_dir.copy()))
            elif cmd == "]":  # Restore state
                current_pos, current_dir = stack.pop()
                leaves.append(current_pos.copy())
        
        return {"branches": branches, "leaves": leaves}


def foreach_batch_process(
    items: List[Any],
    process_func: Callable[[Any, int], Any]
) -> List[Any]:
    """
    Houdini: For-Each Connected Piece / For-Each Number
    
    Batch process items with an index.
    
    Args:
        items: List of items to process
        process_func: Function(item, index) -> result
        
    Returns:
        List of results
    """
    results = []
    for i, item in enumerate(items):
        result = process_func(item, i)
        if result is not None:
            results.append(result)
    return results


# =============================================================================
# 4. Solaris / Stage 层 (Variant LOP, Component Builder)
# =============================================================================

class VariantSet:
    """
    Houdini: Variant LOP
    
    Manages different asset variants at the same location.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.variants: Dict[str, Any] = {}
        self.active_variant: Optional[str] = None
    
    def add_variant(self, variant_name: str, data: Any) -> None:
        """Add a variant."""
        self.variants[variant_name] = data
    
    def switch_variant(self, variant_name: str) -> bool:
        """Switch active variant. Returns True if successful."""
        if variant_name in self.variants:
            self.active_variant = variant_name
            return True
        return False
    
    def get_active(self) -> Optional[Any]:
        """Get currently active variant data."""
        if self.active_variant:
            return self.variants[self.active_variant]
        return None


class ComponentBuilder:
    """
    Houdini: Component Builder
    
    Packages procedural generation logic into reusable components.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.parameters: Dict[str, Dict[str, Any]] = {}
        self.build_func: Optional[Callable] = None
    
    def add_parameter(
        self,
        name: str,
        default_value: Any,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> None:
        """Add an exposed parameter with optional min/max constraints."""
        self.parameters[name] = {
            "value": default_value,
            "min": min_val,
            "max": max_val
        }
    
    def set_build_function(self, func: Callable) -> None:
        """Set the build function."""
        self.build_func = func
    
    def build(self, **kwargs) -> Any:
        """Execute build with merged parameters."""
        params = {k: v["value"] for k, v in self.parameters.items()}
        params.update(kwargs)
        
        # Validate constraints
        for name, value in params.items():
            if name in self.parameters:
                p = self.parameters[name]
                if p["min"] is not None and value < p["min"]:
                    params[name] = p["min"]
                if p["max"] is not None and value > p["max"]:
                    params[name] = p["max"]
        
        if self.build_func:
            return self.build_func(**params)
        return None


# =============================================================================
# 5. 实用工具 (City Street Generator, Tree Generator)
# =============================================================================

def generate_city_block_points(
    block_size: float = 50.0,
    street_width: float = 10.0,
    num_blocks_x: int = 5,
    num_blocks_z: int = 5
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float, float]]]:
    """
    Houdini: Labs City Street Generator
    
    Generates city block centers and street bounds.
    
    Returns:
        Tuple of (block_centers, street_rectangles)
        street_rectangles are (x, z, width, depth)
    """
    block_centers = []
    streets = []
    
    full_block = block_size + street_width
    
    for bx in range(num_blocks_x):
        for bz in range(num_blocks_z):
            # Block center
            block_x = (bx - num_blocks_x // 2) * full_block
            block_z = (bz - num_blocks_z // 2) * full_block
            block_centers.append((block_x, block_z))
            
            # Street rectangle (horizontal)
            if bz < num_blocks_z - 1:
                streets.append((
                    block_x - block_size / 2,
                    block_z + block_size / 2,
                    block_size,
                    street_width
                ))
            
            # Street rectangle (vertical)
            if bx < num_blocks_x - 1:
                streets.append((
                    block_x + block_size / 2,
                    block_z - block_size / 2,
                    street_width,
                    block_size
                ))
    
    return block_centers, streets


def generate_tree_structure(
    tree_type: str = "oak",
    height: float = 10.0,
    seed: int = 0
) -> Dict[str, Any]:
    """
    Houdini: Labs Tree Generator (simplified)
    
    Generates procedural tree structure data.
    
    Returns:
        Dictionary with trunk, branches, and foliage data
    """
    np.random.seed(seed)
    
    # Trunk
    trunk_height = height * 0.6
    trunk_radius = 0.3
    
    # Branches
    branches = []
    num_branches = np.random.randint(3, 6)
    
    for i in range(num_branches):
        branch_data = {
            "height": height * 0.4 + np.random.uniform(0, height * 0.3),
            "angle": np.random.uniform(0, 2 * np.pi),
            "length": height * np.random.uniform(0.2, 0.4),
            "radius": trunk_radius * np.random.uniform(0.3, 0.6)
        }
        branches.append(branch_data)
    
    # Foliage clusters
    foliage = []
    num_clusters = np.random.randint(5, 12)
    
    for i in range(num_clusters):
        foliage.append({
            "position": (
                np.random.uniform(-2, 2),
                height * 0.7 + np.random.uniform(-1, 2),
                np.random.uniform(-2, 2)
            ),
            "radius": np.random.uniform(0.8, 1.5)
        })
    
    return {
        "type": tree_type,
        "trunk": {"height": trunk_height, "radius": trunk_radius},
        "branches": branches,
        "foliage": foliage
    }


# =============================================================================
# 6. 工具函数
# =============================================================================

def create_grid_instances_data(
    rows: int = 5,
    cols: int = 5,
    spacing: float = 2.0,
    scale_range: Tuple[float, float] = (0.8, 1.2)
) -> List[ScatterPoint]:
    """
    Houdini: Copy and Transform
    
    Creates grid array with random variations.
    """
    points = []
    
    for row in range(rows):
        for col in range(cols):
            x = (col - cols // 2) * spacing
            z = (row - rows // 2) * spacing
            
            scale = np.random.uniform(*scale_range)
            angle = np.random.uniform(0, 2 * np.pi)
            
            point = ScatterPoint(
                position=(x, 0.0, z),
                scale=scale,
                rotation=(0.0, 1.0, 0.0, angle),
                attributes={"row": row, "col": col}
            )
            points.append(point)
    
    return points


def sample_height_from_function(
    x: float,
    z: float,
    height_func: Callable[[float, float], float]
) -> float:
    """Sample height from a height function."""
    return height_func(x, z)


def calculate_slope(
    x: float,
    z: float,
    height_func: Callable[[float, float], float],
    sample_distance: float = 0.1
) -> float:
    """
    Calculate terrain slope at given position.
    
    Returns:
        Slope angle in degrees
    """
    h_center = height_func(x, z)
    h_east = height_func(x + sample_distance, z)
    h_north = height_func(x, z + sample_distance)
    
    dx = (h_east - h_center) / sample_distance
    dz = (h_north - h_center) / sample_distance
    
    slope = np.sqrt(dx**2 + dz**2)
    return np.degrees(np.arctan(slope))


def extrude_building(
    base_points: List[Tuple[float, float]],
    height: float = 10.0,
    floors: int = 5
) -> Dict[str, Any]:
    """
    Houdini: PolyExtrude
    
    Extrude a building from a base footprint polygon.
    
    Args:
        base_points: List of (x, z) points defining the building footprint
        height: Total building height
        floors: Number of floors
        
    Returns:
        Dictionary with building geometry data
    """
    floor_height = height / floors
    
    # Generate wall faces
    walls = []
    for i in range(len(base_points)):
        p1 = base_points[i]
        p2 = base_points[(i + 1) % len(base_points)]
        
        wall = {
            "bottom_edge": (p1, p2),
            "height": height,
            "vertices": [
                (p1[0], 0, p1[1]),
                (p2[0], 0, p2[1]),
                (p2[0], height, p2[1]),
                (p1[0], height, p1[1])
            ]
        }
        walls.append(wall)
    
    # Generate roof
    roof = {
        "vertices": [(p[0], height, p[1]) for p in base_points],
        "height": height
    }
    
    # Generate floor slabs
    floor_slabs = []
    for f in range(floors + 1):
        y = f * floor_height
        floor_slabs.append({
            "level": f,
            "height": y,
            "vertices": [(p[0], y, p[1]) for p in base_points]
        })
    
    return {
        "footprint": base_points,
        "height": height,
        "floors": floors,
        "floor_height": floor_height,
        "walls": walls,
        "roof": roof,
        "floor_slabs": floor_slabs
    }


def create_cylinder_mesh(
    radius: float = 1.0,
    height: float = 2.0,
    segments: int = 16
) -> Dict[str, Any]:
    """
    Houdini: Labs Tool Helper
    
    Generate cylinder mesh data (used for tree trunks, columns, etc.)
    
    Args:
        radius: Cylinder radius
        height: Cylinder height
        segments: Number of radial segments
        
    Returns:
        Dictionary with cylinder mesh data
    """
    import numpy as np
    
    vertices = []
    indices = []
    
    # Bottom and top center vertices
    bottom_center = (0, 0, 0)
    top_center = (0, height, 0)
    vertices.extend([bottom_center, top_center])
    
    # Generate circle vertices
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        
        # Bottom ring
        vertices.append((x, 0, z))
        # Top ring
        vertices.append((x, height, z))
    
    # Generate faces
    # Bottom cap (triangles from center to ring)
    for i in range(segments):
        next_i = (i + 1) % segments
        indices.append((0, 2 + 2 * i, 2 + 2 * next_i))
    
    # Top cap
    for i in range(segments):
        next_i = (i + 1) % segments
        indices.append((1, 2 + 2 * next_i + 1, 2 + 2 * i + 1))
    
    # Side faces (quads as two triangles)
    for i in range(segments):
        next_i = (i + 1) % segments
        
        # Current segment indices
        curr_bottom = 2 + 2 * i
        curr_top = 2 + 2 * i + 1
        next_bottom = 2 + 2 * next_i
        next_top = 2 + 2 * next_i + 1
        
        # Two triangles per quad
        indices.append((curr_bottom, next_bottom, curr_top))
        indices.append((next_bottom, next_top, curr_top))
    
    return {
        "vertices": vertices,
        "indices": indices,
        "radius": radius,
        "height": height,
        "segments": segments
    }
