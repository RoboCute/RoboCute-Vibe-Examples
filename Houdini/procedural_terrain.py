"""
Procedural Terrain Generation - HeightField System
==================================================

Houdini Equivalent Nodes:
- HeightField → HeightFieldTerrain
- HeightField Erode → thermal_erosion
- HeightField Scatter → scatter_on_terrain
- HeightField Mask by Feature → get_slope_at, get_height_at
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class TerrainScatterResult:
    """Result of terrain scatter operation."""
    points: List[Tuple[float, float, float]]
    placed_count: int
    attempts: int


class HeightFieldTerrain:
    """
    Houdini: HeightField system
    
    Procedural terrain generation with erosion and feature-based scattering.
    """
    
    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        cell_size: float = 1.0,
        seed: int = 0
    ):
        """
        Initialize HeightField terrain.
        
        Args:
            width: Number of cells in x direction
            height: Number of cells in z direction
            cell_size: Size of each cell in world units
            seed: Random seed for noise generation
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.seed = seed
        self.height_map = np.zeros((height, width), dtype=np.float32)
        self.sediment_map = np.zeros((height, width), dtype=np.float32)
        
        np.random.seed(seed)
    
    # =========================================================================
    # 基础地形生成
    # =========================================================================
    
    def generate_noise_terrain(
        self,
        scale: float = 50.0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0
    ) -> None:
        """
        Generate base terrain using layered noise.
        
        Args:
            scale: Noise coordinate scale
            octaves: Number of noise layers
            persistence: Amplitude decay per octave
            lacunarity: Frequency increase per octave
        """
        for y in range(self.height):
            for x in range(self.width):
                nx = x / scale
                ny = y / scale
                
                amplitude = 1.0
                frequency = 1.0
                value = 0.0
                max_value = 0.0
                
                for _ in range(octaves):
                    # Simple sine-based noise for demonstration
                    # In production, use proper Perlin/Simplex noise
                    value += amplitude * (
                        np.sin(nx * frequency * np.pi * 2) * 
                        np.cos(ny * frequency * np.pi * 2)
                    )
                    max_value += amplitude
                    amplitude *= persistence
                    frequency *= lacunarity
                
                # Normalize and scale
                self.height_map[y, x] = (value / max_value) * 20.0 + 10.0
    
    def generate_fractal_terrain(
        self,
        roughness: float = 0.5,
        initial_height: float = 100.0
    ) -> None:
        """
        Generate terrain using diamond-square algorithm.
        
        Args:
            roughness: Height variation factor
            initial_height: Initial corner heights
        """
        size = max(self.width, self.height)
        # Round up to power of 2 + 1
        power = int(np.ceil(np.log2(size))) + 1
        grid_size = 2 ** power + 1
        
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Initialize corners
        grid[0, 0] = np.random.uniform(0, initial_height)
        grid[0, -1] = np.random.uniform(0, initial_height)
        grid[-1, 0] = np.random.uniform(0, initial_height)
        grid[-1, -1] = np.random.uniform(0, initial_height)
        
        step = grid_size - 1
        scale = initial_height
        
        while step > 1:
            half_step = step // 2
            
            # Diamond step
            for y in range(half_step, grid_size, step):
                for x in range(half_step, grid_size, step):
                    avg = (
                        grid[y - half_step, x - half_step] +
                        grid[y - half_step, x + half_step] +
                        grid[y + half_step, x - half_step] +
                        grid[y + half_step, x + half_step]
                    ) / 4.0
                    grid[y, x] = avg + np.random.uniform(-scale, scale)
            
            # Square step
            for y in range(0, grid_size, half_step):
                for x in range((y + half_step) % step, grid_size, step):
                    count = 0
                    avg = 0.0
                    
                    if y >= half_step:
                        avg += grid[y - half_step, x]
                        count += 1
                    if y + half_step < grid_size:
                        avg += grid[y + half_step, x]
                        count += 1
                    if x >= half_step:
                        avg += grid[y, x - half_step]
                        count += 1
                    if x + half_step < grid_size:
                        avg += grid[y, x + half_step]
                        count += 1
                    
                    grid[y, x] = avg / count + np.random.uniform(-scale, scale)
            
            step = half_step
            scale *= roughness
        
        # Copy to heightmap
        self.height_map = grid[:self.height, :self.width].copy()
    
    # =========================================================================
    # 侵蚀模拟
    # =========================================================================
    
    def thermal_erosion(
        self,
        iterations: int = 50,
        talus_angle: float = 45.0,
        fraction: float = 0.5
    ) -> None:
        """
        Houdini: HeightField Erode (Thermal)
        
        Simulates thermal erosion - material falls down steep slopes.
        
        Args:
            iterations: Number of erosion iterations
            talus_angle: Critical angle in degrees
            fraction: Amount of material to move per iteration
        """
        talus = np.tan(np.radians(talus_angle))
        
        for _ in range(iterations):
            new_height = self.height_map.copy()
            
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    h = self.height_map[y, x]
                    max_diff = 0.0
                    max_x, max_y = x, y
                    
                    # Check 8 neighbors
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            
                            neighbor_h = self.height_map[y + dy, x + dx]
                            # Account for diagonal distance
                            dist = self.cell_size * np.sqrt(dx*dx + dy*dy)
                            slope = (h - neighbor_h) / dist
                            
                            if slope > max_diff:
                                max_diff = slope
                                max_x, max_y = x + dx, y + dy
                    
                    # If slope exceeds talus angle, move material
                    if max_diff > talus:
                        dist = np.sqrt((max_x - x)**2 + (max_y - y)**2) * self.cell_size
                        amount = (max_diff - talus) * dist * fraction
                        new_height[y, x] -= amount
                        new_height[max_y, max_x] += amount
            
            self.height_map = new_height
    
    def hydraulic_erosion(
        self,
        iterations: int = 100,
        rain_rate: float = 0.01,
        evaporation: float = 0.1,
        capacity: float = 0.1
    ) -> None:
        """
        Houdini: HeightField Erode (Hydraulic)
        
        Simulates water-based erosion.
        
        Args:
            iterations: Number of simulation steps
            rain_rate: Amount of rain per iteration
            evaporation: Water evaporation rate
            capacity: Sediment transport capacity
        """
        water_map = np.zeros_like(self.height_map)
        sediment_map = np.zeros_like(self.height_map)
        
        for _ in range(iterations):
            # Add rain
            water_map += np.random.uniform(0, rain_rate, water_map.shape)
            
            # Erode and transport
            new_height = self.height_map.copy()
            
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if water_map[y, x] <= 0:
                        continue
                    
                    # Find lowest neighbor
                    h = self.height_map[y, x] + water_map[y, x]
                    min_h = h
                    min_x, min_y = x, y
                    
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            neighbor_h = self.height_map[y + dy, x + dx] + water_map[y + dy, x + dx]
                            if neighbor_h < min_h:
                                min_h = neighbor_h
                                min_x, min_y = x + dx, y + dy
                    
                    if min_x != x or min_y != y:
                        # Move water and sediment
                        transfer = min(water_map[y, x] * 0.5, h - min_h)
                        sediment_transfer = transfer * capacity
                        
                        water_map[y, x] -= transfer
                        water_map[min_y, min_x] += transfer * (1 - evaporation)
                        
                        new_height[y, x] -= sediment_transfer
                        sediment_map[min_y, min_x] += sediment_transfer
            
            self.height_map = new_height
            water_map *= (1 - evaporation)
    
    def smooth(self, iterations: int = 1, strength: float = 0.5) -> None:
        """
        Houdini: HeightField Smooth
        
        Smooth the terrain heightmap.
        
        Args:
            iterations: Number of smoothing passes
            strength: Smoothing strength (0-1)
        """
        kernel = np.array([
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]
        ])
        
        for _ in range(iterations):
            from scipy.ndimage import convolve
            smoothed = convolve(self.height_map, kernel, mode='nearest')
            self.height_map = self.height_map * (1 - strength) + smoothed * strength
    
    # =========================================================================
    # 查询函数
    # =========================================================================
    
    def get_height_at(self, x: float, z: float) -> float:
        """
        Get height at world position.
        
        Args:
            x: World x coordinate
            z: World z coordinate
            
        Returns:
            Height value at position
        """
        ix = int(x / self.cell_size)
        iz = int(z / self.cell_size)
        
        # Clamp to bounds
        ix = max(0, min(ix, self.width - 1))
        iz = max(0, min(iz, self.height - 1))
        
        return float(self.height_map[iz, ix])
    
    def get_slope_at(self, x: float, z: float) -> float:
        """
        Get slope angle at world position.
        
        Args:
            x: World x coordinate
            z: World z coordinate
            
        Returns:
            Slope angle in degrees
        """
        ix = int(x / self.cell_size)
        iz = int(z / self.cell_size)
        
        # Clamp to valid area for gradient calculation
        ix = max(1, min(ix, self.width - 2))
        iz = max(1, min(iz, self.height - 2))
        
        # Calculate gradient using central differences
        dx = (self.height_map[iz, ix + 1] - self.height_map[iz, ix - 1]) / (2 * self.cell_size)
        dz = (self.height_map[iz + 1, ix] - self.height_map[iz - 1, ix]) / (2 * self.cell_size)
        
        slope = np.sqrt(dx**2 + dz**2)
        return float(np.degrees(np.arctan(slope)))
    
    def get_normal_at(self, x: float, z: float) -> Tuple[float, float, float]:
        """
        Get surface normal at world position.
        
        Args:
            x: World x coordinate
            z: World z coordinate
            
        Returns:
            Normal vector (nx, ny, nz)
        """
        ix = int(x / self.cell_size)
        iz = int(z / self.cell_size)
        
        ix = max(1, min(ix, self.width - 2))
        iz = max(1, min(iz, self.height - 2))
        
        dx = (self.height_map[iz, ix + 1] - self.height_map[iz, ix - 1]) / (2 * self.cell_size)
        dz = (self.height_map[iz + 1, ix] - self.height_map[iz - 1, ix]) / (2 * self.cell_size)
        
        # Normal is perpendicular to gradient
        normal = np.array([-dx, 1.0, -dz])
        normal = normal / np.linalg.norm(normal)
        
        return (float(normal[0]), float(normal[1]), float(normal[2]))
    
    # =========================================================================
    # 散布功能
    # =========================================================================
    
    def scatter_on_terrain(
        self,
        count: int = 100,
        min_slope: float = 0.0,
        max_slope: float = 90.0,
        min_height: float = -float('inf'),
        max_height: float = float('inf'),
        seed: int = 0
    ) -> TerrainScatterResult:
        """
        Houdini: HeightField Scatter
        
        Scatter points on terrain based on constraints.
        
        Args:
            count: Number of points to place
            min_slope: Minimum slope angle (degrees)
            max_slope: Maximum slope angle (degrees)
            min_height: Minimum height constraint
            max_height: Maximum height constraint
            seed: Random seed
            
        Returns:
            TerrainScatterResult with placed points and statistics
        """
        np.random.seed(seed)
        
        points = []
        placed = 0
        attempts = 0
        max_attempts = count * 20
        
        world_width = self.width * self.cell_size
        world_height = self.height * self.cell_size
        
        while placed < count and attempts < max_attempts:
            x = np.random.uniform(0, world_width)
            z = np.random.uniform(0, world_height)
            
            h = self.get_height_at(x, z)
            slope = self.get_slope_at(x, z)
            
            # Check constraints
            if (min_height <= h <= max_height and
                min_slope <= slope <= max_slope):
                
                # Random rotation
                angle = np.random.uniform(0, 2 * np.pi)
                # Random scale
                scale = np.random.uniform(0.8, 1.2)
                
                points.append((x, h, z))
                placed += 1
            
            attempts += 1
        
        return TerrainScatterResult(
            points=points,
            placed_count=placed,
            attempts=attempts
        )
    
    def create_mask_by_feature(
        self,
        feature: str = "slope",
        min_value: float = 0.0,
        max_value: float = 90.0
    ) -> np.ndarray:
        """
        Houdini: HeightField Mask by Feature
        
        Create a mask based on terrain features.
        
        Args:
            feature: Feature type ("slope", "height", "curvature")
            min_value: Minimum feature value
            max_value: Maximum feature value
            
        Returns:
            2D mask array (0-1 values)
        """
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        for y in range(self.height):
            for x in range(self.width):
                wx = x * self.cell_size
                wz = y * self.cell_size
                
                if feature == "slope":
                    value = self.get_slope_at(wx, wz)
                elif feature == "height":
                    value = self.get_height_at(wx, wz)
                elif feature == "curvature":
                    value = self._get_curvature_at(x, y)
                else:
                    value = 0.0
                
                if min_value <= value <= max_value:
                    mask[y, x] = 1.0
                else:
                    # Smooth falloff at boundaries
                    range_size = max_value - min_value
                    if value < min_value:
                        mask[y, x] = max(0.0, 1.0 - (min_value - value) / (range_size * 0.1))
                    else:
                        mask[y, x] = max(0.0, 1.0 - (value - max_value) / (range_size * 0.1))
        
        return mask
    
    def _get_curvature_at(self, x: int, y: int) -> float:
        """Calculate surface curvature at grid position."""
        if x < 1 or x >= self.width - 1 or y < 1 or y >= self.height - 1:
            return 0.0
        
        # Laplacian
        center = self.height_map[y, x]
        neighbors = (
            self.height_map[y-1, x] +
            self.height_map[y+1, x] +
            self.height_map[y, x-1] +
            self.height_map[y, x+1]
        )
        
        return abs(neighbors - 4 * center) / self.cell_size
    
    # =========================================================================
    # 导出功能
    # =========================================================================
    
    def to_mesh_data(
        self,
        max_error: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert heightfield to mesh data.
        
        Returns:
            Tuple of (vertices, indices) arrays
        """
        vertices = []
        indices = []
        
        for y in range(self.height):
            for x in range(self.width):
                vx = x * self.cell_size
                vz = y * self.cell_size
                vy = self.height_map[y, x]
                vertices.append([vx, vy, vz])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Generate triangles
        for y in range(self.height - 1):
            for x in range(self.width - 1):
                # Current quad vertices
                i0 = y * self.width + x
                i1 = y * self.width + (x + 1)
                i2 = (y + 1) * self.width + x
                i3 = (y + 1) * self.width + (x + 1)
                
                # Two triangles per quad
                indices.extend([i0, i2, i1])
                indices.extend([i1, i2, i3])
        
        indices = np.array(indices, dtype=np.uint32)
        
        return vertices, indices
    
    def export_heightmap(self, filepath: str) -> None:
        """Export heightmap as image."""
        import imageio
        
        # Normalize to 0-255
        h_min = self.height_map.min()
        h_max = self.height_map.max()
        normalized = ((self.height_map - h_min) / (h_max - h_min) * 255).astype(np.uint8)
        
        imageio.imwrite(filepath, normalized)
    
    def get_statistics(self) -> dict:
        """Get terrain statistics."""
        return {
            "min_height": float(self.height_map.min()),
            "max_height": float(self.height_map.max()),
            "mean_height": float(self.height_map.mean()),
            "std_height": float(self.height_map.std()),
            "dimensions": (self.width, self.height),
            "cell_size": self.cell_size,
            "world_size": (self.width * self.cell_size, self.height * self.cell_size)
        }
