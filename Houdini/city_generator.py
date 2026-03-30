"""
City Generation System
======================

Houdini Equivalent Nodes:
- Labs Building Generator → BuildingComponent
- Labs City Street Generator → CityGenerator
- PolyExtrude → extrude_building_footprint
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class BuildingStyle(Enum):
    """Architectural styles for procedural buildings."""
    MODERN = "modern"
    CLASSIC = "classic"
    INDUSTRIAL = "industrial"
    RESIDENTIAL = "residential"
    SKYSCRAPER = "skyscraper"


@dataclass
class BuildingParameters:
    """Parameters for procedural building generation."""
    width: float = 10.0
    depth: float = 10.0
    height: float = 20.0
    floors: int = 5
    style: BuildingStyle = BuildingStyle.MODERN
    seed: int = 0
    
    # Detail parameters
    window_spacing: float = 2.0
    floor_height: float = 3.0
    setback: float = 0.0


@dataclass
class BuildingData:
    """Generated building data."""
    params: BuildingParameters
    footprint: List[Tuple[float, float]]
    floor_heights: List[float]
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CityBlock:
    """Represents a city block."""
    center: Tuple[float, float]
    size: float
    buildings: List[BuildingData] = field(default_factory=list)
    
    
class CityGenerator:
    """
    Houdini: Labs City Street Generator
    
    Generates city layouts with blocks and streets.
    """
    
    def __init__(
        self,
        block_size: float = 50.0,
        street_width: float = 10.0,
        seed: int = 0
    ):
        """
        Initialize city generator.
        
        Args:
            block_size: Size of each city block
            street_width: Width of streets between blocks
            seed: Random seed
        """
        self.block_size = block_size
        self.street_width = street_width
        self.seed = seed
        np.random.seed(seed)
        
        self.blocks: List[CityBlock] = []
        self.streets: List[Tuple[float, float, float, float]] = []
    
    def generate_grid(
        self,
        num_blocks_x: int = 5,
        num_blocks_z: int = 5
    ) -> Tuple[List[CityBlock], List[Tuple[float, float, float, float]]]:
        """
        Generate grid-based city layout.
        
        Args:
            num_blocks_x: Number of blocks in x direction
            num_blocks_z: Number of blocks in z direction
            
        Returns:
            Tuple of (blocks, streets) where streets are (x, z, width, depth)
        """
        self.blocks = []
        self.streets = []
        
        full_block = self.block_size + self.street_width
        offset_x = (num_blocks_x - 1) * full_block / 2
        offset_z = (num_blocks_z - 1) * full_block / 2
        
        for bx in range(num_blocks_x):
            for bz in range(num_blocks_z):
                # Calculate block center
                block_x = bx * full_block - offset_x
                block_z = bz * full_block - offset_z
                
                block = CityBlock(
                    center=(block_x, block_z),
                    size=self.block_size
                )
                self.blocks.append(block)
        
        # Generate streets
        total_width = num_blocks_x * self.block_size + (num_blocks_x + 1) * self.street_width
        total_depth = num_blocks_z * self.block_size + (num_blocks_z + 1) * self.street_width
        
        # Horizontal streets
        for i in range(num_blocks_z + 1):
            z = i * full_block - offset_z - self.street_width / 2 - self.block_size / 2
            self.streets.append((
                -total_width / 2,
                z,
                total_width,
                self.street_width
            ))
        
        # Vertical streets
        for i in range(num_blocks_x + 1):
            x = i * full_block - offset_x - self.street_width / 2 - self.block_size / 2
            self.streets.append((
                x,
                -total_depth / 2,
                self.street_width,
                total_depth
            ))
        
        return self.blocks, self.streets
    
    def generate_organic(
        self,
        num_main_roads: int = 3,
        city_radius: float = 200.0
    ) -> Tuple[List[CityBlock], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
        """
        Generate organic city layout with winding roads.
        
        Returns:
            Tuple of (blocks, road_segments) where road_segments are ((x1,z1), (x2,z2))
        """
        # Generate main roads as Bezier curves
        road_segments = []
        
        for _ in range(num_main_roads):
            # Random start and end points on circle
            angle1 = np.random.uniform(0, 2 * np.pi)
            angle2 = np.random.uniform(0, 2 * np.pi)
            
            p1 = (np.cos(angle1) * city_radius, np.sin(angle1) * city_radius)
            p2 = (np.cos(angle2) * city_radius, np.sin(angle2) * city_radius)
            
            # Control point for curve
            cp = (np.random.uniform(-city_radius/2, city_radius/2),
                  np.random.uniform(-city_radius/2, city_radius/2))
            
            # Sample points along Bezier curve
            num_points = 20
            for t in np.linspace(0, 1, num_points):
                # Quadratic Bezier
                x = (1-t)**2 * p1[0] + 2*(1-t)*t * cp[0] + t**2 * p2[0]
                z = (1-t)**2 * p1[1] + 2*(1-t)*t * cp[1] + t**2 * p2[1]
                
                if t > 0:
                    road_segments.append(((prev_x, prev_z), (x, z)))
                
                prev_x, prev_z = x, z
        
        return [], road_segments


class BuildingGenerator:
    """
    Houdini: Labs Building Generator + PolyExtrude
    
    Generates procedural buildings from parameters.
    """
    
    def __init__(self, seed: int = 0):
        self.seed = seed
        np.random.seed(seed)
    
    def create_footprint(
        self,
        width: float,
        depth: float,
        corner_radius: float = 0.0
    ) -> List[Tuple[float, float]]:
        """
        Create building footprint polygon.
        
        Args:
            width: Building width
            depth: Building depth
            corner_radius: Radius for rounded corners
            
        Returns:
            List of (x, z) vertices
        """
        if corner_radius <= 0:
            # Simple rectangle
            hw, hd = width / 2, depth / 2
            return [
                (-hw, -hd),
                (hw, -hd),
                (hw, hd),
                (-hw, hd)
            ]
        else:
            # Rounded rectangle
            hw, hd = width / 2 - corner_radius, depth / 2 - corner_radius
            points = []
            
            # Generate rounded corners
            for corner in range(4):
                base_angle = corner * np.pi / 2
                center_x = hw if corner in [0, 1] else -hw
                center_z = hd if corner in [0, 3] else -hd
                
                for i in range(5):
                    angle = base_angle + i * np.pi / 10
                    x = center_x + corner_radius * np.cos(angle)
                    z = center_z + corner_radius * np.sin(angle)
                    points.append((x, z))
            
            return points
    
    def generate_building(
        self,
        params: BuildingParameters
    ) -> BuildingData:
        """
        Generate building data from parameters.
        
        Args:
            params: Building parameters
            
        Returns:
            BuildingData with generated geometry info
        """
        np.random.seed(params.seed)
        
        # Create footprint
        if params.style == BuildingStyle.MODERN:
            footprint = self.create_footprint(params.width, params.depth, corner_radius=1.0)
        elif params.style == BuildingStyle.CLASSIC:
            footprint = self.create_footprint(params.width, params.depth, corner_radius=0.0)
        elif params.style == BuildingStyle.INDUSTRIAL:
            # Irregular shape
            footprint = self._create_industrial_footprint(params.width, params.depth)
        else:
            footprint = self.create_footprint(params.width, params.depth)
        
        # Calculate floor heights
        floor_heights = []
        current_height = 0.0
        
        for floor in range(params.floors):
            floor_height = params.floor_height
            
            # Vary floor heights by style
            if params.style == BuildingStyle.CLASSIC and floor == 0:
                floor_height *= 1.5  # Higher ground floor
            elif params.style == BuildingStyle.SKYSCRAPER:
                # Tapering height for skyscrapers
                floor_height *= (1.0 - 0.1 * floor / params.floors)
            
            current_height += floor_height
            floor_heights.append(current_height)
        
        # Additional attributes based on style
        attributes = {
            "roof_type": self._get_roof_type(params.style),
            "window_pattern": self._get_window_pattern(params.style),
            "has_setback": params.setback > 0,
            "total_height": current_height
        }
        
        return BuildingData(
            params=params,
            footprint=footprint,
            floor_heights=floor_heights,
            attributes=attributes
        )
    
    def _create_industrial_footprint(
        self,
        width: float,
        depth: float
    ) -> List[Tuple[float, float]]:
        """Create irregular industrial building footprint."""
        hw, hd = width / 2, depth / 2
        
        # L-shaped building
        points = [
            (-hw, -hd),
            (hw * 0.3, -hd),
            (hw * 0.3, -hd * 0.3),
            (hw, -hd * 0.3),
            (hw, hd),
            (-hw, hd)
        ]
        
        return points
    
    def _get_roof_type(self, style: BuildingStyle) -> str:
        """Get roof type for building style."""
        roof_types = {
            BuildingStyle.MODERN: "flat",
            BuildingStyle.CLASSIC: "pitched",
            BuildingStyle.INDUSTRIAL: "flat",
            BuildingStyle.RESIDENTIAL: "pitched",
            BuildingStyle.SKYSCRAPER: "mechanical"
        }
        return roof_types.get(style, "flat")
    
    def _get_window_pattern(self, style: BuildingStyle) -> str:
        """Get window pattern for building style."""
        patterns = {
            BuildingStyle.MODERN: "floor_to_ceiling",
            BuildingStyle.CLASSIC: "regular_grid",
            BuildingStyle.INDUSTRIAL: "large_industrial",
            BuildingStyle.RESIDENTIAL: "small_regular",
            BuildingStyle.SKYSCRAPER: "curtain_wall"
        }
        return patterns.get(style, "regular_grid")
    
    def populate_block(
        self,
        block: CityBlock,
        density: float = 0.7,
        height_range: Tuple[float, float] = (10.0, 50.0),
        style_distribution: Optional[Dict[BuildingStyle, float]] = None
    ) -> List[BuildingData]:
        """
        Populate a city block with buildings.
        
        Args:
            block: City block to populate
            density: Building density (0-1)
            height_range: (min, max) building height
            style_distribution: Style probability distribution
            
        Returns:
            List of generated buildings
        """
        if style_distribution is None:
            style_distribution = {
                BuildingStyle.MODERN: 0.3,
                BuildingStyle.CLASSIC: 0.2,
                BuildingStyle.RESIDENTIAL: 0.4,
                BuildingStyle.INDUSTRIAL: 0.1
            }
        
        buildings = []
        
        # Calculate available space
        margin = 2.0
        available_size = block.size - 2 * margin
        
        # Simple grid placement
        building_width = 15.0
        building_depth = 15.0
        
        num_x = int(available_size / building_width)
        num_z = int(available_size / building_depth)
        
        spacing_x = available_size / max(num_x, 1)
        spacing_z = available_size / max(num_z, 1)
        
        offset_x = block.center[0] - block.size / 2 + margin
        offset_z = block.center[1] - block.size / 2 + margin
        
        for ix in range(num_x):
            for iz in range(num_z):
                if np.random.random() > density:
                    continue
                
                # Select style
                style = np.random.choice(
                    list(style_distribution.keys()),
                    p=list(style_distribution.values())
                )
                
                # Generate parameters
                height = np.random.uniform(*height_range)
                floors = max(1, int(height / 3.0))
                
                params = BuildingParameters(
                    width=building_width - 2,
                    depth=building_depth - 2,
                    height=height,
                    floors=floors,
                    style=style,
                    seed=np.random.randint(0, 10000)
                )
                
                building = self.generate_building(params)
                
                # Offset footprint to block position
                x_pos = offset_x + ix * spacing_x + spacing_x / 2
                z_pos = offset_z + iz * spacing_z + spacing_z / 2
                
                building.footprint = [
                    (x + x_pos, z + z_pos) for x, z in building.footprint
                ]
                
                buildings.append(building)
        
        block.buildings = buildings
        return buildings


class StreetNetwork:
    """
    Houdini: Road network generation utilities.
    """
    
    @staticmethod
    def create_road_mesh(
        start: Tuple[float, float],
        end: Tuple[float, float],
        width: float,
        segments: int = 10
    ) -> Tuple[List[Tuple[float, float, float]], List[int]]:
        """
        Create road mesh vertices and indices.
        
        Args:
            start: Start position (x, z)
            end: End position (x, z)
            width: Road width
            segments: Number of segments
            
        Returns:
            Tuple of (vertices, indices)
        """
        vertices = []
        
        direction = np.array([end[0] - start[0], end[1] - start[1]])
        length = np.linalg.norm(direction)
        direction = direction / length
        
        # Perpendicular
        perp = np.array([-direction[1], direction[0]])
        half_width = width / 2
        
        for i in range(segments + 1):
            t = i / segments
            pos = np.array(start) * (1 - t) + np.array(end) * t
            
            # Left and right edges
            left = pos + perp * half_width
            right = pos - perp * half_width
            
            vertices.append((left[0], 0.1, left[1]))  # Slightly above ground
            vertices.append((right[0], 0.1, right[1]))
        
        # Generate indices
        indices = []
        for i in range(segments):
            base = i * 2
            indices.extend([base, base + 2, base + 1])
            indices.extend([base + 1, base + 2, base + 3])
        
        return vertices, indices
    
    @staticmethod
    def generate_sidewalks(
        streets: List[Tuple[float, float, float, float]],
        sidewalk_width: float = 1.5
    ) -> List[Tuple[float, float, float, float]]:
        """
        Generate sidewalks for streets.
        
        Args:
            streets: List of street rectangles (x, z, width, depth)
            sidewalk_width: Width of sidewalks
            
        Returns:
            List of sidewalk rectangles
        """
        sidewalks = []
        
        for x, z, width, depth in streets:
            # Expand by sidewalk width
            sidewalks.append((
                x - sidewalk_width,
                z - sidewalk_width,
                width + 2 * sidewalk_width,
                depth + 2 * sidewalk_width
            ))
        
        return sidewalks


def generate_complete_city(
    num_blocks_x: int = 8,
    num_blocks_z: int = 8,
    block_size: float = 40.0,
    street_width: float = 8.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate a complete city with buildings and streets.
    
    Args:
        num_blocks_x: Number of blocks in x direction
        num_blocks_z: Number of blocks in z direction
        block_size: Size of each block
        street_width: Width of streets
        seed: Random seed
        
    Returns:
        Dictionary with city data
    """
    np.random.seed(seed)
    
    # Generate city layout
    city_gen = CityGenerator(block_size, street_width, seed)
    blocks, streets = city_gen.generate_grid(num_blocks_x, num_blocks_z)
    
    # Generate buildings
    building_gen = BuildingGenerator(seed)
    all_buildings = []
    
    for block in blocks:
        # Vary parameters by block position
        height_range = (
            np.random.uniform(8, 15),
            np.random.uniform(30, 80)
        )
        
        buildings = building_gen.populate_block(
            block,
            density=np.random.uniform(0.5, 0.8),
            height_range=height_range
        )
        all_buildings.extend(buildings)
    
    return {
        "blocks": blocks,
        "streets": streets,
        "buildings": all_buildings,
        "statistics": {
            "num_blocks": len(blocks),
            "num_streets": len(streets),
            "num_buildings": len(all_buildings),
            "city_size": (
                num_blocks_x * (block_size + street_width),
                num_blocks_z * (block_size + street_width)
            )
        }
    }
