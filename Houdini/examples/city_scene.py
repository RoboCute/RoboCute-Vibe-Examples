"""
City Scene Generation Example
=============================

Houdini Node Graph:
-------------------
Grid (基础网格) → Connectivity/Partition (分块) → 
For-Each Piece (循环处理) → Copy to Points (放置建筑) →
PolyExtrude (挤出建筑) → Merge (合并)

This example demonstrates:
- Grid-based city layout generation
- Building component system with variants
- Street network generation
- Procedural building generation with styles
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from city_generator import (
    CityGenerator, BuildingGenerator, BuildingParameters,
    BuildingStyle, CityBlock, StreetNetwork,
    generate_complete_city
)
from houdini_nodes import (
    ComponentBuilder, VariantSet, foreach_batch_process,
    hexagonal_packing, rectangular_packing
)


def generate_city_scene(
    num_blocks_x: int = 6,
    num_blocks_z: int = 6,
    block_size: float = 40.0,
    street_width: float = 8.0,
    seed: int = 123
) -> dict:
    """
    Generate a complete city scene.
    
    Houdini Workflow:
    1. Grid - Create base grid
    2. Connectivity/Partition - Divide into blocks
    3. For-Each Piece - Process each block
    4. Copy to Points - Place buildings
    5. PolyExtrude - Generate building geometry
    6. Merge - Combine all geometry
    
    Args:
        num_blocks_x: Number of city blocks in x direction
        num_blocks_z: Number of city blocks in z direction
        block_size: Size of each block
        street_width: Width of streets
        seed: Random seed
        
    Returns:
        Dictionary with city scene data
    """
    print("=" * 60)
    print("City Scene Generation")
    print("=" * 60)
    
    np.random.seed(seed)
    
    # =========================================================================
    # 1. Grid + Connectivity/Partition (创建城市网格)
    # =========================================================================
    print("\n[1/5] Creating city grid layout...")
    
    city_gen = CityGenerator(
        block_size=block_size,
        street_width=street_width,
        seed=seed
    )
    
    blocks, streets = city_gen.generate_grid(
        num_blocks_x=num_blocks_x,
        num_blocks_z=num_blocks_z
    )
    
    total_city_width = num_blocks_x * (block_size + street_width)
    total_city_depth = num_blocks_z * (block_size + street_width)
    
    print(f"  City size: {total_city_width:.1f} x {total_city_depth:.1f}")
    print(f"  Blocks: {len(blocks)}")
    print(f"  Streets: {len(streets)}")
    
    # =========================================================================
    # 2. For-Each Piece + Copy to Points (处理每个街区)
    # =========================================================================
    print("\n[2/5] Populating blocks with buildings...")
    
    building_gen = BuildingGenerator(seed)
    
    # Define different districts with different characteristics
    districts = {
        "downtown": {
            "center": (0, 0),
            "radius": total_city_width * 0.3,
            "height_range": (40, 100),
            "density": 0.9,
            "styles": {
                BuildingStyle.SKYSCRAPER: 0.4,
                BuildingStyle.MODERN: 0.4,
                BuildingStyle.CLASSIC: 0.2
            }
        },
        "residential": {
            "center": (total_city_width * 0.3, total_city_depth * 0.3),
            "radius": total_city_width * 0.4,
            "height_range": (10, 25),
            "density": 0.6,
            "styles": {
                BuildingStyle.RESIDENTIAL: 0.7,
                BuildingStyle.CLASSIC: 0.3
            }
        },
        "industrial": {
            "center": (-total_city_width * 0.3, -total_city_depth * 0.3),
            "radius": total_city_width * 0.25,
            "height_range": (15, 30),
            "density": 0.7,
            "styles": {
                BuildingStyle.INDUSTRIAL: 0.8,
                BuildingStyle.MODERN: 0.2
            }
        }
    }
    
    def process_block(block: CityBlock, index: int) -> List[Dict]:
        """Process a single city block (For-Each loop body)."""
        
        # Determine district based on distance to district centers
        block_buildings = []
        
        for district_name, district in districts.items():
            dist = np.sqrt(
                (block.center[0] - district["center"][0])**2 +
                (block.center[1] - district["center"][1])**2
            )
            
            if dist < district["radius"]:
                # Generate buildings for this district
                buildings = building_gen.populate_block(
                    block,
                    density=district["density"],
                    height_range=district["height_range"],
                    style_distribution=district["styles"]
                )
                
                for b in buildings:
                    b.attributes["district"] = district_name
                
                block_buildings.extend(buildings)
                break
        else:
            # Default mixed district
            buildings = building_gen.populate_block(
                block,
                density=0.5,
                height_range=(15, 40),
                style_distribution={
                    BuildingStyle.MODERN: 0.3,
                    BuildingStyle.RESIDENTIAL: 0.4,
                    BuildingStyle.CLASSIC: 0.3
                }
            )
            
            for b in buildings:
                b.attributes["district"] = "mixed"
            
            block_buildings.extend(buildings)
        
        return block_buildings
    
    # Process all blocks
    all_buildings = foreach_batch_process(blocks, process_block)
    # Flatten list
    all_buildings = [b for sublist in all_buildings for b in sublist]
    
    print(f"  Generated {len(all_buildings)} buildings")
    
    # Count by district
    district_counts = {}
    for b in all_buildings:
        d = b.attributes.get("district", "unknown")
        district_counts[d] = district_counts.get(d, 0) + 1
    
    print("  Buildings by district:")
    for district, count in sorted(district_counts.items()):
        print(f"    {district}: {count}")
    
    # =========================================================================
    # 3. PolyExtrude (生成建筑几何体数据)
    # =========================================================================
    print("\n[3/5] Generating building geometry...")
    
    building_meshes = []
    
    for building in all_buildings:
        mesh_data = generate_building_mesh(building)
        building_meshes.append({
            "building": building,
            "mesh": mesh_data
        })
    
    print(f"  Generated {len(building_meshes)} building meshes")
    
    # =========================================================================
    # 4. Component Builder (创建可复用建筑组件)
    # =========================================================================
    print("\n[4/5] Creating building component library...")
    
    # Create component builders for each style
    components = {}
    
    for style in BuildingStyle:
        comp = ComponentBuilder(f"Building_{style.value}")
        comp.add_parameter("width", 15.0, 5.0, 30.0)
        comp.add_parameter("depth", 15.0, 5.0, 30.0)
        comp.add_parameter("height", 20.0, 5.0, 100.0)
        comp.add_parameter("floors", 5, 1, 30)
        comp.add_parameter("seed", 0)
        
        def make_building_data(width, depth, height, floors, seed, s=style):
            params = BuildingParameters(
                width=width,
                depth=depth,
                height=height,
                floors=floors,
                style=s,
                seed=seed
            )
            return building_gen.generate_building(params)
        
        comp.set_build_function(make_building_data)
        components[style] = comp
    
    # Generate variants using components
    variants = []
    for style, comp in components.items():
        for i in range(3):  # 3 variants per style
            variant = comp.build(seed=i * 100)
            variants.append({
                "style": style.value,
                "data": variant
            })
    
    print(f"  Created {len(variants)} building variants")
    
    # =========================================================================
    # 5. Street Network + Merge (街道网络)
    # =========================================================================
    print("\n[5/5] Generating street network...")
    
    street_network = StreetNetwork()
    
    # Generate sidewalk data
    sidewalks = street_network.generate_sidewalks(streets, sidewalk_width=1.5)
    
    # Create road meshes
    road_meshes = []
    for street in streets:
        x, z, width, depth = street
        
        # Create simple road mesh (just corners for now)
        corners = [
            (x, 0.1, z),
            (x + width, 0.1, z),
            (x + width, 0.1, z + depth),
            (x, 0.1, z + depth)
        ]
        
        road_meshes.append({
            "type": "road",
            "bounds": street,
            "corners": corners
        })
    
    print(f"  Generated {len(road_meshes)} road segments")
    print(f"  Generated {len(sidewalks)} sidewalks")
    
    # =========================================================================
    # Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("City Generation Complete!")
    print("=" * 60)
    
    scene_data = {
        "blocks": blocks,
        "streets": streets,
        "buildings": all_buildings,
        "building_meshes": building_meshes,
        "road_meshes": road_meshes,
        "sidewalks": sidewalks,
        "components": components,
        "variants": variants,
        "statistics": {
            "num_blocks": len(blocks),
            "num_streets": len(streets),
            "num_buildings": len(all_buildings),
            "city_size": (total_city_width, total_city_depth),
            "districts": district_counts
        }
    }
    
    return scene_data


def generate_building_mesh(building) -> Dict[str, Any]:
    """
    Generate mesh data for a building (PolyExtrude equivalent).
    
    Args:
        building: BuildingData object
        
    Returns:
        Dictionary with mesh data
    """
    params = building.params
    footprint = building.footprint
    floor_heights = building.floor_heights
    
    vertices = []
    indices = []
    normals = []
    uvs = []
    
    # Ground floor vertices
    base_vertices = [(p[0], 0, p[1]) for p in footprint]
    
    # Generate walls for each floor
    for floor_idx, floor_height in enumerate(floor_heights):
        prev_height = floor_heights[floor_idx - 1] if floor_idx > 0 else 0
        
        # Create wall segments for each edge of footprint
        for i in range(len(footprint)):
            p1 = footprint[i]
            p2 = footprint[(i + 1) % len(footprint)]
            
            # Wall quad vertices (bottom and top)
            v_bottom_1 = (p1[0], prev_height, p1[1])
            v_bottom_2 = (p2[0], prev_height, p2[1])
            v_top_1 = (p1[0], floor_height, p1[1])
            v_top_2 = (p2[0], floor_height, p2[1])
            
            base_idx = len(vertices)
            vertices.extend([v_bottom_1, v_bottom_2, v_top_1, v_top_2])
            
            # Two triangles per wall quad
            indices.extend([base_idx, base_idx + 2, base_idx + 1])
            indices.extend([base_idx + 1, base_idx + 2, base_idx + 3])
            
            # Calculate normal
            edge = np.array([p2[0] - p1[0], 0, p2[1] - p1[1]])
            up = np.array([0, 1, 0])
            normal = np.cross(edge, up)
            normal = normal / np.linalg.norm(normal)
            
            for _ in range(4):
                normals.append((normal[0], normal[1], normal[2]))
            
            # Simple UVs
            uvs.extend([(0, 0), (1, 0), (0, 1), (1, 1)])
    
    # Roof
    if footprint:
        roof_height = floor_heights[-1] if floor_heights else params.height
        center = np.mean(footprint, axis=0)
        
        if params.style == BuildingStyle.CLASSIC:
            # Pitched roof
            roof_peak = (center[0], roof_height + params.height * 0.3, center[1])
            
            for i in range(len(footprint)):
                p = footprint[i]
                p_next = footprint[(i + 1) % len(footprint)]
                
                base_idx = len(vertices)
                vertices.extend([
                    (p[0], roof_height, p[1]),
                    (p_next[0], roof_height, p_next[1]),
                    roof_peak
                ])
                
                indices.extend([base_idx, base_idx + 1, base_idx + 2])
        else:
            # Flat roof
            roof_vertices = [(p[0], roof_height, p[1]) for p in footprint]
            
            # Triangulate roof (simple fan triangulation)
            base_idx = len(vertices)
            vertices.extend(roof_vertices)
            
            for i in range(1, len(footprint) - 1):
                indices.extend([base_idx, base_idx + i, base_idx + i + 1])
    
    return {
        "vertices": vertices,
        "indices": indices,
        "normals": normals,
        "uvs": uvs,
        "style": params.style.value,
        "floors": params.floors,
        "height": params.height
    }


def export_city_data(scene_data: dict, output_dir: str = "./output"):
    """
    Export city data to files.
    
    Args:
        scene_data: City scene data
        output_dir: Output directory
    """
    import json
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Export statistics
    with open(f"{output_dir}/city_stats.json", "w") as f:
        json.dump(scene_data["statistics"], f, indent=2)
    
    # Export building data
    buildings_data = []
    for b in scene_data["buildings"]:
        buildings_data.append({
            "style": b.params.style.value,
            "width": b.params.width,
            "depth": b.params.depth,
            "height": b.params.height,
            "floors": b.params.floors,
            "district": b.attributes.get("district", "unknown"),
            "position": (
                sum(p[0] for p in b.footprint) / len(b.footprint),
                sum(p[1] for p in b.footprint) / len(b.footprint)
            ) if b.footprint else (0, 0)
        })
    
    with open(f"{output_dir}/city_buildings.json", "w") as f:
        json.dump(buildings_data, f, indent=2)
    
    print(f"\nCity data exported to {output_dir}/")
    print(f"  - city_stats.json")
    print(f"  - city_buildings.json")


def main():
    """Main execution."""
    # Generate city
    city = generate_city_scene(
        num_blocks_x=6,
        num_blocks_z=6,
        block_size=40.0,
        street_width=8.0,
        seed=123
    )
    
    # Print summary
    print("\nCity Summary:")
    print(f"  Total buildings: {city['statistics']['num_buildings']}")
    print(f"  City size: {city['statistics']['city_size'][0]:.1f} x {city['statistics']['city_size'][1]:.1f}")
    print(f"  Districts: {list(city['statistics']['districts'].keys())}")
    
    # Export
    export_city_data(city)
    
    return city


if __name__ == "__main__":
    main()
