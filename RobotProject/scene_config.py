#!/usr/bin/env python3
"""
Scene and App Configuration using JSON

This example demonstrates how to use JSON as project and App configuration
in RoboCute. It shows:

1. Loading App configuration from JSON (display, backend, camera settings)
2. Loading Project configuration from JSON (assets, scenes, render settings)
3. Creating and serializing scene configurations (entities, components, materials)
4. Saving and loading complete scene configurations

Usage:
    python scene_config.py --config app_config.json --project ./my_project
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

import robocute as rbc
import robocute.rbc_ext as re
import robocute.rbc_ext.luisa as lc
import samples.mat_builtin as mat


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class DisplayConfig:
    """Display/window configuration."""
    width: int = 1920
    height: int = 1080
    title: str = "RoboCute Scene"
    resizable: bool = True
    vsync: bool = True
    
    def to_resolution(self) -> lc.uint2:
        return lc.uint2(self.width, self.height)


@dataclass
class BackendConfig:
    """Graphics backend configuration."""
    name: str = "dx"  # dx, vk, cuda
    shader_path: Optional[str] = None
    

@dataclass
class CameraConfig:
    """Camera configuration for display camera."""
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, -3.0])
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    fov: float = 60.0
    near_plane: float = 0.01
    far_plane: float = 1000.0
    aperture: float = 0.0
    focus_distance: float = 10.0
    
    def apply_to(self, camera: re.world.CameraComponent):
        """Apply config to a camera component."""
        camera.set_fov(self.fov)
        camera.set_near_plane(self.near_plane)
        camera.set_far_plane(self.far_plane)
        camera.set_aperture(self.aperture)
        camera.set_focus_distance(self.focus_distance)


@dataclass
class RenderSettingsConfig:
    """Render settings configuration - maps to RenderSettings."""
    # Exposure and tonemapping
    global_exposure: float = 0.0
    gamma: float = 2.2
    use_auto_exposure: bool = False
    
    # Sky/Sun settings
    sky_color: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7])
    sky_max_lum: float = 1.0
    sky_angle: float = 0.0
    sun_color: List[float] = field(default_factory=lambda: [1.0, 0.98, 0.95])
    sun_intensity: float = 1.0
    sun_dir: List[float] = field(default_factory=lambda: [0.0, -1.0, 0.0])
    sun_angle: float = 0.5
    
    # Path tracing settings
    offline_spp: int = 128
    offline_origin_bounce: int = 4
    offline_indirect_bounce: int = 4
    
    # Post-processing
    denoise: bool = False
    use_hdr_display: bool = False
    use_hdr_10: bool = False
    max_luminance: float = 1000.0
    min_luminance: float = 0.01
    
    def apply_to(self, settings: re.world.RenderSettings):
        """Apply config to RenderSettings object."""
        settings.set_global_exposure(self.global_exposure)
        settings.set_gamma(self.gamma)
        settings.set_use_auto_exposure(self.use_auto_exposure)
        settings.set_sky_color(lc.float3(*self.sky_color))
        settings.set_sky_max_lum(self.sky_max_lum)
        settings.set_sky_angle(self.sky_angle)
        settings.set_sun_color(lc.float3(*self.sun_color))
        settings.set_sun_intensity(self.sun_intensity)
        settings.set_sun_dir(lc.float3(*self.sun_dir))
        settings.set_sun_angle(self.sun_angle)
        settings.set_offline_spp(self.offline_spp)
        settings.set_offline_origin_bounce(self.offline_origin_bounce)
        settings.set_offline_indirect_bounce(self.offline_indirect_bounce)
        settings.set_denoise(self.denoise)
        settings.set_use_hdr_display(self.use_hdr_display)
        settings.set_use_hdr_10(self.use_hdr_10)
        settings.set_max_luminance(self.max_luminance)
        settings.set_min_luminance(self.min_luminance)
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "RenderSettingsConfig":
        return cls(**json.loads(json_str))


@dataclass
class GroundPlaneConfig:
    """Ground plane configuration."""
    enabled: bool = True
    mode: str = "grid"  # none, grid, solid
    scale: float = 100.0
    height: float = 0.0
    roughness: float = 0.5
    metallic: float = 0.3
    albedo: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8])


@dataclass
class MaterialConfig:
    """Material configuration for creating materials via JSON."""
    name: str
    type: str = "pbr"
    # Base properties
    base_albedo: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    base_albedo_tex: Optional[str] = None
    
    # Metallic/Roughness
    metallic: float = 0.0
    roughness: float = 0.3
    
    # Emission
    emission: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    # Other properties
    transmission: float = 0.0
    ior: float = 1.5
    
    def create_material(self, project: re.world.Project) -> re.world.MaterialResource:
        """Create a MaterialResource from this config."""
        mat_interface = mat.OpenPBRInterface(project)
        mat_interface.set_base_albedo(tuple(self.base_albedo))
        mat_interface.set_weight_metallic(self.metallic)
        mat_interface.set_specular_roughness(self.roughness)
        mat_interface.set_emission_luminance(tuple(self.emission))
        mat_interface.set_weight_transmission(self.transmission)
        mat_interface.set_specular_ior(self.ior)
        
        # Load texture if specified
        if self.base_albedo_tex:
            tex = project.import_texture(self.base_albedo_tex, 4, True)
            if tex:
                mat_interface.set_base_albedo_tex(tex)
        
        material = re.world.MaterialResource()
        material.load_from_json(mat_interface.dump_to_json())
        return material


@dataclass
class TransformConfig:
    """Transform component configuration."""
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    
    def apply_to(self, transform: re.world.TransformComponent, recursive: bool = False):
        """Apply config to TransformComponent."""
        transform.set_pos(lc.double3(*self.position), recursive)
        transform.set_rotation(lc.float4(*self.rotation), recursive)
        transform.set_scale(lc.double3(*self.scale), recursive)


@dataclass
class LightConfig:
    """Light component configuration."""
    type: str = "point"  # point, spot, disk, area
    luminance: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    visible: bool = True
    # For spot light
    angle_radians: float = 0.5
    small_angle_radians: float = 0.1
    angle_atten_pow: float = 1.0
    
    def apply_to(self, light: re.world.LightComponent):
        """Apply config to LightComponent."""
        lum = lc.float3(*self.luminance)
        if self.type == "point":
            light.add_point_light(lum, self.visible)
        elif self.type == "spot":
            light.add_spot_light(
                lum, self.angle_radians, self.small_angle_radians,
                self.angle_atten_pow, self.visible
            )
        elif self.type == "disk":
            light.add_disk_light(lum, self.visible)
        elif self.type == "area":
            light.add_area_light(lum, self.visible)


@dataclass
class EntityConfig:
    """Entity configuration."""
    name: str
    transform: TransformConfig = field(default_factory=TransformConfig)
    mesh_path: Optional[str] = None
    material_names: List[str] = field(default_factory=list)
    light: Optional[LightConfig] = None
    children: List["EntityConfig"] = field(default_factory=list)
    
    def create_entity(self, scene: re.world.Scene, project: re.world.Project,
                      material_cache: Dict[str, re.world.MaterialResource]) -> re.world.Entity:
        """Create an entity from this config."""
        entity = scene.add_entity()
        entity.set_name(self.name)
        
        # Add transform
        transform = re.world.TransformComponent(entity.add_component("TransformComponent"))
        self.transform.apply_to(transform)
        
        # Add render component if mesh is specified
        if self.mesh_path:
            mesh = project.import_mesh(self.mesh_path)
            if mesh:
                render = re.world.RenderComponent(entity.add_component("RenderComponent"))
                mat_vector = lc.capsule_vector()
                for mat_name in self.material_names:
                    if mat_name in material_cache:
                        mat_vector.emplace_back(material_cache[mat_name]._handle)
                render.update_object(mat_vector, mesh)
        
        # Add light component if specified
        if self.light:
            light = re.world.LightComponent(entity.add_component("LightComponent"))
            self.light.apply_to(light)
        
        return entity


@dataclass
class SceneConfig:
    """Complete scene configuration."""
    name: str = "unnamed_scene"
    entities: List[EntityConfig] = field(default_factory=list)
    materials: List[MaterialConfig] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneConfig":
        """Create from dictionary."""
        # Parse materials
        materials = [MaterialConfig(**m) for m in data.get("materials", [])]
        
        # Parse entities (recursive)
        def parse_entity(e_data: Dict) -> EntityConfig:
            transform = TransformConfig(**e_data.get("transform", {}))
            light = None
            if "light" in e_data and e_data["light"]:
                light = LightConfig(**e_data["light"])
            children = [parse_entity(c) for c in e_data.get("children", [])]
            return EntityConfig(
                name=e_data["name"],
                transform=transform,
                mesh_path=e_data.get("mesh_path"),
                material_names=e_data.get("material_names", []),
                light=light,
                children=children
            )
        
        entities = [parse_entity(e) for e in data.get("entities", [])]
        
        return cls(
            name=data.get("name", "unnamed_scene"),
            entities=entities,
            materials=materials
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "SceneConfig":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def save_to_file(self, path: Path):
        """Save configuration to JSON file."""
        path.write_text(self.to_json(), encoding="utf-8")
    
    @classmethod
    def load_from_file(cls, path: Path) -> "SceneConfig":
        """Load configuration from JSON file."""
        return cls.from_json(path.read_text(encoding="utf-8"))


@dataclass
class AppConfig:
    """Complete App configuration."""
    display: DisplayConfig = field(default_factory=DisplayConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    render_settings: RenderSettingsConfig = field(default_factory=RenderSettingsConfig)
    ground_plane: GroundPlaneConfig = field(default_factory=GroundPlaneConfig)
    scene_file: Optional[str] = None
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> "AppConfig":
        data = json.loads(json_str)
        return cls(
            display=DisplayConfig(**data.get("display", {})),
            backend=BackendConfig(**data.get("backend", {})),
            camera=CameraConfig(**data.get("camera", {})),
            render_settings=RenderSettingsConfig(**data.get("render_settings", {})),
            ground_plane=GroundPlaneConfig(**data.get("ground_plane", {})),
            scene_file=data.get("scene_file")
        )
    
    def save_to_file(self, path: Path):
        path.write_text(self.to_json(), encoding="utf-8")
    
    @classmethod
    def load_from_file(cls, path: Path) -> "AppConfig":
        return cls.from_json(path.read_text(encoding="utf-8"))


# =============================================================================
# Configuration Manager
# =============================================================================

class ConfigManager:
    """Manages loading and applying configurations to the App."""
    
    def __init__(self, app: rbc.app.App):
        self.app = app
    
    def apply_app_config(self, config: AppConfig):
        """Apply AppConfig to the running App instance."""
        # Apply camera settings if display cam exists
        if self.app.display_cam:
            config.camera.apply_to(self.app.display_cam)
            
            # Apply render settings
            render_settings = self.app.display_cam.render_settings()
            config.render_settings.apply_to(render_settings)
        
        # Apply ground plane settings
        if config.ground_plane.enabled:
            # Create custom material if specified
            if config.ground_plane.mode != "none":
                material = None
                if self.app._project:
                    mat_interface = mat.OpenPBRInterface(self.app._project)
                    mat_interface.set_specular_roughness(config.ground_plane.roughness)
                    mat_interface.set_weight_metallic(config.ground_plane.metallic)
                    mat_interface.set_base_albedo(tuple(config.ground_plane.albedo))
                    material = mat_interface
                
                self.app.set_ground_plane_mode(
                    config.ground_plane.mode,
                    config.ground_plane.scale,
                    config.ground_plane.height,
                    material
                )
        else:
            self.app.set_ground_plane_mode("none")
    
    def load_scene_config(self, config: SceneConfig) -> List[re.world.Entity]:
        """Load a SceneConfig into the current scene."""
        if not self.app.scene or not self.app._project:
            raise RuntimeError("App scene and project must be initialized first")
        
        # Create materials
        material_cache = {}
        for mat_config in config.materials:
            material = mat_config.create_material(self.app._project)
            material_cache[mat_config.name] = material
        
        # Create entities
        entities = []
        for entity_config in config.entities:
            entity = entity_config.create_entity(
                self.app.scene, self.app._project, material_cache
            )
            entities.append(entity)
        
        return entities


# =============================================================================
# Example Configurations
# =============================================================================

def create_example_app_config() -> AppConfig:
    """Create an example App configuration."""
    return AppConfig(
        display=DisplayConfig(
            width=1920,
            height=1080,
            title="RoboCute JSON Config Example",
            resizable=True,
            vsync=True
        ),
        backend=BackendConfig(
            name="dx",
            shader_path=None  # Use default
        ),
        camera=CameraConfig(
            position=[0.0, 1.5, -3.0],
            rotation=[0.0, 0.0, 0.0, 1.0],
            fov=60.0,
            near_plane=0.01,
            far_plane=1000.0,
            aperture=0.0,
            focus_distance=10.0
        ),
        render_settings=RenderSettingsConfig(
            global_exposure=0.0,
            gamma=2.2,
            use_auto_exposure=False,
            sky_color=[0.5, 0.6, 0.7],
            sky_max_lum=1.0,
            sun_color=[1.0, 0.98, 0.95],
            sun_intensity=1.0,
            sun_dir=[0.2, -1.0, 0.3],
            offline_spp=128,
            offline_origin_bounce=4,
            offline_indirect_bounce=4,
            denoise=True
        ),
        ground_plane=GroundPlaneConfig(
            enabled=True,
            mode="grid",
            scale=50.0,
            height=0.0,
            roughness=0.5,
            metallic=0.2,
            albedo=[0.9, 0.9, 0.9]
        ),
        scene_file="example_scene.json"
    )


def create_example_scene_config() -> SceneConfig:
    """Create an example Scene configuration."""
    return SceneConfig(
        name="example_scene",
        materials=[
            MaterialConfig(
                name="red_plastic",
                base_albedo=[0.8, 0.2, 0.2],
                metallic=0.0,
                roughness=0.3
            ),
            MaterialConfig(
                name="blue_metal",
                base_albedo=[0.2, 0.4, 0.8],
                metallic=0.8,
                roughness=0.2
            ),
            MaterialConfig(
                name="green_emissive",
                base_albedo=[0.1, 0.8, 0.2],
                emission=[0.2, 1.0, 0.4],
                metallic=0.0,
                roughness=0.5
            )
        ],
        entities=[
            EntityConfig(
                name="main_light",
                transform=TransformConfig(
                    position=[2.0, 3.0, -2.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                    scale=[1.0, 1.0, 1.0]
                ),
                light=LightConfig(
                    type="point",
                    luminance=[10.0, 10.0, 9.0],
                    visible=True
                )
            ),
            EntityConfig(
                name="fill_light",
                transform=TransformConfig(
                    position=[-2.0, 2.0, 2.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                    scale=[1.0, 1.0, 1.0]
                ),
                light=LightConfig(
                    type="point",
                    luminance=[3.0, 3.0, 4.0],
                    visible=False
                )
            ),
            EntityConfig(
                name="red_cube",
                transform=TransformConfig(
                    position=[-1.0, 0.5, 0.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                    scale=[0.5, 0.5, 0.5]
                ),
                mesh_path="meshes/cube.obj",
                material_names=["red_plastic"]
            ),
            EntityConfig(
                name="blue_sphere",
                transform=TransformConfig(
                    position=[1.0, 0.5, 0.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                    scale=[0.5, 0.5, 0.5]
                ),
                mesh_path="meshes/sphere.obj",
                material_names=["blue_metal"]
            ),
            EntityConfig(
                name="ground_platform",
                transform=TransformConfig(
                    position=[0.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                    scale=[5.0, 0.1, 5.0]
                ),
                mesh_path="meshes/cube.obj",
                material_names=["green_emissive"]
            )
        ]
    )


# =============================================================================
# Main Example
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RoboCute JSON Config Example")
    parser.add_argument(
        "-b", "--backend",
        type=str,
        default="dx",
        help="Graphics backend (dx, vk)"
    )
    parser.add_argument(
        "-p", "--project",
        type=str,
        required=True,
        help="Path to project directory containing rbc_project.json"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to App config JSON file (optional)"
    )
    parser.add_argument(
        "-s", "--scene",
        type=str,
        default=None,
        help="Path to Scene config JSON file (optional)"
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save example configs to files and exit"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=".",
        help="Directory to save/load config files"
    )
    args = parser.parse_args()
    
    project_path = Path(args.project)
    config_dir = Path(args.config_dir)
    
    # Save example configs if requested
    if args.save_config:
        app_config = create_example_app_config()
        scene_config = create_example_scene_config()
        
        app_config_path = config_dir / "example_app_config.json"
        scene_config_path = config_dir / "example_scene_config.json"
        
        app_config.save_to_file(app_config_path)
        scene_config.save_to_file(scene_config_path)
        
        print(f"Saved example configs:")
        print(f"  App config: {app_config_path}")
        print(f"  Scene config: {scene_config_path}")
        print("\nExample App config:")
        print(app_config.to_json())
        print("\nExample Scene config:")
        print(scene_config.to_json())
        return
    
    # Load or create App config
    if args.config:
        print(f"Loading App config from: {args.config}")
        app_config = AppConfig.load_from_file(Path(args.config))
    else:
        print("Using default App config")
        app_config = create_example_app_config()
        app_config.backend.name = args.backend
    
    # Initialize App
    print(f"Initializing App with backend: {app_config.backend.name}")
    app = rbc.app.App()
    app.init(
        backend_name=app_config.backend.name,
        project_path=project_path,
        require_render=True
    )
    
    if not app.ctx:
        print("Failed to initialize context!")
        return
    
    # Initialize display
    print(f"Creating display: {app_config.display.width}x{app_config.display.height}")
    app.init_display(
        x=app_config.display.width,
        y=app_config.display.height,
        display_title=app_config.display.title
    )
    
    if not app.display_cam:
        print("Failed to create display camera!")
        return
    
    # Apply App configuration
    print("Applying App configuration...")
    config_manager = ConfigManager(app)
    config_manager.apply_app_config(app_config)
    
    # Load or create Scene config
    scene_config_path = None
    if args.scene:
        scene_config_path = Path(args.scene)
    elif app_config.scene_file:
        scene_config_path = config_dir / app_config.scene_file
    
    if scene_config_path and scene_config_path.exists():
        print(f"Loading Scene config from: {scene_config_path}")
        scene_config = SceneConfig.load_from_file(scene_config_path)
        config_manager.load_scene_config(scene_config)
    else:
        print("No scene config file found, skipping entity creation")
    
    # Setup camera position from config
    transform = app.get_display_transform()
    if transform:
        pos = app_config.camera.position
        transform.set_pos(lc.double3(*pos), False)
    
    # Enable camera control
    app.ctx.enable_camera_control()
    
    print("\nConfiguration loaded successfully!")
    print("Running main loop (close window to exit)...\n")
    
    # Run the app
    app.run()


if __name__ == "__main__":
    main()
