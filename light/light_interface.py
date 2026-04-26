"""Light JSON-RPC Interface — high-level light/capture API built on JSONRPCServer.

This module registers light management, scene capture, and quality-control
methods onto a ``JSONRPCServer`` instance.  All state is kept in an
in-memory ``LightBackend`` so the server can be exercised without a live
UE5 editor session.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Import the generic JSON-RPC server from the network package
# ---------------------------------------------------------------------------
from network.rpc_server import JSONRPCServer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    position: list[float] = field(default_factory=lambda: [0.0, 150.0, 200.0])
    rotation: list[float] = field(default_factory=lambda: [0.0, -15.0, 0.0])
    fov: float = 90.0
    resolution: list[int] = field(default_factory=lambda: [512, 512])


@dataclass
class SceneInfo:
    scene_name: str = "default_scene"
    bbox_min: list[float] = field(default_factory=lambda: [-500.0, -100.0, -500.0])
    bbox_max: list[float] = field(default_factory=lambda: [500.0, 300.0, 500.0])
    coordinate_system: str = "UE5_LeftHanded_ZUp"


@dataclass
class LightParam:
    type: str = "PointLight"
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_6d: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    color_rgb: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    log_intensity: float = 0.0
    source_radius: float = 0.05
    temperature: int = 6500
    attenuation_radius: float = 10.0
    bCastShadows: bool = True
    inner_cone_angle: float = 0.0
    outer_cone_angle: float = 0.0
    source_width: float = 0.0
    source_height: float = 0.0


@dataclass
class CaptureConfig:
    output_dir: str = "D:/dataset/default_scene"
    max_capture_radius: float = 50.0
    frustum_expansion: float = 1.5
    min_intensity_threshold: float = 0.01
    include_disabled: bool = False


# ---------------------------------------------------------------------------
# LightBackend — in-memory mock of the UE5 lighting subsystem
# ---------------------------------------------------------------------------

class LightBackend:
    """In-memory backend that simulates a UE5 scene with lights."""

    def __init__(self) -> None:
        self._lights: list[LightParam] = []
        self._visibilities: list[bool] = []
        self._scene_info = SceneInfo()
        self._camera = Camera()
        self._capture_config = CaptureConfig()
        self._capture_counter = 0

    # -- seed data for demos / tests --
    def seed_defaults(self, count: int = 3) -> None:
        """Populate with *count* default lights."""
        self._lights.clear()
        self._visibilities.clear()
        for i in range(count):
            lp = LightParam(
                type="SpotLight" if i % 2 == 0 else "PointLight",
                position=[round(random.random(), 3) for _ in range(3)],
                log_intensity=random.uniform(1.0, 4.0),
                temperature=random.choice([3200, 4500, 5600, 6500]),
                outer_cone_angle=0.79 if i % 2 == 0 else 0.0,
            )
            self._lights.append(lp)
            self._visibilities.append(True)

    # -- light CRUD --
    def list_lights(self) -> list[LightParam]:
        return list(self._lights)

    def get_light(self, index: int) -> Optional[LightParam]:
        if 0 <= index < len(self._lights):
            return self._lights[index]
        return None

    def create_light(self, light: LightParam) -> int:
        self._lights.append(light)
        self._visibilities.append(True)
        return len(self._lights) - 1

    def update_light(self, index: int, light: LightParam) -> bool:
        if 0 <= index < len(self._lights):
            self._lights[index] = light
            return True
        return False

    def delete_light(self, index: int) -> bool:
        if 0 <= index < len(self._lights):
            del self._lights[index]
            del self._visibilities[index]
            return True
        return False

    def set_visibility(self, index: int, visible: bool) -> bool:
        if 0 <= index < len(self._visibilities):
            self._visibilities[index] = visible
            return True
        return False

    def scan_lights(self) -> list[LightParam]:
        """Return lights that pass a trivial relevance filter."""
        cfg = self._capture_config
        result: list[LightParam] = []
        for light in self._lights:
            if light.log_intensity < cfg.min_intensity_threshold:
                continue
            result.append(light)
        return result

    # -- scene / camera --
    def get_scene_info(self) -> SceneInfo:
        return self._scene_info

    def get_camera(self) -> Camera:
        return self._camera

    def set_capture_config(self, config: CaptureConfig) -> bool:
        self._capture_config = config
        return True

    # -- capture simulation --
    def capture_gbuffer(self, out_path: str) -> dict[str, str]:
        os.makedirs(os.path.join(out_path, "gbuffer"), exist_ok=True)
        files = {
            "world_normal": "gbuffer/world_normal.png",
            "albedo": "gbuffer/albedo.png",
            "packed_rm_depth": "gbuffer/packed_rm_depth.png",
        }
        for name, relpath in files.items():
            with open(os.path.join(out_path, relpath), "w") as fh:
                fh.write(f"# stub {name}\n")
        return files

    def render_olat(self, out_path: str, light_indices: Optional[list[int]] = None) -> dict[str, Any]:
        targets = light_indices if light_indices is not None else list(range(len(self._lights)))
        os.makedirs(os.path.join(out_path, "olat"), exist_ok=True)
        light_files: list[str] = []
        for idx, li in enumerate(targets):
            fname = f"olat/light_{idx+1:03d}.exr"
            with open(os.path.join(out_path, fname), "w") as fh:
                fh.write(f"# stub OLAT for light {li}\n")
            light_files.append(fname)
        for name in ("rgb_ambient.exr", "rgb_composite.exr"):
            with open(os.path.join(out_path, name), "w") as fh:
                fh.write(f"# stub {name}\n")
        return {
            "ambient": "rgb_ambient.exr",
            "lights": light_files,
            "composite": "rgb_composite.exr",
        }

    def capture_full(self, out_path: str) -> dict[str, Any]:
        capture_id = f"capture_{self._capture_counter:04d}"
        self._capture_counter += 1
        cap_dir = os.path.join(out_path, capture_id)
        os.makedirs(cap_dir, exist_ok=True)
        gbuf = self.capture_gbuffer(cap_dir)
        olat = self.render_olat(cap_dir)
        lp = {
            "camera": asdict(self._camera),
            "scene_bbox": {
                "min": self._scene_info.bbox_min,
                "max": self._scene_info.bbox_max,
            },
            "lights": [asdict(l) for l in self._lights],
            "filtered_lights": [asdict(l) for l in self.scan_lights()],
        }
        lp_path = os.path.join(cap_dir, "light_params.json")
        with open(lp_path, "w", encoding="utf-8") as fh:
            json.dump(lp, fh, indent=2, ensure_ascii=False)
        return {
            "capture_id": capture_id,
            "light_count": len(self._lights),
            "gbuffer": gbuf,
            "olat": olat,
            "light_params_file": "light_params.json",
        }

    # -- quality control --
    def check_quality(self, capture_path: str) -> dict[str, Any]:
        issues: list[str] = []
        # simplistic checks on stub files
        for fname in ("rgb_composite.exr", "rgb_ambient.exr"):
            fpath = os.path.join(capture_path, fname)
            if not os.path.exists(fpath):
                issues.append(f"Missing {fname}")
        gbuf_dir = os.path.join(capture_path, "gbuffer")
        for fname in ("world_normal.png", "albedo.png", "packed_rm_depth.png"):
            if not os.path.exists(os.path.join(gbuf_dir, fname)):
                issues.append(f"Missing gbuffer/{fname}")
        # fake ratio based on light count
        ratio = min(1.0, len(self._lights) * 0.15)
        category = (
            "Light-dominant" if ratio > 0.6 else
            "Mixed" if ratio > 0.3 else
            "Emissive-dominant"
        )
        return {
            "pass": len(issues) == 0,
            "light_contribution_ratio": round(ratio, 2),
            "scene_category": category,
            "issues": issues,
        }


# ---------------------------------------------------------------------------
# LightInterface — registers methods onto a JSONRPCServer instance
# ---------------------------------------------------------------------------

class LightInterface:
    """Wraps a ``LightBackend`` and registers JSON-RPC methods on a server."""

    def __init__(self, backend: LightBackend, server: JSONRPCServer):
        self._backend = backend
        self._server = server
        self._register_all()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _light_not_found(index: int) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32001,
                "message": f"Light not found: index {index}",
            },
        }

    @staticmethod
    def _ok(result: Any) -> Any:
        return result

    def _register_all(self) -> None:
        s = self._server
        b = self._backend

        # -- light.scan --
        def light_scan(client_id: int) -> list[dict[str, Any]]:
            return [asdict(l) for l in b.scan_lights()]

        # -- light.list --
        def light_list(client_id: int) -> list[dict[str, Any]]:
            return [asdict(l) for l in b.list_lights()]

        # -- light.get --
        def light_get(client_id: int, index: int) -> Optional[dict[str, Any]]:
            light = b.get_light(index)
            return asdict(light) if light else None

        # -- light.create --
        def light_create(client_id: int, light: dict[str, Any]) -> int:
            lp = LightParam(**light)
            return b.create_light(lp)

        # -- light.update --
        def light_update(client_id: int, index: int, light: dict[str, Any]) -> bool:
            lp = LightParam(**light)
            if not b.update_light(index, lp):
                raise RuntimeError(f"Light not found: index {index}")
            return True

        # -- light.delete --
        def light_delete(client_id: int, index: int) -> bool:
            if not b.delete_light(index):
                raise RuntimeError(f"Light not found: index {index}")
            return True

        # -- light.set_visibility --
        def light_set_visibility(client_id: int, index: int, visible: bool) -> bool:
            if not b.set_visibility(index, visible):
                raise RuntimeError(f"Light not found: index {index}")
            return True

        # -- scene.get_info --
        def scene_get_info(client_id: int) -> dict[str, Any]:
            return asdict(b.get_scene_info())

        # -- scene.get_camera --
        def scene_get_camera(client_id: int) -> dict[str, Any]:
            return asdict(b.get_camera())

        # -- capture.set_config --
        def capture_set_config(client_id: int, config: dict[str, Any]) -> bool:
            cfg = CaptureConfig(**config)
            return b.set_capture_config(cfg)

        # -- capture.gbuffer --
        def capture_gbuffer(client_id: int, out_path: str) -> dict[str, str]:
            return b.capture_gbuffer(out_path)

        # -- capture.olat --
        def capture_olat(
            client_id: int, out_path: str, light_indices: Optional[list[int]] = None
        ) -> dict[str, Any]:
            return b.render_olat(out_path, light_indices)

        # -- capture.full --
        def capture_full(client_id: int, out_path: str) -> dict[str, Any]:
            return b.capture_full(out_path)

        # -- quality.check --
        def quality_check(client_id: int, capture_path: str) -> dict[str, Any]:
            return b.check_quality(capture_path)

        # -- system.ping --
        def system_ping(client_id: int) -> str:
            return "pong"

        # -- system.version --
        def system_version(client_id: int) -> str:
            return "light-rpc/1.0.0"

        # ------------------------------------------------------------------
        # Register every method
        # ------------------------------------------------------------------
        s.register("light.scan", light_scan)
        s.register("light.list", light_list)
        s.register("light.get", light_get)
        s.register("light.create", light_create)
        s.register("light.update", light_update)
        s.register("light.delete", light_delete)
        s.register("light.set_visibility", light_set_visibility)
        s.register("scene.get_info", scene_get_info)
        s.register("scene.get_camera", scene_get_camera)
        s.register("capture.set_config", capture_set_config)
        s.register("capture.gbuffer", capture_gbuffer)
        s.register("capture.olat", capture_olat)
        s.register("capture.full", capture_full)
        s.register("quality.check", quality_check)
        s.register("system.ping", system_ping)
        s.register("system.version", system_version)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def create_light_server(
    host: str = "127.0.0.1",
    port: int = 8888,
    backend: Optional[LightBackend] = None,
) -> tuple[JSONRPCServer, LightBackend, LightInterface]:
    """Create and wire up a JSONRPCServer + LightBackend + LightInterface trio."""
    server = JSONRPCServer(host=host, port=port)
    back = backend or LightBackend()
    iface = LightInterface(back, server)
    return server, back, iface
