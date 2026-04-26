"""Comprehensive tests for the Light JSON-RPC interface.

Run with:
    pytest tests/test_light_interface.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time

import pytest

# Ensure parent package is on path when running from repo root
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from light_interface import create_light_server, LightBackend, LightParam, CaptureConfig
from network.rpc_server import JSONRPCServer
from network.rpc_client import JSONRPCClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    path = tempfile.mkdtemp(prefix="light_rpc_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="function")
def server_and_client():
    """Spin up a LightInterface server and a connected JSON-RPC client."""
    backend = LightBackend()
    backend.seed_defaults(count=3)
    port = 29997
    max_retry = 5
    srv = None
    client = None
    for attempt in range(max_retry):
        try_port = port + attempt
        try:
            srv, back, iface = create_light_server(
                host="127.0.0.1", port=try_port, backend=backend
            )
            srv.start(blocking=False)
            time.sleep(0.2)
            client = JSONRPCClient(host="127.0.0.1", port=try_port)
            if client.connect():
                break
            client.disconnect()
            srv.stop()
        except Exception:
            if srv is not None:
                try:
                    srv.stop()
                except Exception:
                    pass
            if client is not None:
                try:
                    client.disconnect()
                except Exception:
                    pass
            if attempt == max_retry - 1:
                raise
    else:
        raise RuntimeError("Failed to start server and connect client")

    yield srv, back, client

    if client is not None:
        client.disconnect()
    if srv is not None:
        srv.stop()
    time.sleep(0.1)


# ---------------------------------------------------------------------------
# System / health
# ---------------------------------------------------------------------------

def test_system_ping(server_and_client):
    _, _, client = server_and_client
    assert client.call("system.ping") == "pong"


def test_system_version(server_and_client):
    _, _, client = server_and_client
    assert client.call("system.version") == "light-rpc/1.0.0"


def test_method_not_found(server_and_client):
    _, _, client = server_and_client
    with pytest.raises(RuntimeError, match="Method not found"):
        client.call("nonexistent.method")


def test_invalid_params_type(server_and_client):
    _, _, client = server_and_client
    # light.get expects an int; passing a string should trigger TypeError
    with pytest.raises(RuntimeError, match="Invalid params"):
        client.call("light.get", "not_an_int")


# ---------------------------------------------------------------------------
# Light management
# ---------------------------------------------------------------------------

def test_light_list_returns_three(server_and_client):
    _, backend, client = server_and_client
    result = client.call("light.list")
    assert isinstance(result, list)
    assert len(result) == 3


def test_light_scan_returns_subset(server_and_client):
    _, backend, client = server_and_client
    # all seeded lights pass the default threshold
    result = client.call("light.scan")
    assert isinstance(result, list)
    assert len(result) == 3
    # tweak threshold so none pass
    backend.set_capture_config(CaptureConfig(min_intensity_threshold=999.0))
    result = client.call("light.scan")
    assert len(result) == 0


def test_light_get_valid_index(server_and_client):
    _, _, client = server_and_client
    light = client.call("light.get", 0)
    assert isinstance(light, dict)
    assert "type" in light
    assert "position" in light


def test_light_get_invalid_index(server_and_client):
    _, _, client = server_and_client
    assert client.call("light.get", 99) is None


def test_light_create(server_and_client):
    _, backend, client = server_and_client
    new_light = {
        "type": "RectLight",
        "position": [0.1, 0.2, 0.3],
        "rotation_6d": [0.0] * 6,
        "color_rgb": [1.0, 0.0, 0.0],
        "log_intensity": 3.0,
        "source_radius": 0.1,
        "temperature": 5600,
        "attenuation_radius": 20.0,
        "bCastShadows": False,
        "inner_cone_angle": 0.0,
        "outer_cone_angle": 0.0,
        "source_width": 2.0,
        "source_height": 1.0,
    }
    idx = client.call("light.create", new_light)
    assert isinstance(idx, int)
    assert idx == 3
    fetched = client.call("light.get", idx)
    assert fetched["type"] == "RectLight"
    assert fetched["source_width"] == 2.0


def test_light_update(server_and_client):
    _, _, client = server_and_client
    updated = {
        "type": "DirectionalLight",
        "position": [0.5, 0.5, 0.5],
        "rotation_6d": [0.0] * 6,
        "color_rgb": [0.0, 0.0, 1.0],
        "log_intensity": 5.0,
        "source_radius": 0.0,
        "temperature": 6500,
        "attenuation_radius": 0.0,
        "bCastShadows": True,
        "inner_cone_angle": 0.0,
        "outer_cone_angle": 0.0,
        "source_width": 0.0,
        "source_height": 0.0,
    }
    assert client.call("light.update", 0, updated) is True
    fetched = client.call("light.get", 0)
    assert fetched["type"] == "DirectionalLight"
    assert fetched["log_intensity"] == 5.0


def test_light_update_not_found(server_and_client):
    _, _, client = server_and_client
    dummy = {k: (0 if k != "type" else "PointLight") for k in LightParam.__dataclass_fields__}
    dummy["position"] = [0.0, 0.0, 0.0]
    dummy["rotation_6d"] = [0.0] * 6
    dummy["color_rgb"] = [1.0, 1.0, 1.0]
    with pytest.raises(RuntimeError, match="Light not found"):
        client.call("light.update", 99, dummy)


def test_light_delete(server_and_client):
    _, _, client = server_and_client
    assert client.call("light.delete", 0) is True
    # list shrinks; remaining lights shift down
    assert len(client.call("light.list")) == 2
    assert client.call("light.get", 2) is None


def test_light_delete_not_found(server_and_client):
    _, _, client = server_and_client
    with pytest.raises(RuntimeError, match="Light not found"):
        client.call("light.delete", 99)


def test_light_set_visibility(server_and_client):
    _, backend, client = server_and_client
    assert client.call("light.set_visibility", 0, False) is True
    assert backend._visibilities[0] is False
    assert client.call("light.set_visibility", 0, True) is True
    assert backend._visibilities[0] is True


def test_light_set_visibility_not_found(server_and_client):
    _, _, client = server_and_client
    with pytest.raises(RuntimeError, match="Light not found"):
        client.call("light.set_visibility", 99, False)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def test_scene_get_info(server_and_client):
    _, _, client = server_and_client
    info = client.call("scene.get_info")
    assert info["scene_name"] == "default_scene"
    assert "bbox_min" in info
    assert "bbox_max" in info


def test_scene_get_camera(server_and_client):
    _, _, client = server_and_client
    cam = client.call("scene.get_camera")
    assert "position" in cam
    assert "fov" in cam
    assert cam["resolution"] == [512, 512]


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def test_capture_set_config(server_and_client):
    _, backend, client = server_and_client
    cfg = {
        "output_dir": "D:/dataset/test_scene",
        "max_capture_radius": 100.0,
        "frustum_expansion": 2.0,
        "min_intensity_threshold": 0.05,
        "include_disabled": True,
    }
    assert client.call("capture.set_config", cfg) is True
    assert backend._capture_config.output_dir == "D:/dataset/test_scene"
    assert backend._capture_config.min_intensity_threshold == 0.05


def test_capture_gbuffer(server_and_client, temp_dir):
    _, _, client = server_and_client
    result = client.call("capture.gbuffer", temp_dir)
    assert "world_normal" in result
    assert os.path.exists(os.path.join(temp_dir, result["world_normal"]))
    assert os.path.exists(os.path.join(temp_dir, result["albedo"]))
    assert os.path.exists(os.path.join(temp_dir, result["packed_rm_depth"]))


def test_capture_olat_all_lights(server_and_client, temp_dir):
    _, _, client = server_and_client
    result = client.call("capture.olat", temp_dir)
    assert "ambient" in result
    assert "composite" in result
    assert len(result["lights"]) == 3
    for lf in result["lights"]:
        assert os.path.exists(os.path.join(temp_dir, lf))


def test_capture_olat_subset(server_and_client, temp_dir):
    _, _, client = server_and_client
    result = client.call("capture.olat", temp_dir, [0, 2])
    assert len(result["lights"]) == 2


def test_capture_full(server_and_client, temp_dir):
    _, _, client = server_and_client
    result = client.call("capture.full", temp_dir)
    assert result["capture_id"].startswith("capture_")
    assert result["light_count"] == 3
    cap_dir = os.path.join(temp_dir, result["capture_id"])
    assert os.path.isdir(cap_dir)
    assert os.path.exists(os.path.join(cap_dir, result["light_params_file"]))
    # verify gbuffer & olat stubs exist
    for f in ("gbuffer/world_normal.png", "rgb_ambient.exr", "rgb_composite.exr"):
        assert os.path.exists(os.path.join(cap_dir, f))


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

def test_quality_check_pass(server_and_client, temp_dir):
    _, _, client = server_and_client
    client.call("capture.full", temp_dir)
    cap_dir = os.path.join(temp_dir, "capture_0000")
    result = client.call("quality.check", cap_dir)
    assert result["pass"] is True
    assert result["light_contribution_ratio"] >= 0.0
    assert result["scene_category"] in ("Light-dominant", "Mixed", "Emissive-dominant")
    assert isinstance(result["issues"], list)


def test_quality_check_missing_files(server_and_client, temp_dir):
    _, _, client = server_and_client
    result = client.call("quality.check", temp_dir)
    assert result["pass"] is False
    assert len(result["issues"]) > 0


# ---------------------------------------------------------------------------
# Multi-client stress (smoke)
# ---------------------------------------------------------------------------

def test_multiple_clients():
    """Two clients talking to the same server should not interfere."""
    backend = LightBackend()
    backend.seed_defaults(count=2)
    srv, back, iface = create_light_server(host="127.0.0.1", port=29990, backend=backend)
    srv.start(blocking=False)
    time.sleep(0.2)

    c1 = JSONRPCClient(host="127.0.0.1", port=29990)
    c2 = JSONRPCClient(host="127.0.0.1", port=29990)
    assert c1.connect()
    assert c2.connect()

    try:
        assert c1.call("light.list") == c2.call("light.list")
        c1.call("light.delete", 0)
        assert len(c1.call("light.list")) == 1
        assert len(c2.call("light.list")) == 1
    finally:
        c1.disconnect()
        c2.disconnect()
        srv.stop()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_create_light_minimal_fields(server_and_client):
    _, _, client = server_and_client
    # only supply the required fields the rest fall back to dataclass defaults
    idx = client.call("light.create", {"type": "PointLight"})
    fetched = client.call("light.get", idx)
    assert fetched["type"] == "PointLight"
    assert fetched["position"] == [0.0, 0.0, 0.0]


def test_empty_light_list(server_and_client):
    _, _, client = server_and_client
    client.call("light.delete", 0)
    client.call("light.delete", 0)
    client.call("light.delete", 0)
    assert client.call("light.list") == []
    assert client.call("light.scan") == []
