"""
Isaac Sim scene setup for HydraLab-USV.

Provides scene-construction helpers for the BlueboatEnv._setup_scene() method.
All code is gated behind the Isaac Lab import check so the module imports
cleanly when Isaac Sim is absent.

Scene layout
────────────
World/
  GroundPlane        – large flat plane representing the water surface
  DomeLight          – uniform sky illumination
  WaterMaterial      – simple blue PBR material on the ground plane
  Vessel_0 … N       – per-environment vessel rigid bodies (asset USD)
  Goal_0 … N         – per-environment goal markers (small sphere)
  DisturbanceArrow_0  – (optional) wind/current visualisation arrows

Coordinate convention:
    Isaac Sim uses Y-up.  Our NED convention (X-North, Y-East, Z-Down) must
    be mapped at the Isaac interface layer.  Within this file we work in
    Isaac's Y-up frame; the environment state is always stored in NED.

    NED → Isaac:  (x_ned, y_ned) → (x_isaac=x_ned, z_isaac=-y_ned, y_isaac=0)
    Isaac → NED:  (x_isaac, z_isaac) → (x_ned=x_isaac, y_ned=-z_isaac)

TODO: Replace primitive vessel mesh with a proper BlueBoat USD asset.
TODO: Add per-env origin offsets so vessels don't overlap in the viewport.
TODO: Wire up articulation joints for thruster visual animation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
    import omni.isaac.core.utils.prims as prim_utils
    _ISAAC_AVAILABLE = True
except ImportError:
    _ISAAC_AVAILABLE = False

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Vessel geometry constants
# ---------------------------------------------------------------------------

VESSEL_LENGTH = 1.35   # m (BlueBoat hull length)
VESSEL_BEAM = 0.57     # m (between thruster pontoons)
VESSEL_HEIGHT = 0.25   # m (hull height above waterline)

# Spacing between parallel environments in the viewport (m)
ENV_GRID_SPACING = 10.0


# ---------------------------------------------------------------------------
# Scene setup functions
# ---------------------------------------------------------------------------

def setup_water_surface(prim_path: str = "/World/WaterSurface") -> None:
    """Create a large flat ground plane styled as a water surface.

    Args:
        prim_path: USD prim path for the plane.
    """
    if not _ISAAC_AVAILABLE:
        return
    cfg = sim_utils.GroundPlaneCfg(
        size=(1000.0, 1000.0),
        color=(0.05, 0.35, 0.65),  # ocean blue
    )
    cfg.func(prim_path, cfg)


def setup_lighting(prim_path: str = "/World/SkyLight") -> None:
    """Add a dome light for uniform outdoor illumination."""
    if not _ISAAC_AVAILABLE:
        return
    cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(1.0, 0.98, 0.9),  # warm daylight
    )
    cfg.func(prim_path, cfg)


def build_vessel_cfg(env_idx: int = 0) -> "RigidObjectCfg | None":
    """Build an Isaac Lab RigidObjectCfg for a single vessel instance.

    Uses a primitive box as a placeholder until a USD asset is available.

    Args:
        env_idx: Environment index (for unique prim naming).

    Returns:
        RigidObjectCfg or None if Isaac Lab is unavailable.
    """
    if not _ISAAC_AVAILABLE:
        return None

    # TODO: Replace with:
    #   spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/blueboat.usd")
    spawn_cfg = sim_utils.CuboidCfg(
        size=(VESSEL_LENGTH, VESSEL_BEAM, VESSEL_HEIGHT),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,   # vessel is on water; buoyancy handled externally
            linear_damping=0.0,     # we apply our own damping in dynamics
            angular_damping=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.8, 0.3, 0.1),  # orange hull
        ),
    )

    return RigidObjectCfg(
        prim_path=f"/World/Vessel_{env_idx}",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, VESSEL_HEIGHT / 2, 0.0),  # Y-up: sitting on waterline
        ),
    )


def build_goal_marker_cfg(env_idx: int = 0) -> "sim_utils.SphereCfg | None":
    """Build a small sphere to mark the current goal position.

    Args:
        env_idx: Environment index.

    Returns:
        SphereCfg or None.
    """
    if not _ISAAC_AVAILABLE:
        return None
    return sim_utils.SphereCfg(
        radius=0.3,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.1, 0.1),  # red goal sphere
            opacity=0.7,
        ),
    )


def env_origin(env_idx: int, envs_per_row: int = 32) -> tuple[float, float]:
    """Compute the Isaac Sim world-frame origin for environment `env_idx`.

    Environments are laid out on a regular grid so they don't overlap
    in the viewport.

    Args:
        env_idx:      Environment index.
        envs_per_row: Number of environments per grid row.

    Returns:
        (x_origin, z_origin) in Isaac Sim Y-up frame.
    """
    row = env_idx // envs_per_row
    col = env_idx % envs_per_row
    return col * ENV_GRID_SPACING, row * ENV_GRID_SPACING


def ned_to_isaac(x_ned: float, y_ned: float) -> tuple[float, float, float]:
    """Convert NED 2D position to Isaac Sim 3D position (Y-up).

    Returns:
        (x_isaac, y_isaac, z_isaac)
    """
    return x_ned, 0.0, -y_ned


def isaac_to_ned(x_isaac: float, z_isaac: float) -> tuple[float, float]:
    """Convert Isaac Sim position to NED.

    Returns:
        (x_ned, y_ned)
    """
    return x_isaac, -z_isaac


def update_vessel_poses(
    vessel_prims,          # list of Isaac prim handles
    vessel_states,         # (num_envs, 6) tensor in NED
    env_origins,           # (num_envs, 2) grid offsets
) -> None:
    """Synchronise Isaac Sim vessel prim poses with the dynamics state.

    Called every render frame (not every physics step).

    Args:
        vessel_prims:  List of Isaac rigid-body prim handles.
        vessel_states: (num_envs, 6) NED state tensor.
        env_origins:   (num_envs, 2) per-env grid offsets in Isaac frame.
    """
    if not _ISAAC_AVAILABLE:
        return

    for i, prim in enumerate(vessel_prims):
        x_ned = float(vessel_states[i, 0])
        y_ned = float(vessel_states[i, 1])
        psi = float(vessel_states[i, 2])

        ox, oz = env_origins[i]
        xi, yi, zi = ned_to_isaac(x_ned + ox, y_ned + oz)

        # TODO: set prim translation and yaw rotation via Isaac API
        # prim.set_world_pose(position=(xi, yi, zi), orientation=yaw_quat(psi))
        _ = (xi, yi, zi, psi)  # silence unused-variable warnings
