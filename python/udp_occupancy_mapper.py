# ============================================================================
# UDP Occupancy Mapper
# ============================================================================
# This script builds a 2D occupancy grid map from the robot's LiDAR data.
#
# Main tasks:
# - Receives binary UDP packets from the Arduino.
# - Reads encoder, steering, and LiDAR distance values from each packet.
# - Updates the robot pose using encoder-based odometry.
# - Projects LiDAR rays into the map to mark free and occupied cells.
# - Detects AprilTags and stores them as landmarks in the map.
# - Displays the live map in an OpenCV window while the robot is driven.
# - Saves the final map as PNG, PGM, YAML, tag JSON, and metrics JSON files.
# ============================================================================

import math
import argparse
import json
import pickle
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# Change these values only if your robot / room / scan orientation needs tuning.
UDP_PORT = 4010
SOCKET_TIMEOUT_S = 0.02

# LiDAR packet / scan settings
EXPECTED_NUM_BINS = 180
LIDAR_RANGE_MIN_M = 0.05
LIDAR_RANGE_MAX_M = 3.5
OCCUPIED_ENDPOINT_MAX_M = 2.5
FREE_RAY_MAX_M = 2.0

# Odometry settings
METERS_PER_TICK = 0.00019
WHEELBASE_M = 0.26
STEERING_DEG_PER_CMD = 1.5
STEERING_SIGN = -1.0   # use -1.0 if map turns opposite to command; use +1.0 if correct
STEERING_OFFSET_CMD = 3.0

# Treat small steering changes around the straight trim as correction/noise,
# not as real turning. This is important because we often manually change
# -7 to -6/-8/-4 just to keep the robot physically straight.
STRAIGHT_STEER_DEADBAND_CMD = 4.0

# Optional: make turning less aggressive in the odometry model.
# If turns still look too curved, reduce this to 0.5.
STEERING_ODOM_SCALE = 0.6

# LiDAR angle handling
# If the map looks mirrored left/right, toggle FLIP_SCAN.
# If the map looks rotated, change ANGLE_ZERO_OFFSET_DEG by 90 / 180 / -90, etc.
FLIP_SCAN = True
ANGLE_ZERO_OFFSET_DEG = 0.0

# laser position relative to robot base center
LASER_X_M = 0.16
LASER_Y_M = 0.00

# Occupancy grid settings
MAP_RESOLUTION_M = 0.03
MAP_SIZE_X_M = 14.0
MAP_SIZE_Y_M = 14.0

# Log-odds update strengths
LOG_ODDS_OCC = 1.00
LOG_ODDS_FREE = -0.10
LOG_ODDS_MIN = -4.0
LOG_ODDS_MAX = 4.0

# Display settings
DISPLAY_SCALE = 2
DISPLAY_EVERY_N_PACKETS = 1
MAP_GROW_MARGIN_M = 3.5   # keep at least this much free border around the robot/sensor
PRINT_EVERY_N_PACKETS = 10

# Raw packet logging / replay
LOG_FLUSH_EVERY_N_PACKETS = 25
REPLAY_SLEEP_CAP_S = 0.20

# ----------------------------------------------------------------------------
# Scan-matching / Manhattan-world settings
# ----------------------------------------------------------------------------
SCAN_MATCH_ENABLE = False
SCAN_MATCH_MIN_POINTS = 40
SCAN_MATCH_OCC_THRESH = 0.60
SCAN_MATCH_MAX_RANGE_M = 3.0
SCAN_MATCH_SUBSAMPLE = 4

# Two-stage local correlative search around odometry-predicted pose
SCAN_MATCH_COARSE_XY_SEARCH_M = 0.12
SCAN_MATCH_COARSE_XY_STEP_M = 0.06
SCAN_MATCH_COARSE_THETA_SEARCH_DEG = 8.0
SCAN_MATCH_COARSE_THETA_STEP_DEG = 4.0

SCAN_MATCH_FINE_XY_SEARCH_M = 0.03
SCAN_MATCH_FINE_XY_STEP_M = 0.03
SCAN_MATCH_FINE_THETA_SEARCH_DEG = 2.0
SCAN_MATCH_FINE_THETA_STEP_DEG = 1.0

# Wider scan-match search while turning / corridor lock is weak.
TURN_SCAN_MATCH_COARSE_XY_SEARCH_M = 0.15
TURN_SCAN_MATCH_COARSE_XY_STEP_M = 0.05
TURN_SCAN_MATCH_COARSE_THETA_SEARCH_DEG = 20.0
TURN_SCAN_MATCH_COARSE_THETA_STEP_DEG = 5.0

TURN_SCAN_MATCH_FINE_XY_SEARCH_M = 0.05
TURN_SCAN_MATCH_FINE_XY_STEP_M = 0.025
TURN_SCAN_MATCH_FINE_THETA_SEARCH_DEG = 6.0
TURN_SCAN_MATCH_FINE_THETA_STEP_DEG = 1.0

MANHATTAN_ENABLE = True
MANHATTAN_SNAP_THRESH_DEG = 45.0
MANHATTAN_BLEND = 1.0
FORCE_90_RELOCK_AFTER_TURN = False


# Hard Manhattan mode:
# In straight mode, do not let noisy wall fits rotate the robot heading.
# The robot moves only along locked 0/90/180/-90 headings.
HARD_LOCK_STRAIGHT_HEADING = False

# If the forced 90-degree turn goes the wrong direction, flip this to -1.0.
TURN_DIRECTION_SIGN = -1.0

# The red diagonals are mainly pose/heading errors, not endpoint-filter errors.
# Turning this on too early can delete real hallway wall points too.
FILTER_DIAGONAL_ENDPOINTS = False
ENDPOINT_AXIS_TOL_DEG = 18.0

WALL_FIT_MIN_POINTS = 8
WALL_FIT_MAX_RANGE_M = 2.5
LEFT_WALL_MIN_DEG = 55.0
LEFT_WALL_MAX_DEG = 125.0
RIGHT_WALL_MIN_DEG = 235.0
RIGHT_WALL_MAX_DEG = 305.0

# ----------------------------------------------------------------------------
# Lean mapper state machine
# ----------------------------------------------------------------------------
MODE_STRAIGHT = "STRAIGHT"
MODE_TURNING = "TURNING"
MODE_RELOCK = "RELOCK"

# If the applied steering deviates this much from nominal straight,
# treat it as a turn hint.
TURN_STEER_TRIGGER_CMD = 999.0

# If scan matching has to rotate the pose by this much in one update,
# that is also a strong hint that we are in a turn / transition region.
TURN_SCAN_MATCH_CORR_THRESH_DEG = 6.0

# Corridor-width sanity check used for STRAIGHT / RELOCK gating.
CORRIDOR_WIDTH_MIN_M = 0.60
CORRIDOR_WIDTH_MAX_M = 3.00
CORRIDOR_SIDE_HALF_WIDTH_DEG = 12
CORRIDOR_SIDE_MIN_RANGE_M = 0.15
CORRIDOR_SIDE_MAX_RANGE_M = 2.50

# RELOCK logic: corridor axis must be stable for a few scans and be close
# either to the old axis or the old axis +/- 90 deg.
RELOCK_REQUIRED_STABLE_SCANS = 2
RELOCK_AXIS_MATCH_THRESH_DEG = 45.0
RELOCK_AXIS_JITTER_THRESH_DEG = 8.0

# Save location
SAVE_DIR = Path("occupancy_maps")

# ----------------------------------------------------------------------------
# AprilTag landmark mapping
# ----------------------------------------------------------------------------
# Enable with:
#   python udp_occupancy_mapper.py --tags --camera-index 1
#
# Assumption:
# - Camera is rigidly mounted on the robot and faces forward.
# - OpenCV camera frame: x=right, y=down, z=forward.
# - Robot/map frame: x=forward, y=left.
#
# If tag positions look shifted, tune CAMERA_X_M / CAMERA_Y_M.
# If left/right is flipped, the x_cam -> y_robot conversion below is the place to fix.
APRILTAG_CAMERA_INDEX = 1
APRILTAG_SIZE_M = 0.167
APRILTAG_MIN_Z_M = 0.15
APRILTAG_MAX_Z_M = 3.00
APRILTAG_MIN_OBSERVATIONS_TO_SAVE = 3
APRILTAG_MAX_OBS_PER_ID = 200
APRILTAG_PROCESS_EVERY_S = 0.05
APRILTAG_STATUS_WRITE_EVERY_S = 0.25

# Camera pose relative to robot base center.
# Start with these, then tune if needed.
CAMERA_X_M = 0.16
CAMERA_Y_M = 0.00
CAMERA_YAW_OFFSET_DEG = 0.0

LIVE_TAG_STATUS_PATH = SAVE_DIR / "live_tag_status.json"
CAMERA_RESTART_REQUEST_PATH = SAVE_DIR / "restart_camera_request.json"

# Same calibration values from your AprilTag tracker.
CAM_MATRIX = np.array([
    [656.96495861,   0.0,        630.84080906],
    [  0.0,        658.39179162, 378.57855594],
    [  0.0,          0.0,          1.0       ],
], dtype=np.float32)

DIST_COEFFS = np.array(
    [[0.10811153, -0.27102501, 0.00188884, 0.00495593, 0.20428807]],
    dtype=np.float32,
)

# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def angle_wrap(angle_rad: float) -> float:
    """Wrap any angle to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def angle_deg_wrap_360(angle_deg: float) -> float:
    """Wrap any angle to [0, 360)."""
    return angle_deg % 360.0

def line_angle_diff_abs(a_rad: float, b_rad: float) -> float:
    """
    Smallest absolute difference between two undirected line angles.
    Result is in [0, pi/2].
    """
    return abs(line_angle_normalize(a_rad - b_rad))

def line_angle_normalize(angle_rad: float) -> float:
    """
    Normalize an undirected line angle to [-pi/2, pi/2).
    A line at theta and theta+pi is the same physical line.
    """
    return ((angle_rad + math.pi / 2.0) % math.pi) - math.pi / 2.0

def nearest_manhattan_heading(angle_rad: float) -> float:
    """
    Snap an oriented robot heading to nearest 0, 90, 180, -90 degree direction.
    """
    return angle_wrap(round(angle_rad / (math.pi / 2.0)) * (math.pi / 2.0))

def circular_mean_rad(angles_rad):
    """
    Mean of angles without breaking around +/- pi.
    """
    if not angles_rad:
        return 0.0

    s = sum(math.sin(a) for a in angles_rad)
    c = sum(math.cos(a) for a in angles_rad)
    return math.atan2(s, c)

def bearing_is_axis_aligned(angle_deg: float, tol_deg: float = ENDPOINT_AXIS_TOL_DEG) -> bool:
    """
    True if a LiDAR beam direction is close to 0/90/180/270 degrees.
    This rejects diagonal endpoint hits from open doors and random clutter.
    """
    a = angle_deg % 90.0
    return min(a, 90.0 - a) <= tol_deg

def timestamp_string() -> str:
    """Create a compact timestamp string for filenames."""
    return time.strftime("%Y%m%d_%H%M%S")


def logodds_to_probability(log_odds: np.ndarray) -> np.ndarray:
    """Convert log-odds values to probabilities in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-log_odds))


def world_to_map_xy(x_m: float, y_m: float, origin_x_m: float, origin_y_m: float, resolution_m: float):
    """
    Convert world coordinates to map-cell coordinates in a bottom-left-origin map frame.

    Returned gx, gy are integer grid coordinates where:
    - gx increases to the right
    - gy increases upward
    """
    gx = int(math.floor((x_m - origin_x_m) / resolution_m))
    gy = int(math.floor((y_m - origin_y_m) / resolution_m))
    return gx, gy


def map_xy_to_row_col(gx: int, gy: int, height_cells: int):
    """
    Convert bottom-left-origin map coordinates to NumPy image indices.

    NumPy image rows grow downward, so row is flipped from gy.
    """
    row = height_cells - 1 - gy
    col = gx
    return row, col


def bresenham_line(x0: int, y0: int, x1: int, y1: int):
    """
    Integer Bresenham line from (x0, y0) to (x1, y1), inclusive.

    Returns a list of (x, y) map-cell coordinates in bottom-left-origin map space.
    """
    points = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    x = x0
    y = y0

    while True:
        points.append((x, y))

        if x == x1 and y == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x += sx

        if e2 < dx:
            err += dx
            y += sy

    return points


# ----------------------------------------------------------------------------
# Robot state and mapper
# ----------------------------------------------------------------------------
@dataclass
class RobotPose:
    """Simple pose container."""
    x_m: float = 0.0
    y_m: float = 0.0
    theta_rad: float = 0.0


class OccupancyMapper:
    def __init__(self):
        # Create the output folder early so saving works immediately.
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        # Store map geometry.
        self.resolution_m = MAP_RESOLUTION_M
        self.width_cells = int(round(MAP_SIZE_X_M / MAP_RESOLUTION_M))
        self.height_cells = int(round(MAP_SIZE_Y_M / MAP_RESOLUTION_M))

        # Put the initial robot pose at the center of the map by making the map
        # origin the bottom-left corner of a centered rectangle.
        self.origin_x_m = -0.5 * MAP_SIZE_X_M
        self.origin_y_m = -0.5 * MAP_SIZE_Y_M

        # The main log-odds grid starts at zero = unknown.
        self.log_odds = np.zeros((self.height_cells, self.width_cells), dtype=np.float32)

        # Keep a robot pose and the previous encoder count for odometry updates.
        self.pose = RobotPose()
        self.prev_encoder = None

        # Keep a short path history for visualization.
        self.path_world = []

        # Count packets so we can throttle display if needed.
        self.packet_count = 0
        self.last_packet_steering_cmd = 0.0
        self.last_odom_steering_cmd = 0.0

        # Debug / correction state
        self.last_scan_match_score = None
        self.last_scan_match_used = False
        self.last_corridor_axis_deg = None
        self.last_snap_applied = False
        self.last_scan_match_theta_correction_deg = 0.0
        self.last_integrated_scan = False

        # Lean mapper state machine
        self.mode = MODE_STRAIGHT
        self.last_locked_axis_rad = 0.0
        self.pre_turn_axis_rad = None
        self.relock_target_axis_rad = None
        self.relock_stable_count = 0
        self.turn_direction = 0.0

        # AprilTag landmark observations.
        # Format:
        # {
        #   "3": [
        #       {"x": ..., "y": ..., "approach_theta_rad": ..., "z_cam": ..., "timestamp": ...},
        #       ...
        #   ]
        # }
        self.tag_observations = {}
        self.last_seen_tags = []
        self._last_tag_status_write_s = 0.0
        self.mapping_start_s = time.time()

    def reset(self):
        """Clear the map and reset odometry to the origin."""
        self.log_odds.fill(0.0)
        self.pose = RobotPose()
        self.prev_encoder = None
        self.path_world = []
        self.packet_count = 0
        self.last_packet_steering_cmd = 0.0
        self.last_scan_match_score = None
        self.last_scan_match_used = False
        self.last_odom_steering_cmd = 0.0
        self.last_corridor_axis_deg = None
        self.last_snap_applied = False
        self.last_scan_match_theta_correction_deg = 0.0
        self.last_integrated_scan = False
        self.mode = MODE_STRAIGHT
        self.last_locked_axis_rad = 0.0
        self.pre_turn_axis_rad = None
        self.relock_target_axis_rad = None
        self.relock_stable_count = 0
        self.turn_direction = 0.0
        self.tag_observations = {}
        self.last_seen_tags = []
        self._last_tag_status_write_s = 0.0
        self.mapping_start_s = time.time()

    def compute_path_length_m(self) -> float:
        """
        Approximate driven path length from stored mapper poses.
        """
        if len(self.path_world) < 2:
            return 0.0

        total = 0.0
        for (x0, y0), (x1, y1) in zip(self.path_world[:-1], self.path_world[1:]):
            total += math.hypot(x1 - x0, y1 - y0)
        return float(total)

    def compute_mapping_metrics(self):
        """
        Compute mapping/debug metrics when saving.

        The loss is not a training loss; it is a simple mapping-quality score
        where lower is better.
        """
        now = time.time()
        duration_s = max(1e-6, now - self.mapping_start_s)

        prob = logodds_to_probability(self.log_odds)

        free_mask = prob < 0.35
        occupied_mask = prob > 0.65
        known_mask = free_mask | occupied_mask
        unknown_mask = ~known_mask

        cell_area_m2 = self.resolution_m * self.resolution_m
        total_cells = int(prob.size)
        free_cells = int(np.count_nonzero(free_mask))
        occupied_cells = int(np.count_nonzero(occupied_mask))
        known_cells = int(np.count_nonzero(known_mask))
        unknown_cells = int(np.count_nonzero(unknown_mask))

        explored_area_m2 = known_cells * cell_area_m2
        free_area_m2 = free_cells * cell_area_m2
        occupied_area_m2 = occupied_cells * cell_area_m2
        unknown_area_m2 = unknown_cells * cell_area_m2
        known_fraction = known_cells / max(1, total_cells)

        path_length_m = self.compute_path_length_m()
        packets_per_second = self.packet_count / duration_s

        tag_summary = self.get_tag_summary(include_unconfirmed=True)
        confirmed_tags = {
            tag_id: tag for tag_id, tag in tag_summary.items()
            if tag.get("confirmed", False)
        }

        tag_spreads = []
        tag_obs_counts = []
        for tag in tag_summary.values():
            sx = float(tag.get("spread_x_m", 0.0))
            sy = float(tag.get("spread_y_m", 0.0))
            tag_spreads.append(math.hypot(sx, sy))
            tag_obs_counts.append(int(tag.get("num_observations", 0)))

        mean_tag_spread_m = float(np.mean(tag_spreads)) if tag_spreads else None
        max_tag_spread_m = float(np.max(tag_spreads)) if tag_spreads else None
        mean_tag_observations = float(np.mean(tag_obs_counts)) if tag_obs_counts else 0.0

        # Simple loss terms. Lower is better.
        # These make intuitive sense for reporting:
        # - less unknown area is better
        # - lower tag spread is better
        # - fewer unconfirmed tags is better
        # - more confirmed tags is better
        unknown_fraction = unknown_cells / max(1, total_cells)
        unconfirmed_tag_count = len(tag_summary) - len(confirmed_tags)
        confirmed_tag_count = len(confirmed_tags)

        unknown_area_loss = 10.0 * unknown_fraction
        tag_spread_loss = 0.0 if mean_tag_spread_m is None else 20.0 * mean_tag_spread_m
        unconfirmed_tag_loss = 2.0 * unconfirmed_tag_count
        confirmed_tag_bonus = -1.0 * confirmed_tag_count
        mapping_loss = (
            unknown_area_loss
            + tag_spread_loss
            + unconfirmed_tag_loss
            + confirmed_tag_bonus
        )

        return {
            "duration_s": duration_s,
            "packets_processed": int(self.packet_count),
            "packets_per_second": float(packets_per_second),
            "path_length_m": float(path_length_m),
            "map_width_cells": int(self.width_cells),
            "map_height_cells": int(self.height_cells),
            "map_resolution_m": float(self.resolution_m),
            "known_fraction": float(known_fraction),
            "unknown_fraction": float(unknown_fraction),
            "explored_area_m2": float(explored_area_m2),
            "free_area_m2": float(free_area_m2),
            "occupied_area_m2": float(occupied_area_m2),
            "unknown_area_m2": float(unknown_area_m2),
            "total_tags_seen": int(len(tag_summary)),
            "confirmed_tags": int(confirmed_tag_count),
            "unconfirmed_tags": int(unconfirmed_tag_count),
            "mean_tag_observations": float(mean_tag_observations),
            "mean_tag_position_spread_m": mean_tag_spread_m,
            "max_tag_position_spread_m": max_tag_spread_m,
            "loss_terms": {
                "unknown_area_loss": float(unknown_area_loss),
                "tag_spread_loss": float(tag_spread_loss),
                "unconfirmed_tag_loss": float(unconfirmed_tag_loss),
                "confirmed_tag_bonus": float(confirmed_tag_bonus),
            },
            "mapping_loss": float(mapping_loss),
        }

    def save_mapping_metrics_file(self, metrics_path: Path, stem: str):
        """
        Save and print mapping metrics when the user presses 's'.
        """
        metrics = self.compute_mapping_metrics()

        payload = {
            "map_name": stem,
            "timestamp": time.time(),
            "metrics": metrics,
            "tag_summary": self.get_tag_summary(include_unconfirmed=True),
        }

        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("\n================ MAPPING METRICS ================")
        print(f"Duration: {metrics['duration_s']:.2f} s")
        print(f"Packets processed: {metrics['packets_processed']}")
        print(f"Packets/sec: {metrics['packets_per_second']:.2f}")
        print(f"Path length: {metrics['path_length_m']:.2f} m")
        print(f"Known map fraction: {100.0 * metrics['known_fraction']:.2f}%")
        print(f"Explored area: {metrics['explored_area_m2']:.2f} m^2")
        print(f"Free area: {metrics['free_area_m2']:.2f} m^2")
        print(f"Occupied area: {metrics['occupied_area_m2']:.2f} m^2")
        print(f"Total tags seen: {metrics['total_tags_seen']}")
        print(f"Confirmed tags: {metrics['confirmed_tags']}")
        print(f"Unconfirmed tags: {metrics['unconfirmed_tags']}")
        print(f"Mean tag observations: {metrics['mean_tag_observations']:.2f}")

        if metrics["mean_tag_position_spread_m"] is not None:
            print(f"Mean tag position spread: {metrics['mean_tag_position_spread_m']:.3f} m")
            print(f"Max tag position spread: {metrics['max_tag_position_spread_m']:.3f} m")
        else:
            print("Mean tag position spread: N/A")
            print("Max tag position spread: N/A")

        print("Loss terms:")
        for key, value in metrics["loss_terms"].items():
            print(f"  {key}: {value:.3f}")
        print(f"Mapping loss: {metrics['mapping_loss']:.3f}")
        print(f"Saved mapping metrics: {metrics_path}")
        print("=================================================\n")

        return metrics

    def add_tag_observation(self, tag_id: int, x_m: float, y_m: float,
                            approach_theta_rad: float, z_cam_m: float,
                            tag_facing_theta_rad: float | None = None):
        """
        Store one AprilTag observation in map/world coordinates.

        We only record tags while the mapper is in STRAIGHT mode because
        pose is most reliable there. During turning/relock, detections
        may be visually valid but the robot pose is less trustworthy.
        """
        if self.mode != MODE_STRAIGHT:
            return

        tag_key = str(int(tag_id))
        obs = self.tag_observations.setdefault(tag_key, [])

        obs.append({
            "x": float(x_m),
            "y": float(y_m),
            "approach_theta_rad": float(angle_wrap(approach_theta_rad)),
            "tag_facing_theta_rad": (
                None if tag_facing_theta_rad is None
                else float(angle_wrap(tag_facing_theta_rad))
            ),
            "z_cam_m": float(z_cam_m),
            "robot_x": float(self.pose.x_m),
            "robot_y": float(self.pose.y_m),
            "robot_theta_rad": float(self.pose.theta_rad),
            "timestamp": float(time.time()),
        })

        # Keep memory bounded.
        if len(obs) > APRILTAG_MAX_OBS_PER_ID:
            self.tag_observations[tag_key] = obs[-APRILTAG_MAX_OBS_PER_ID:]

    def get_tag_summary(self, include_unconfirmed: bool = True):
        """
        Convert raw tag observations into stable landmark estimates.

        x/y use median for robustness. Heading uses circular mean.
        """
        tags = {}

        for tag_key, obs in self.tag_observations.items():
            if not obs:
                continue

            n = len(obs)
            if (not include_unconfirmed) and n < APRILTAG_MIN_OBSERVATIONS_TO_SAVE:
                continue

            xs = np.array([o["x"] for o in obs], dtype=np.float32)
            ys = np.array([o["y"] for o in obs], dtype=np.float32)
            approach_headings = [o["approach_theta_rad"] for o in obs]
            facing_headings = [
                o["tag_facing_theta_rad"]
                for o in obs
                if o.get("tag_facing_theta_rad") is not None
            ]
            z_vals = np.array([o["z_cam_m"] for o in obs], dtype=np.float32)

            x_med = float(np.median(xs))
            y_med = float(np.median(ys))
            approach_theta_rad = circular_mean_rad(approach_headings)
            tag_facing_theta_rad = (
                circular_mean_rad(facing_headings)
                if facing_headings
                else approach_theta_rad
            )

            tags[tag_key] = {
                # Navigation script uses x_m/y_m/theta_rad directly.
                "x_m": x_med,
                "y_m": y_med,
                "theta_rad": float(approach_theta_rad),
                "theta_deg": float(math.degrees(approach_theta_rad)),

                # Backward-compatible / display fields.
                "x": x_med,
                "y": y_med,
                "approach_theta_rad": float(approach_theta_rad),
                "approach_theta_deg": float(math.degrees(approach_theta_rad)),
                "tag_facing_theta_rad": float(tag_facing_theta_rad),
                "tag_facing_theta_deg": float(math.degrees(tag_facing_theta_rad)),
                "num_observations": int(n),
                "confirmed": bool(n >= APRILTAG_MIN_OBSERVATIONS_TO_SAVE),
                "spread_x_m": float(np.std(xs)),
                "spread_y_m": float(np.std(ys)),
                "median_camera_z_m": float(np.median(z_vals)),
                "stop_distance_m": 0.55,
            }

        return tags

    def write_live_tag_status(self):
        """
        Write lightweight live tag status for the NiceGUI teleop panel.
        """
        now = time.time()
        if now - self._last_tag_status_write_s < APRILTAG_STATUS_WRITE_EVERY_S:
            return

        self._last_tag_status_write_s = now
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        payload = {
            "timestamp": now,
            "pose": {
                "x": float(self.pose.x_m),
                "y": float(self.pose.y_m),
                "theta_deg": float(math.degrees(self.pose.theta_rad)),
            },
            "last_seen_tags": self.last_seen_tags,
            "tags": self.get_tag_summary(include_unconfirmed=True),
        }

        try:
            LIVE_TAG_STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"[WARN] Could not write live tag status: {exc}")

    def save_tag_map_file(self, tag_map_path: Path, stem: str):
        """
        Save AprilTag landmarks beside the occupancy map.
        """
        tags = self.get_tag_summary(include_unconfirmed=False)

        payload = {
            "map_name": stem,
            "coordinate_frame": "same world/map frame as the saved PGM/YAML map",
            "min_observations_required": APRILTAG_MIN_OBSERVATIONS_TO_SAVE,
            "camera_to_robot_assumption": {
                "camera_x_m": CAMERA_X_M,
                "camera_y_m": CAMERA_Y_M,
                "camera_yaw_offset_deg": CAMERA_YAW_OFFSET_DEG,
                "opencv_camera_frame": "x right, y down, z forward",
                "robot_frame": "x forward, y left",
                "theta_rad_meaning": "desired robot approach heading in map/world frame",
                "tag_facing_theta_rad_meaning": "estimated AprilTag plane normal heading in map/world frame",
            },
            "tags": tags,
        }

        tag_map_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return len(tags)

    def expand_map_if_needed(self, x_m: float, y_m: float, margin_m: float = MAP_GROW_MARGIN_M):
        """
        Grow the map if the given world point is too close to any border.

        We pad the log-odds grid and update the world-frame origin accordingly.
        """
        margin_cells = int(math.ceil(margin_m / self.resolution_m))

        gx, gy = world_to_map_xy(
            x_m,
            y_m,
            self.origin_x_m,
            self.origin_y_m,
            self.resolution_m,
        )

        grow_left = max(0, margin_cells - gx)
        grow_right = max(0, gx - (self.width_cells - 1 - margin_cells))
        grow_bottom = max(0, margin_cells - gy)
        grow_top = max(0, gy - (self.height_cells - 1 - margin_cells))

        if grow_left == 0 and grow_right == 0 and grow_bottom == 0 and grow_top == 0:
            return

        self.log_odds = np.pad(
            self.log_odds,
            ((grow_top, grow_bottom), (grow_left, grow_right)),
            mode="constant",
            constant_values=0.0,
        )

        # Expanding left/bottom changes the world-frame origin.
        self.origin_x_m -= grow_left * self.resolution_m
        self.origin_y_m -= grow_bottom * self.resolution_m

        self.height_cells, self.width_cells = self.log_odds.shape

    def in_bounds_xy(self, gx: int, gy: int) -> bool:
        """Check whether a map-cell coordinate is inside the grid."""
        return 0 <= gx < self.width_cells and 0 <= gy < self.height_cells

    def add_log_odds_xy(self, gx: int, gy: int, delta: float):
        """
        Update one map cell in log-odds form.

        We clip values so cells do not grow without bound after many updates.
        """
        if not self.in_bounds_xy(gx, gy):
            return

        row, col = map_xy_to_row_col(gx, gy, self.height_cells)
        self.log_odds[row, col] = np.clip(
            self.log_odds[row, col] + delta,
            LOG_ODDS_MIN,
            LOG_ODDS_MAX,
        )

    def scan_to_points_robot(self, distances_mm):
        """
        Convert one scan to XY points in the robot/base frame.
        These are used for scan matching and corridor-axis estimation.
        """
        bins = len(distances_mm)
        if bins <= 0:
            return np.empty((0, 2), dtype=np.float32)

        angle_step_deg = 360.0 / bins
        pts = []

        for raw_idx, d_mm in enumerate(distances_mm):
            if d_mm <= 0:
                continue

            r_m = d_mm / 1000.0
            if r_m < LIDAR_RANGE_MIN_M or r_m > min(LIDAR_RANGE_MAX_M, SCAN_MATCH_MAX_RANGE_M):
                continue

            if FLIP_SCAN:
                logical_idx = (bins - raw_idx) % bins
            else:
                logical_idx = raw_idx

            beam_angle_deg = logical_idx * angle_step_deg + ANGLE_ZERO_OFFSET_DEG
            beam_angle_rad = math.radians(beam_angle_deg)

            # Store in robot frame, not world frame.
            x_r = LASER_X_M + r_m * math.cos(beam_angle_rad)
            y_r = LASER_Y_M + r_m * math.sin(beam_angle_rad)
            pts.append((x_r, y_r))

        if not pts:
            return np.empty((0, 2), dtype=np.float32)

        pts = np.asarray(pts, dtype=np.float32)
        if SCAN_MATCH_SUBSAMPLE > 1:
            pts = pts[::SCAN_MATCH_SUBSAMPLE]
        return pts

    def raw_scan_index_to_robot_angle_deg(self, raw_idx: int, bins: int) -> float:
        """
        Convert a raw scan index into a robot-frame angle in degrees,
        using the same flip/offset convention as the map integration code.
        """
        angle_step_deg = 360.0 / bins
        if FLIP_SCAN:
            logical_idx = (bins - raw_idx) % bins
        else:
            logical_idx = raw_idx
        return angle_deg_wrap_360(logical_idx * angle_step_deg + ANGLE_ZERO_OFFSET_DEG)

    def average_scan_range_m(self, distances_mm, target_deg: float, half_width_deg: float,
                             min_range_m: float, max_range_m: float):
        """
        Average valid ranges around a target bearing in robot frame.
        Returns None if no valid points are found in that window.
        """
        bins = len(distances_mm)
        sum_m = 0.0
        count = 0

        for raw_idx, d_mm in enumerate(distances_mm):
            if d_mm <= 0:
                continue

            r_m = d_mm / 1000.0
            if r_m < min_range_m or r_m > max_range_m:
                continue

            a_deg = self.raw_scan_index_to_robot_angle_deg(raw_idx, bins)
            delta = ((a_deg - target_deg + 180.0) % 360.0) - 180.0
            if abs(delta) <= half_width_deg:
                sum_m += r_m
                count += 1

        if count == 0:
            return None
        return sum_m / count

    def estimate_corridor_width_m(self, distances_mm):
        """
        Estimate corridor width from left/right side windows.
        Returns (width_m_or_None, width_ok_bool).
        """
        left_m = self.average_scan_range_m(
            distances_mm,
            target_deg=90.0,
            half_width_deg=CORRIDOR_SIDE_HALF_WIDTH_DEG,
            min_range_m=CORRIDOR_SIDE_MIN_RANGE_M,
            max_range_m=CORRIDOR_SIDE_MAX_RANGE_M,
        )
        right_m = self.average_scan_range_m(
            distances_mm,
            target_deg=270.0,
            half_width_deg=CORRIDOR_SIDE_HALF_WIDTH_DEG,
            min_range_m=CORRIDOR_SIDE_MIN_RANGE_M,
            max_range_m=CORRIDOR_SIDE_MAX_RANGE_M,
        )

        if left_m is None or right_m is None:
            return None, False

        width_m = left_m + right_m
        width_ok = CORRIDOR_WIDTH_MIN_M <= width_m <= CORRIDOR_WIDTH_MAX_M
        return width_m, width_ok

    def transform_points_to_world(self, pts_robot: np.ndarray, pose: RobotPose | None = None):
        """Transform robot-frame XY points into world-frame XY points."""
        if pts_robot.size == 0:
            return np.empty((0, 2), dtype=np.float32)

        if pose is None:
            pose = self.pose

        c = math.cos(pose.theta_rad)
        s = math.sin(pose.theta_rad)

        xw = pose.x_m + c * pts_robot[:, 0] - s * pts_robot[:, 1]
        yw = pose.y_m + s * pts_robot[:, 0] + c * pts_robot[:, 1]
        return np.column_stack((xw, yw)).astype(np.float32)

    def world_points_to_rows_cols(self, pts_world: np.ndarray):
        """Convert world XY points into image row/col indices."""
        if pts_world.size == 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=bool),
            )

        gx = np.floor((pts_world[:, 0] - self.origin_x_m) / self.resolution_m).astype(np.int32)
        gy = np.floor((pts_world[:, 1] - self.origin_y_m) / self.resolution_m).astype(np.int32)

        valid = (
            (gx >= 0) & (gx < self.width_cells) &
            (gy >= 0) & (gy < self.height_cells)
        )

        rows = self.height_cells - 1 - gy
        cols = gx
        return rows, cols, valid

    def build_scan_match_distance_map(self):
        """
        Build a distance-to-occupied-cell image from the current map.
        Lower distance = better alignment target.
        """
        prob = logodds_to_probability(self.log_odds)
        occ = (prob > SCAN_MATCH_OCC_THRESH).astype(np.uint8)

        if int(occ.sum()) < 40:
            return None

        # distanceTransform wants non-zero as traversable and zero as obstacle.
        traversable = 1 - occ
        dist_px = cv2.distanceTransform(traversable, cv2.DIST_L2, 3)
        return dist_px * self.resolution_m

    def score_pose_candidate(self, pts_robot: np.ndarray, dist_map: np.ndarray, pose: RobotPose) -> float:
        """Lower mean distance to occupied structure => better score."""
        pts_world = self.transform_points_to_world(pts_robot, pose)
        rows, cols, valid = self.world_points_to_rows_cols(pts_world)

        valid_count = int(valid.sum())
        if valid_count < max(10, len(pts_robot) // 4):
            return -1e9

        d = dist_map[rows[valid], cols[valid]]
        mean_dist = float(np.mean(d))
        coverage_bonus = valid_count / max(1, len(pts_robot))

        # Bigger is better.
        return -mean_dist + 0.10 * coverage_bonus

    def local_correlative_search(
        self,
        pts_robot: np.ndarray,
        dist_map: np.ndarray,
        center_pose: RobotPose,
        xy_search_m: float,
        xy_step_m: float,
        th_search_deg: float,
        th_step_deg: float,
    ):
        """Search around an odometry seed pose for the best LiDAR/map alignment."""
        best_pose = RobotPose(center_pose.x_m, center_pose.y_m, center_pose.theta_rad)
        best_score = self.score_pose_candidate(pts_robot, dist_map, best_pose)

        dx_vals = np.arange(-xy_search_m, xy_search_m + 1e-9, xy_step_m)
        dy_vals = np.arange(-xy_search_m, xy_search_m + 1e-9, xy_step_m)
        dth_vals = np.radians(np.arange(-th_search_deg, th_search_deg + 1e-9, th_step_deg))

        for dth in dth_vals:
            for dx in dx_vals:
                for dy in dy_vals:
                    pose = RobotPose(
                        center_pose.x_m + float(dx),
                        center_pose.y_m + float(dy),
                        angle_wrap(center_pose.theta_rad + float(dth)),
                    )
                    score = self.score_pose_candidate(pts_robot, dist_map, pose)
                    if score > best_score:
                        best_score = score
                        best_pose = pose

        return best_pose, best_score

    def refine_pose_with_scan_match(self, pts_robot: np.ndarray, raw_steering_cmd: float):
        """
        Use odometry as an initial guess, then align the scan to the current map.
        """
        self.last_scan_match_used = False
        self.last_scan_match_score = None
        self.last_scan_match_theta_correction_deg = 0.0

        if not SCAN_MATCH_ENABLE or len(pts_robot) < SCAN_MATCH_MIN_POINTS:
            return

        dist_map = self.build_scan_match_distance_map()
        if dist_map is None:
            return

        # If we are already turning, or if the current steering command is far
        # from nominal straight, widen the angular search.
        turning_hint = abs(raw_steering_cmd + STEERING_OFFSET_CMD) >= TURN_STEER_TRIGGER_CMD
        turning_mode = (self.mode == MODE_TURNING) or (self.mode == MODE_RELOCK) or turning_hint

        if turning_mode:
            coarse_xy_search_m = TURN_SCAN_MATCH_COARSE_XY_SEARCH_M
            coarse_xy_step_m = TURN_SCAN_MATCH_COARSE_XY_STEP_M
            coarse_theta_search_deg = TURN_SCAN_MATCH_COARSE_THETA_SEARCH_DEG
            coarse_theta_step_deg = TURN_SCAN_MATCH_COARSE_THETA_STEP_DEG
            fine_xy_search_m = TURN_SCAN_MATCH_FINE_XY_SEARCH_M
            fine_xy_step_m = TURN_SCAN_MATCH_FINE_XY_STEP_M
            fine_theta_search_deg = TURN_SCAN_MATCH_FINE_THETA_SEARCH_DEG
            fine_theta_step_deg = TURN_SCAN_MATCH_FINE_THETA_STEP_DEG
        else:
            coarse_xy_search_m = SCAN_MATCH_COARSE_XY_SEARCH_M
            coarse_xy_step_m = SCAN_MATCH_COARSE_XY_STEP_M
            coarse_theta_search_deg = SCAN_MATCH_COARSE_THETA_SEARCH_DEG
            coarse_theta_step_deg = SCAN_MATCH_COARSE_THETA_STEP_DEG
            fine_xy_search_m = SCAN_MATCH_FINE_XY_SEARCH_M
            fine_xy_step_m = SCAN_MATCH_FINE_XY_STEP_M
            fine_theta_search_deg = SCAN_MATCH_FINE_THETA_SEARCH_DEG
            fine_theta_step_deg = SCAN_MATCH_FINE_THETA_STEP_DEG

        odom_seed_theta = self.pose.theta_rad

        coarse_pose, coarse_score = self.local_correlative_search(
            pts_robot,
            dist_map,
            self.pose,
            coarse_xy_search_m,
            coarse_xy_step_m,
            coarse_theta_search_deg,
            coarse_theta_step_deg,
        )

        fine_pose, fine_score = self.local_correlative_search(
            pts_robot,
            dist_map,
            coarse_pose,
            fine_xy_search_m,
            fine_xy_step_m,
            fine_theta_search_deg,
            fine_theta_step_deg,
        )

        self.pose = fine_pose
        self.last_scan_match_score = fine_score
        self.last_scan_match_used = True
        self.last_scan_match_theta_correction_deg = math.degrees(
            angle_wrap(self.pose.theta_rad - odom_seed_theta)
        )

        # Update the last path point so the rendered trajectory uses the
        # scan-matched pose, not the raw odometry-only pose.
        if self.path_world:
            self.path_world[-1] = (self.pose.x_m, self.pose.y_m)

    def fit_line_angle_from_points(self, pts: np.ndarray):
        """Fit an undirected line angle with PCA. Return (angle_rad, linearity)."""
        if len(pts) < WALL_FIT_MIN_POINTS:
            return None

        centered = pts - np.mean(pts, axis=0, keepdims=True)
        try:
            _, s, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        direction = vh[0]
        angle = math.atan2(float(direction[1]), float(direction[0]))
        angle = line_angle_normalize(angle)

        # Smaller ratio => more line-like
        linearity = float(s[1] / max(s[0], 1e-6))
        return angle, linearity

    def estimate_corridor_axis_world_with_info(self, pts_robot: np.ndarray):
        """
        Estimate hallway axis from left/right wall points in robot frame.
        Returns:
        - world-frame corridor axis angle in radians, or None if weak
        - number of accepted wall fits used
        """
        if len(pts_robot) < WALL_FIT_MIN_POINTS:
            return None, 0

        angles_deg = np.degrees(np.arctan2(pts_robot[:, 1], pts_robot[:, 0]))
        angles_deg = np.mod(angles_deg, 360.0)
        ranges_m = np.linalg.norm(pts_robot, axis=1)

        left_mask = (
            (angles_deg >= LEFT_WALL_MIN_DEG) &
            (angles_deg <= LEFT_WALL_MAX_DEG) &
            (ranges_m <= WALL_FIT_MAX_RANGE_M)
        )
        right_mask = (
            (angles_deg >= RIGHT_WALL_MIN_DEG) &
            (angles_deg <= RIGHT_WALL_MAX_DEG) &
            (ranges_m <= WALL_FIT_MAX_RANGE_M)
        )

        wall_angles = []
        for mask in (left_mask, right_mask):
            fit = self.fit_line_angle_from_points(pts_robot[mask])
            if fit is None:
                continue
            angle_robot, linearity = fit
            if linearity < 0.35:
                wall_angles.append(angle_robot)

        if not wall_angles:
            return None, 0

        # Average line directions modulo pi
        c2 = sum(math.cos(2.0 * a) for a in wall_angles)
        s2 = sum(math.sin(2.0 * a) for a in wall_angles)
        axis_robot = 0.5 * math.atan2(s2, c2)
        axis_robot = line_angle_normalize(axis_robot)

        axis_world = angle_wrap(self.pose.theta_rad + axis_robot)
        return axis_world, len(wall_angles)

    def estimate_corridor_axis_world(self, pts_robot: np.ndarray):
        """Backward-compatible thin wrapper."""
        axis_world, _wall_count = self.estimate_corridor_axis_world_with_info(pts_robot)
        return axis_world

    def apply_manhattan_snap_from_axis(self, axis_world: float):
        """
        Record/display the observed corridor axis, but do NOT continuously rotate
        the robot pose from noisy wall fits. That was causing th=-20 deg while
        the robot was physically moving straight.
        """
        if not MANHATTAN_ENABLE:
            return
        
        self.last_corridor_axis_deg = math.degrees(axis_world)

        # Straight heading is locked separately in update_odometry().
        self.last_snap_applied = False

    def choose_relock_target_axis(self, measured_axis_rad: float):
        """
        During RELOCK, choose whether the new corridor should align with:
        - the old axis
        - the old axis + 90 deg
        - the old axis - 90 deg

        Returns the snapped target axis or None if nothing matches closely enough.
        """
        if self.pre_turn_axis_rad is None:
            return measured_axis_rad

        candidates = [
            line_angle_normalize(self.pre_turn_axis_rad),
            line_angle_normalize(self.pre_turn_axis_rad + math.pi / 2.0),
            line_angle_normalize(self.pre_turn_axis_rad - math.pi / 2.0),
        ]

        best_axis = None
        best_err = 1e9
        for cand in candidates:
            err = line_angle_diff_abs(measured_axis_rad, cand)
            if err < best_err:
                best_err = err
                best_axis = cand

        if math.degrees(best_err) <= RELOCK_AXIS_MATCH_THRESH_DEG:
            return best_axis
        return None

    def snap_pose_heading_to_axis(self, measured_axis_rad: float, target_axis_rad: float):
        """
        Rotate the robot pose so the measured corridor axis lands exactly on
        the chosen target axis.
        """
        delta = line_angle_normalize(target_axis_rad - measured_axis_rad)
        self.pose.theta_rad = angle_wrap(self.pose.theta_rad + delta)
        self.last_snap_applied = True

        # Keep displayed path consistent with the corrected pose.
        if self.path_world:
            self.path_world[-1] = (self.pose.x_m, self.pose.y_m)

    def update_mapper_state(self, scan_points_robot: np.ndarray, distances_mm, raw_steering_cmd: float):
        """
        Lean STRAIGHT / TURNING / RELOCK state machine.

        STRAIGHT:
        - corridor axis visible
        - use Manhattan snap
        - use normal scan matching

        TURNING:
        - corridor breaks or steering is clearly non-straight
        - widen scan-match search
        - do not force corridor-axis snap here

        RELOCK:
        - corridor comes back
        - wait a few scans for stable axis
        - snap to old axis or old axis +/- 90 deg
        - return to STRAIGHT
        """
        self.last_snap_applied = False

        measured_axis_rad, wall_count = self.estimate_corridor_axis_world_with_info(scan_points_robot)
        self.last_corridor_axis_deg = None if measured_axis_rad is None else math.degrees(measured_axis_rad)

        axis_visible = measured_axis_rad is not None

        # Important:
        # Only the steering command should decide if we are intentionally turning.
        # Weak corridor estimates should NOT force TURNING, because that bypasses
        # Manhattan correction during normal straight driving.
        turning_hint = abs(raw_steering_cmd + STEERING_OFFSET_CMD) >= TURN_STEER_TRIGGER_CMD

        if self.mode == MODE_STRAIGHT:
            # If we are not intentionally turning and we can see a corridor axis,
            # only display it. Do NOT update last_locked_axis_rad from noisy
            # wall fitting, otherwise the robot heading drifts during straight motion.
            if (not turning_hint) and axis_visible:
                self.apply_manhattan_snap_from_axis(measured_axis_rad)

            # Enter TURNING only for intentional steering turns.
            if turning_hint:
                self.mode = MODE_TURNING
                self.pre_turn_axis_rad = self.last_locked_axis_rad
                turn_cmd = raw_steering_cmd + STEERING_OFFSET_CMD
                self.turn_direction = 1.0 if turn_cmd > 0 else -1.0
                self.relock_target_axis_rad = None
                self.relock_stable_count = 0
                return

        elif self.mode == MODE_TURNING:
            # Once steering comes back near straight and a hallway axis is visible,
            # begin relocking to old axis or old axis +/- 90 degrees.
            if (not turning_hint) and axis_visible:
                # For this robot, odometry underestimates 90-degree turns badly.
                # So after an intentional turn, force the next corridor to be
                # orthogonal to the previous locked corridor.
                if FORCE_90_RELOCK_AFTER_TURN and self.pre_turn_axis_rad is not None:
                    target_axis_rad = nearest_manhattan_heading(
                        self.pre_turn_axis_rad
                        + TURN_DIRECTION_SIGN * self.turn_direction * (math.pi / 2.0)
                    )

                    # Hard set the robot heading to the new perpendicular hallway.
                    # Do not use measured_axis_rad here because door clutter/noisy wall
                    # fits can make the new hallway look parallel.
                    self.pose.theta_rad = target_axis_rad
                    self.last_snap_applied = True
                    self.last_locked_axis_rad = target_axis_rad
                    self.mode = MODE_STRAIGHT
                    self.pre_turn_axis_rad = None
                    self.relock_target_axis_rad = None
                    self.relock_stable_count = 0
                    self.turn_direction = 0.0
                    return
                self.mode = MODE_RELOCK
                self.relock_target_axis_rad = self.choose_relock_target_axis(measured_axis_rad)
                self.relock_stable_count = 1 if self.relock_target_axis_rad is not None else 0
                return

        elif self.mode == MODE_RELOCK:
            if turning_hint:
                self.mode = MODE_TURNING
                self.relock_target_axis_rad = None
                self.relock_stable_count = 0
                return

            if measured_axis_rad is None:
                return

            target_axis_rad = self.choose_relock_target_axis(measured_axis_rad)
            if target_axis_rad is None:
                # Geometry came back, but not as a sane continuation of the old
                # corridor or its orthogonal branch. Keep turning/searching.
                self.mode = MODE_TURNING
                self.relock_target_axis_rad = None
                self.relock_stable_count = 0
                return

            if (
                self.relock_target_axis_rad is not None and
                line_angle_diff_abs(target_axis_rad, self.relock_target_axis_rad)
                <= math.radians(RELOCK_AXIS_JITTER_THRESH_DEG)
            ):
                self.relock_stable_count += 1
            else:
                self.relock_target_axis_rad = target_axis_rad
                self.relock_stable_count = 1

            if self.relock_stable_count >= RELOCK_REQUIRED_STABLE_SCANS:
                self.snap_pose_heading_to_axis(measured_axis_rad, target_axis_rad)
                self.last_locked_axis_rad = target_axis_rad
                self.mode = MODE_STRAIGHT
                self.pre_turn_axis_rad = None
                self.relock_target_axis_rad = None
                self.relock_stable_count = 0
                # After exact relock, also allow a small Manhattan clean-up.
                self.apply_manhattan_snap_from_axis(target_axis_rad)
                return


    def update_odometry(self, encoder_now: int, raw_steering_cmd: float):
        """
        Update pose using the same bicycle-model logic as your ROS UDP bridge.

        This assumes:
        - encoder_count is cumulative
        - steering command in the packet is the ACTUAL steering command
          applied on the robot (base teleop steer plus any corridor correction)
        """
        # On the very first packet, we only initialize the previous encoder
        # so we do not create a fake jump in odometry.
        if self.prev_encoder is None:
            self.prev_encoder = encoder_now
            self.path_world.append((self.pose.x_m, self.pose.y_m))
            return
        
        # If we are intentionally turning, do NOT let wheel/steering odometry
        # bend the map pose. The bicycle model is too unreliable here.
        #
        # We still consume the encoder value so that, when mapping resumes,
        # old turn ticks do not create a fake jump.
        if self.mode == MODE_TURNING or (
            self.mode == MODE_RELOCK and self.relock_stable_count < RELOCK_REQUIRED_STABLE_SCANS
        ):
            self.prev_encoder = encoder_now
            self.last_odom_steering_cmd = 0.0

            # Keep path fixed at the corner instead of drawing a curved turn.
            if self.path_world:
                self.path_world[-1] = (self.pose.x_m, self.pose.y_m)
            else:
                self.path_world.append((self.pose.x_m, self.pose.y_m))

            return
        
        # In straight mode, force the pose heading to the locked Manhattan heading.
        # This prevents th=-19 deg nonsense while steer_pkt=-7.
        if HARD_LOCK_STRAIGHT_HEADING and self.last_locked_axis_rad is not None:
            self.pose.theta_rad = nearest_manhattan_heading(self.last_locked_axis_rad)

        # Encoder delta since the previous packet.
        delta_ticks = encoder_now - self.prev_encoder
        self.prev_encoder = encoder_now

        # Convert encoder ticks to traveled distance in meters.
        delta_s_m = delta_ticks * METERS_PER_TICK

        # Convert the steering command into a wheel angle in radians.
        #
        # raw_steering_cmd is centered around -3 for "straight".
        # Small changes like -6, -8, -4 are often just trim corrections,
        # not intentional turns. If we integrate those as yaw, a straight
        # hallway becomes curved.
        steering_cmd_from_straight = raw_steering_cmd + STEERING_OFFSET_CMD

        if abs(steering_cmd_from_straight) <= STRAIGHT_STEER_DEADBAND_CMD:
            steering_cmd_from_straight = 0.0

        steering_cmd = STEERING_SIGN * STEERING_ODOM_SCALE * steering_cmd_from_straight
        self.last_odom_steering_cmd = steering_cmd
        steering_wheel_rad = math.radians(steering_cmd * STEERING_DEG_PER_CMD)

        # Bicycle-model heading change.
        if abs(steering_wheel_rad) < 1e-6:
            delta_theta = 0.0
        else:
            delta_theta = (delta_s_m / WHEELBASE_M) * math.tan(steering_wheel_rad)

        # Midpoint integration gives a better path than simply using the old heading.
        theta_mid = self.pose.theta_rad + 0.5 * delta_theta
        self.pose.x_m += delta_s_m * math.cos(theta_mid)
        self.pose.y_m += delta_s_m * math.sin(theta_mid)
        self.pose.theta_rad = angle_wrap(self.pose.theta_rad + delta_theta)

        # If steering is straight/noise, snap heading back exactly to the locked
        # hallway direction after movement.
        if HARD_LOCK_STRAIGHT_HEADING and abs(steering_cmd_from_straight) <= STRAIGHT_STEER_DEADBAND_CMD:
            self.pose.theta_rad = nearest_manhattan_heading(self.last_locked_axis_rad)

        self.expand_map_if_needed(
            self.pose.x_m,
            self.pose.y_m,
            margin_m=LIDAR_RANGE_MAX_M + 0.5,
        )

        self.path_world.append((self.pose.x_m, self.pose.y_m))

    def integrate_scan(self, distances_mm):
        """
        Fuse one full scan into the occupancy grid.

        For each valid beam:
        - mark cells along the ray as free
        - mark the end cell as occupied
        """
        bins = len(distances_mm)

        # Compute the laser position in the world frame from the robot base pose.
        cos_t = math.cos(self.pose.theta_rad)
        sin_t = math.sin(self.pose.theta_rad)

        sensor_x_m = self.pose.x_m + cos_t * LASER_X_M - sin_t * LASER_Y_M
        sensor_y_m = self.pose.y_m + sin_t * LASER_X_M + cos_t * LASER_Y_M

        # Grow the map if the sensor is getting close to the border.
        # Use lidar max range so beam endpoints also remain inside the map.
        self.expand_map_if_needed(
            sensor_x_m,
            sensor_y_m,
            margin_m=LIDAR_RANGE_MAX_M + 0.5,
        )

        # Convert the sensor origin after any map growth/origin shift.
        sensor_gx, sensor_gy = world_to_map_xy(
            sensor_x_m,
            sensor_y_m,
            self.origin_x_m,
            self.origin_y_m,
            self.resolution_m,
        )

        if not self.in_bounds_xy(sensor_gx, sensor_gy):
            return

        angle_step_deg = 360.0 / bins

        for raw_idx, d_mm in enumerate(distances_mm):
            # Skip missing / invalid bins directly.
            if d_mm <= 0:
                continue

            r_m = d_mm / 1000.0

            # Skip points that are outside the usable range window.
            if r_m < LIDAR_RANGE_MIN_M or r_m > LIDAR_RANGE_MAX_M:
                continue

            # Reorder the beam index if the scan must be mirrored.
            if FLIP_SCAN:
                logical_idx = (bins - raw_idx) % bins
            else:
                logical_idx = raw_idx

            # Beam angle in the robot frame, then rotated into the world frame.
            beam_angle_deg = logical_idx * angle_step_deg + ANGLE_ZERO_OFFSET_DEG
            beam_angle_world_rad = self.pose.theta_rad + math.radians(beam_angle_deg)

            # Use a shorter ray for free-space carving so long beams do not dominate the map.
            free_r_m = min(r_m, FREE_RAY_MAX_M)

            free_x_m = sensor_x_m + free_r_m * math.cos(beam_angle_world_rad)
            free_y_m = sensor_y_m + free_r_m * math.sin(beam_angle_world_rad)

            free_gx, free_gy = world_to_map_xy(
                free_x_m,
                free_y_m,
                self.origin_x_m,
                self.origin_y_m,
                self.resolution_m,
            )

            # Ray-trace from sensor cell to the shortened free-space endpoint.
            ray_cells = bresenham_line(sensor_gx, sensor_gy, free_gx, free_gy)

            if not ray_cells:
                continue

            # Mark every ray cell except the last one as free.
            for gx, gy in ray_cells[:-1]:
                self.add_log_odds_xy(gx, gy, LOG_ODDS_FREE)

            # Mark the true hit endpoint as occupied only for nearer, more trustworthy hits.
            hit_x_m = sensor_x_m + r_m * math.cos(beam_angle_world_rad)
            hit_y_m = sensor_y_m + r_m * math.sin(beam_angle_world_rad)
            end_gx, end_gy = world_to_map_xy(
                hit_x_m,
                hit_y_m,
                self.origin_x_m,
                self.origin_y_m,
                self.resolution_m,
            )
            if r_m <= OCCUPIED_ENDPOINT_MAX_M:
                # Do not allow diagonal endpoint hits to draw fake diagonal walls.
                # Free-space rays still carve space, but black occupied endpoints
                # are kept only near Manhattan beam directions.
                if FILTER_DIAGONAL_ENDPOINTS and not bearing_is_axis_aligned(beam_angle_deg):
                    continue

                self.add_log_odds_xy(end_gx, end_gy, LOG_ODDS_OCC)

    def render_map(self) -> np.ndarray:
        """
        Convert the log-odds grid into a display image.

        Display conventions:
        - free: white
        - occupied: black
        - unknown: gray
        """
        prob = logodds_to_probability(self.log_odds)

        # Start with unknown = gray.
        img = np.full((self.height_cells, self.width_cells, 3), 205, dtype=np.uint8)

        # Free cells -> white.
        free_mask = prob < 0.35
        img[free_mask] = (255, 255, 255)

        # Occupied cells -> black.
        occ_mask = prob > 0.65
        img[occ_mask] = (0, 0, 0)

        # Draw the robot path in blue-ish color for easier debugging.
        for x_m, y_m in self.path_world:
            gx, gy = world_to_map_xy(
                x_m, y_m, self.origin_x_m, self.origin_y_m, self.resolution_m
            )
            if self.in_bounds_xy(gx, gy):
                row, col = map_xy_to_row_col(gx, gy, self.height_cells)
                img[row, col] = (255, 180, 0)

        # Draw AprilTag landmarks in magenta.
        tag_summary = self.get_tag_summary(include_unconfirmed=True)
        for tag_id, tag in tag_summary.items():
            gx, gy = world_to_map_xy(
                tag["x"],
                tag["y"],
                self.origin_x_m,
                self.origin_y_m,
                self.resolution_m,
            )

            if self.in_bounds_xy(gx, gy):
                row, col = map_xy_to_row_col(gx, gy, self.height_cells)

                # Filled marker if confirmed; hollow marker if still tentative.
                color = (255, 0, 255)
                if tag["confirmed"]:
                    cv2.circle(img, (col, row), 5, color, -1)
                else:
                    cv2.circle(img, (col, row), 5, color, 1)

                theta = float(tag.get("theta_rad", tag.get("approach_theta_rad", 0.0)))
                arrow_len_cells = max(8, int(0.30 / self.resolution_m))
                tip_x = int(round(col + arrow_len_cells * math.cos(theta)))
                tip_y = int(round(row - arrow_len_cells * math.sin(theta)))
                cv2.arrowedLine(
                    img,
                    (col, row),
                    (tip_x, tip_y),
                    color,
                    1,
                    tipLength=0.35,
                )

                cv2.putText(
                    img,
                    f"T{tag_id}:{tag['num_observations']}",
                    (col + 6, row - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        # Draw the current robot position as a green circle.
        robot_gx, robot_gy = world_to_map_xy(
            self.pose.x_m, self.pose.y_m, self.origin_x_m, self.origin_y_m, self.resolution_m
        )
        if self.in_bounds_xy(robot_gx, robot_gy):
            robot_row, robot_col = map_xy_to_row_col(robot_gx, robot_gy, self.height_cells)
            cv2.circle(img, (robot_col, robot_row), 3, (0, 255, 0), -1)

            # Draw a small heading arrow.
            arrow_len_cells = max(6, int(0.20 / self.resolution_m))
            tip_x = int(round(robot_col + arrow_len_cells * math.cos(self.pose.theta_rad)))
            tip_y = int(round(robot_row - arrow_len_cells * math.sin(self.pose.theta_rad)))
            cv2.arrowedLine(
                img,
                (robot_col, robot_row),
                (tip_x, tip_y),
                (0, 255, 0),
                1,
                tipLength=0.3,
            )

        # Add text with pose information.
        pose_text = f"x={self.pose.x_m:+.2f} m   y={self.pose.y_m:+.2f} m   th={math.degrees(self.pose.theta_rad):+.1f} deg"
        cv2.putText(
            img,
            pose_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            (
                f"Keys: q=quit s=save r=reset "
                f"mode={self.mode} "
                f"steer_pkt={self.last_packet_steering_cmd:+.1f} "
                f"odom_steer={self.last_odom_steering_cmd:+.1f} "
                f"map={'Y' if self.last_integrated_scan else 'N'} "
                f"match={'Y' if self.last_scan_match_used else 'N'} "
                f"snap={'Y' if self.last_snap_applied else 'N'}"
            ),
            (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        if self.last_scan_match_score is not None:
            cv2.putText(
                img,
                f"match_score={self.last_scan_match_score:+.3f}",
                (10, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

        if self.last_corridor_axis_deg is not None:
            cv2.putText(
                img,
                f"corridor_axis={self.last_corridor_axis_deg:+.1f} deg  relock_n={self.relock_stable_count}",
                (10, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            img,
            (
                f"tags={len(tag_summary)} "
                f"seen={','.join(map(str, self.last_seen_tags[-5:])) if self.last_seen_tags else 'none'} "
                f"confirmed={sum(1 for t in tag_summary.values() if t['confirmed'])}"
            ),
            (10, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        if DISPLAY_SCALE != 1:
            img = cv2.resize(
                img,
                (img.shape[1] * DISPLAY_SCALE, img.shape[0] * DISPLAY_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )

        return img

    def save_map_files(self):
        """
        Save:
        - PNG preview image
        - ROS-style PGM occupancy image
        - YAML metadata file
        - AprilTag landmark sidecar JSON
        """
        stem = f"map_{timestamp_string()}"

        png_path = SAVE_DIR / f"{stem}.png"
        pgm_path = SAVE_DIR / f"{stem}.pgm"
        yaml_path = SAVE_DIR / f"{stem}.yaml"
        tag_map_path = SAVE_DIR / f"{stem}_tags.json"
        metrics_path = SAVE_DIR / f"{stem}_mapping_metrics.json"

        # Save the human-friendly preview PNG.
        preview_img = self.render_map()
        cv2.imwrite(str(png_path), preview_img)

        # Build the ROS map image directly from probabilities.
        # ROS map_server convention:
        # - occupied   -> 0
        # - free       -> 254
        # - unknown    -> 205
        prob = logodds_to_probability(self.log_odds)

        pgm = np.full((self.height_cells, self.width_cells), 205, dtype=np.uint8)
        pgm[prob < 0.35] = 254
        pgm[prob > 0.65] = 0

        cv2.imwrite(str(pgm_path), pgm)

        # Write the YAML file expected by map_server / Nav2.
        yaml_text = (
            f"image: {pgm_path.name}\n"
            f"mode: trinary\n"
            f"resolution: {self.resolution_m:.6f}\n"
            f"origin: [{self.origin_x_m:.6f}, {self.origin_y_m:.6f}, 0.000000]\n"
            f"negate: 0\n"
            f"occupied_thresh: 0.65\n"
            f"free_thresh: 0.35\n"
        )
        yaml_path.write_text(yaml_text, encoding="utf-8")
        num_saved_tags = self.save_tag_map_file(tag_map_path, stem)
        self.save_mapping_metrics_file(metrics_path, stem)

        print(f"[SAVE] PNG : {png_path}")
        print(f"[SAVE] PGM : {pgm_path}")
        print(f"[SAVE] YAML: {yaml_path}")
        print(f"[SAVE] TAGS: {tag_map_path}  ({num_saved_tags} confirmed tags)")
        print(f"[SAVE] METRICS: {metrics_path}")
        
# ----------------------------------------------------------------------------
# AprilTag camera detector
# ----------------------------------------------------------------------------
class AprilTagLandmarkDetector:
    """
    Detect AprilTags from a camera and store their map/world coordinates.

    The mapper already knows the robot pose. This class estimates the tag
    position relative to the camera, converts it into robot frame, then converts
    it into map/world frame.
    """
    def __init__(self, camera_index: int, tag_size_m: float):
        self.camera_index = camera_index
        self.tag_size_m = tag_size_m
        self.last_process_s = 0.0
        self.enabled = False
        self.cap = None
        self.last_restart_request_timestamp = None
        self.failed_read_count = 0

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        half_size = tag_size_m / 2.0
        self.tag_3d_corners = np.array([
            [-half_size,  half_size, 0.0],
            [ half_size,  half_size, 0.0],
            [ half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ], dtype=np.float32)

        self.reopen_camera(reason="initial open")

    def reopen_camera(self, reason: str = "manual restart"):
        """
        Release and reopen the OpenCV camera without restarting mapping.
        """
        print(f"[TAGS] Reopening camera index {self.camera_index} ({reason})")

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        time.sleep(0.25)

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            self.enabled = False
            print(f"[WARN] Could not open AprilTag camera index {self.camera_index}")
            return False

        self.enabled = True
        self.failed_read_count = 0
        print(f"[TAGS] Camera reopened successfully on index {self.camera_index}")
        return True

    def check_camera_restart_request(self):
        """
        Watch for a GUI-created restart request file.
        """
        if not CAMERA_RESTART_REQUEST_PATH.exists():
            return

        try:
            data = json.loads(CAMERA_RESTART_REQUEST_PATH.read_text(encoding="utf-8"))
            request_timestamp = float(data.get("timestamp", 0.0))
        except Exception as exc:
            print(f"[WARN] Could not read camera restart request: {exc}")
            return

        if self.last_restart_request_timestamp == request_timestamp:
            return

        self.last_restart_request_timestamp = request_timestamp
        self.reopen_camera(reason="GUI restart request")

    def close(self):
        if getattr(self, "cap", None) is not None:
            self.cap.release()
        try:
            cv2.destroyWindow("AprilTag Landmark Camera")
        except cv2.error:
            pass

    def camera_tvec_to_robot_xy(self, tvec):
        """
        Convert OpenCV camera-frame translation into robot-frame x/y.

        OpenCV camera:
        - x = right
        - y = down
        - z = forward

        Robot/map convention:
        - x = forward
        - y = left
        """
        x_cam = float(tvec[0][0])
        z_cam = float(tvec[2][0])

        # Camera forward becomes robot +x.
        # Camera right becomes robot -y.
        x_robot_cam = z_cam
        y_robot_cam = -x_cam

        yaw = math.radians(CAMERA_YAW_OFFSET_DEG)
        c = math.cos(yaw)
        s = math.sin(yaw)

        x_robot = CAMERA_X_M + c * x_robot_cam - s * y_robot_cam
        y_robot = CAMERA_Y_M + s * x_robot_cam + c * y_robot_cam

        return x_robot, y_robot, z_cam

    def camera_vector_to_robot_xy(self, v_cam):
        """
        Convert an OpenCV camera-frame direction vector into robot-frame x/y.

        This is for directions, not positions, so we do NOT add CAMERA_X_M/Y_M.
        """
        x_cam = float(v_cam[0])
        z_cam = float(v_cam[2])

        x_robot_cam = z_cam
        y_robot_cam = -x_cam

        yaw = math.radians(CAMERA_YAW_OFFSET_DEG)
        c = math.cos(yaw)
        s = math.sin(yaw)

        x_robot = c * x_robot_cam - s * y_robot_cam
        y_robot = s * x_robot_cam + c * y_robot_cam

        return x_robot, y_robot

    def tag_facing_heading_world(self, mapper: OccupancyMapper, rvec):
        """
        Estimate the AprilTag plane-normal heading in the map/world frame.

        This is stored as extra info. For actual docking, we mainly use
        approach_theta_rad, which is more reliable for this project.
        """
        R_cam_tag, _ = cv2.Rodrigues(rvec)

        # Tag local +Z axis expressed in the OpenCV camera frame.
        normal_cam = R_cam_tag[:, 2]
        nx_robot, ny_robot = self.camera_vector_to_robot_xy(normal_cam)

        c = math.cos(mapper.pose.theta_rad)
        s = math.sin(mapper.pose.theta_rad)

        nx_world = c * nx_robot - s * ny_robot
        ny_world = s * nx_robot + c * ny_robot

        return math.atan2(ny_world, nx_world)

    def robot_xy_to_world_xy(self, mapper: OccupancyMapper, x_robot: float, y_robot: float):
        """
        Convert robot-frame x/y into mapper world x/y.
        """
        c = math.cos(mapper.pose.theta_rad)
        s = math.sin(mapper.pose.theta_rad)

        x_world = mapper.pose.x_m + c * x_robot - s * y_robot
        y_world = mapper.pose.y_m + s * x_robot + c * y_robot

        return x_world, y_world

    def process_frame(self, mapper: OccupancyMapper, show_window: bool = True):
        """
        Process one camera frame and add any detected tags to the mapper.
        """
        self.check_camera_restart_request()

        if not self.enabled:
            return

        now = time.time()
        if now - self.last_process_s < APRILTAG_PROCESS_EVERY_S:
            return
        self.last_process_s = now

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.failed_read_count += 1
            if self.failed_read_count % 20 == 0:
                print(f"[TAGS] Camera read failed {self.failed_read_count} times")

            # Auto-reopen if camera gets stuck after WiFi switching.
            if self.failed_read_count >= 60:
                self.reopen_camera(reason="too many failed reads")
            return
        
        self.failed_read_count = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = self.detector.detectMarkers(gray)

        seen_tags = []

        if ids is not None:
            for i in range(len(ids)):
                tag_id = int(ids[i][0])
                tag_2d_corners = corners[i][0]

                success, rvec, tvec = cv2.solvePnP(
                    self.tag_3d_corners,
                    tag_2d_corners,
                    CAM_MATRIX,
                    DIST_COEFFS,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )

                if not success:
                    continue

                x_robot, y_robot, z_cam = self.camera_tvec_to_robot_xy(tvec)

                if z_cam < APRILTAG_MIN_Z_M or z_cam > APRILTAG_MAX_Z_M:
                    continue

                x_world, y_world = self.robot_xy_to_world_xy(mapper, x_robot, y_robot)

                # Direction the robot was facing toward the tag. Useful later for
                # visual homing / approach pose.
                approach_theta_rad = math.atan2(
                    y_world - mapper.pose.y_m,
                    x_world - mapper.pose.x_m,
                )

                tag_facing_theta_rad = self.tag_facing_heading_world(mapper, rvec)

                mapper.add_tag_observation(
                    tag_id=tag_id,
                    x_m=x_world,
                    y_m=y_world,
                    approach_theta_rad=approach_theta_rad,
                    z_cam_m=z_cam,
                    tag_facing_theta_rad=tag_facing_theta_rad,
                )

                seen_tags.append(tag_id)

                cv2.polylines(frame, [tag_2d_corners.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.drawFrameAxes(frame, CAM_MATRIX, DIST_COEFFS, rvec, tvec, 0.04)
                cv2.putText(
                    frame,
                    f"ID {tag_id} map=({x_world:+.2f},{y_world:+.2f}) z={z_cam:.2f}",
                    (10, 30 + 28 * len(seen_tags)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        mapper.last_seen_tags = seen_tags
        mapper.write_live_tag_status()

        if show_window:
            mode_text = "RECORDING" if mapper.mode == MODE_STRAIGHT else f"PAUSED: {mapper.mode}"
            cv2.putText(
                frame,
                f"AprilTag mapping: {mode_text}",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if mapper.mode == MODE_STRAIGHT else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("AprilTag Landmark Camera", frame)

# ----------------------------------------------------------------------------
# Packet parsing
# ----------------------------------------------------------------------------
PACKET_FORMAT = "<ihH" + ("h" * EXPECTED_NUM_BINS)
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)


def parse_udp_packet(packet_bytes: bytes):
    """
    Parse one binary Arduino packet.

    Format:
    - int32  encoder_count
    - int16  applied_steering_cmd
    - uint16 num_bins
    - int16[EXPECTED_NUM_BINS] distances_mm
    """
    if len(packet_bytes) != PACKET_SIZE:
        raise ValueError(f"Bad packet size: got {len(packet_bytes)} bytes, expected {PACKET_SIZE}")

    unpacked = struct.unpack(PACKET_FORMAT, packet_bytes)
    encoder_count = int(unpacked[0])
    steering_cmd = float(unpacked[1])
    num_bins = int(unpacked[2])
    distances_mm = list(unpacked[3:])

    if num_bins != EXPECTED_NUM_BINS:
        raise ValueError(f"Unexpected num_bins: got {num_bins}, expected {EXPECTED_NUM_BINS}")

    return encoder_count, steering_cmd, distances_mm


# ----------------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------------
class PacketLogger:
    """
    Stream raw packets to disk so the exact same run can be replayed later.

    Each record stores:
    - timestamp
    - encoder_count
    - steering_cmd
    - distances_mm
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("wb")
        self.count = 0

    def write_packet(self, timestamp_s: float, encoder_count: int, steering_cmd: float, distances_mm):
        record = {
            "timestamp": float(timestamp_s),
            "encoder_count": int(encoder_count),
            "steering_cmd": float(steering_cmd),
            "distances_mm": list(distances_mm),
        }
        pickle.dump(record, self.f, protocol=pickle.HIGHEST_PROTOCOL)
        self.count += 1

        if self.count % LOG_FLUSH_EVERY_N_PACKETS == 0:
            self.f.flush()

    def close(self):
        if getattr(self, "f", None) is not None:
            self.f.flush()
            self.f.close()
            self.f = None


def replay_packet_stream(path: Path):
    """
    Yield logged packet records one-by-one from a replay file.
    """
    with Path(path).open("rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def process_one_packet(mapper: OccupancyMapper, encoder_count: int, steering_cmd: float, distances_mm):
    """
    Shared live/replay packet-processing path so replay stays faithful to live.
    """
    mapper.last_packet_steering_cmd = steering_cmd

    # Build robot-frame scan points once for scan matching / Manhattan snapping.
    scan_points_robot = mapper.scan_to_points_robot(distances_mm)

    valid_count = sum(1 for d in distances_mm if d > 0)
    if mapper.packet_count % PRINT_EVERY_N_PACKETS == 0:
        print(f"[PKT] enc={encoder_count} steer_pkt={steering_cmd} valid={valid_count}")

    if len(distances_mm) != EXPECTED_NUM_BINS:
        print(
            f"[WARN] Expected {EXPECTED_NUM_BINS} bins but received {len(distances_mm)} bins"
        )

    # Update the mode FIRST.
    # This lets update_odometry know whether it should move the pose or freeze
    # during an intentional turn.
    mapper.update_mapper_state(scan_points_robot, distances_mm, steering_cmd)

    # Now update odometry. If the mapper is in TURNING/unstable RELOCK,
    # update_odometry will consume encoder ticks but not move the pose.
    mapper.update_odometry(encoder_count, steering_cmd)
    mapper.refine_pose_with_scan_match(scan_points_robot, steering_cmd)

    # Critical:
    # Do NOT draw scans into the map while the robot is turning or still relocking.
    # Those scans are exactly what create diagonal/curved corner garbage.
    should_integrate = True

    if mapper.mode == MODE_TURNING:
        should_integrate = False

    if mapper.mode == MODE_RELOCK and mapper.relock_stable_count < RELOCK_REQUIRED_STABLE_SCANS:
        should_integrate = False

    if should_integrate:
        mapper.integrate_scan(distances_mm)
        mapper.last_integrated_scan = True
    else:
        mapper.last_integrated_scan = False

    mapper.packet_count += 1

def main():
    parser = argparse.ArgumentParser(description="UDP Occupancy Mapper with raw logging and replay.")
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Replay a previously logged raw packet file instead of listening on UDP.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to save raw packet log during live mapping. Default: occupancy_maps/raw_packets_<timestamp>.pkl",
    )
    parser.add_argument(
        "--replay-real-time",
        action="store_true",
        help="During replay, approximately preserve recorded packet timing.",
    )
    parser.add_argument(
        "--tags",
        action="store_true",
        help="Enable AprilTag landmark detection during live mapping.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=APRILTAG_CAMERA_INDEX,
        help="Camera index used for AprilTag landmark mapping.",
    )
    parser.add_argument(
        "--tag-size-m",
        type=float,
        default=APRILTAG_SIZE_M,
        help="Physical AprilTag side length in meters.",
    )
    parser.add_argument(
        "--no-tag-window",
        action="store_true",
        help="Detect/store AprilTags without opening a separate camera preview window.",
    )
    args = parser.parse_args()

    print("==============================================================")
    print("UDP Occupancy Mapper")
    print("==============================================================")
    print(f"Listening on UDP port {UDP_PORT}")
    print(f"Expected LiDAR bins: {EXPECTED_NUM_BINS}")
    print(f"Expected UDP packet size: {PACKET_SIZE} bytes")
    print(f"Map size: {MAP_SIZE_X_M:.1f} m x {MAP_SIZE_Y_M:.1f} m")
    print(f"Resolution: {MAP_RESOLUTION_M:.3f} m/cell")
    print("Run your existing robot_gui.py separately to drive the robot.")
    print("Press 's' in the map window to save. Press 'q' to quit.")
    print("==============================================================")
    if args.replay is None:
        print("Mode: LIVE")
    else:
        print(f"Mode: REPLAY ({args.replay})")

    mapper = OccupancyMapper()
    tag_detector = None

    if args.tags:
        if args.replay is not None:
            print("[WARN] AprilTag mapping is disabled during replay.")
        else:
            tag_detector = AprilTagLandmarkDetector(
                camera_index=args.camera_index,
                tag_size_m=args.tag_size_m,
            )

    sock = None
    packet_logger = None
    replay_iter = None
    replay_prev_timestamp = None
    replay_finished = False
    replay_finish_announced = False

    if args.replay is None:
        # Live UDP mode
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", UDP_PORT))
        sock.settimeout(SOCKET_TIMEOUT_S)

        log_path = Path(args.log) if args.log else (SAVE_DIR / f"raw_packets_{timestamp_string()}.pkl")
        packet_logger = PacketLogger(log_path)
        print(f"[LOG] Writing raw packets to: {log_path}")
    else:
        # Replay mode
        replay_iter = iter(replay_packet_stream(Path(args.replay)))
        print("[REPLAY] Replay loaded. Press 'q' to quit, 's' to save map.")

    cv2.namedWindow("UDP Occupancy Mapper", cv2.WINDOW_NORMAL)

    last_display_time = 0.0

    try:
        while True:
            if args.replay is None:
                # ----------------------------
                # LIVE MODE
                # ----------------------------
                try:
                    data, _addr = sock.recvfrom(8192)
                    encoder_count, steering_cmd, distances_mm = parse_udp_packet(data)

                    packet_logger.write_packet(
                        timestamp_s=time.time(),
                        encoder_count=encoder_count,
                        steering_cmd=steering_cmd,
                        distances_mm=distances_mm,
                    )

                    process_one_packet(mapper, encoder_count, steering_cmd, distances_mm)

                except socket.timeout:
                    pass
                except ValueError as exc:
                    print(f"[WARN] Bad packet: {exc}")
                except Exception as exc:
                    print(f"[WARN] Packet handling error: {exc}")
            else:
                # ----------------------------
                # REPLAY MODE
                # ----------------------------
                if not replay_finished:
                    try:
                        record = next(replay_iter)

                        timestamp_s = float(record["timestamp"])
                        encoder_count = int(record["encoder_count"])
                        steering_cmd = float(record["steering_cmd"])
                        distances_mm = list(record["distances_mm"])

                        if args.replay_real_time and replay_prev_timestamp is not None:
                            dt = max(0.0, timestamp_s - replay_prev_timestamp)
                            time.sleep(min(dt, REPLAY_SLEEP_CAP_S))
                        replay_prev_timestamp = timestamp_s

                        process_one_packet(mapper, encoder_count, steering_cmd, distances_mm)

                    except StopIteration:
                        replay_finished = True
                    except Exception as exc:
                        print(f"[WARN] Replay handling error: {exc}")

                if replay_finished and not replay_finish_announced:
                    print("[INFO] Replay finished. Press 's' to save map or 'q' to quit.")
                    replay_finish_announced = True

            # Process AprilTag camera frames during live mapping.
            if tag_detector is not None:
                tag_detector.process_frame(
                    mapper,
                    show_window=not args.no_tag_window,
                )

            # Refresh the display.
            now = time.time()
            if mapper.packet_count % DISPLAY_EVERY_N_PACKETS == 0 or (now - last_display_time) > 0.1:
                display_img = mapper.render_map()

                # Add a small LIVE / REPLAY tag to the window.
                mode_text = "LIVE" if args.replay is None else "REPLAY"
                cv2.putText(
                    display_img,
                    mode_text,
                    (10, display_img.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (20, 20, 20),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("UDP Occupancy Mapper", display_img)
                last_display_time = now

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                mapper.save_map_files()
            elif key == ord("r"):
                mapper.reset()
                print("[INFO] Map reset")

    finally:
        if tag_detector is not None:
            tag_detector.close()
        if packet_logger is not None:
            packet_logger.close()
        if sock is not None:
            sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
