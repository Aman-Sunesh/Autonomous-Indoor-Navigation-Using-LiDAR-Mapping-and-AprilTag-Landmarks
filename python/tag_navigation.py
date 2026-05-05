#!/usr/bin/env python3
"""
# ============================================================================
# AprilTag Navigation and Docking
# ============================================================================
# This script drives the robot to a selected AprilTag using a saved map.
#
# Main tasks:
# - Loads the saved 2D occupancy grid map and AprilTag landmark file.
# - Uses the camera to detect a known AprilTag and initialize the robot pose.
# - Uses A* path planning to find a route from the robot to the target AprilTag.
# - Sends speed and steering commands to the Arduino over UDP.
# - Updates the robot pose using encoder-based odometry.
# - Uses LiDAR for front obstacle safety and emergency stopping.
# - Uses LiDAR side-wall centering during straight driving.
# - Uses visual docking when the target AprilTag is visible near the goal.
# - Saves navigation results such as time, final error, collision count, and loss.

Run example:
python tag_navigation.py --target-tag 17 --map-yaml occupancy_maps/map_20260430_211021.yaml --tag-map occupancy_maps/map_20260430_211021_tags.json --camera-index 1 --straight-drive
# ============================================================================
"""

import argparse
import json
import math
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# Robot / UDP / odometry settings
# =============================================================================
ARDUINO_IP_DEFAULT = "<Enter Arduino IP>"
ARDUINO_PORT_DEFAULT = 4010
LOCAL_UDP_PORT_DEFAULT = 4010

EXPECTED_NUM_BINS = 180
PACKET_FORMAT = "<ihH" + ("h" * EXPECTED_NUM_BINS)
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

METERS_PER_TICK = 0.00019
WHEELBASE_M = 0.26
STEERING_DEG_PER_CMD = 1.5
STEERING_SIGN = -1.0
STEERING_OFFSET_CMD = 3.0
STRAIGHT_STEER_DEADBAND_CMD = 4.0
STEERING_ODOM_SCALE = 0.6

STRAIGHT_STEER_CMD = -3
STEER_CMD_MIN = -20
STEER_CMD_MAX = 20

SPEED_MIN = 0
SPEED_MAX = 60

FLIP_SCAN = True
ANGLE_ZERO_OFFSET_DEG = 0.0

# Front safety.
COLLISION_FRONT_WINDOW_DEG = 12.0
FRONT_STOP_RANGE_M = 0.25
FRONT_CRAWL_RANGE_M = 0.40
FRONT_SLOW_RANGE_M = 0.65
COLLISION_RANGE_M = 0.18
COLLISION_EVENT_COOLDOWN_S = 0.8


# =============================================================================
# AprilTag / camera settings
# =============================================================================
APRILTAG_SIZE_M = 0.167
CAMERA_INDEX_DEFAULT = 1

CAMERA_MATRIX = np.array(
    [
        [656.96495861, 0.0, 630.84080906],
        [0.0, 658.39179162, 378.57855594],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

DIST_COEFFS = np.array(
    [[0.10811153, -0.27102501, 0.00188884, 0.00495593, 0.20428807]],
    dtype=np.float32,
)

CAMERA_Y_SIGN = -1.0
TAG_POSITION_CORRECTION_ALPHA = 0.25
TAG_MAX_LOCALIZATION_RANGE_M = 2.5

DOCK_START_RANGE_M = 1.2
DOCK_STOP_RANGE_M = 0.35
DOCK_CENTER_TOL_M = 0.06
GOAL_RADIUS_M = 0.30


# =============================================================================
# Planning / control settings
# =============================================================================
MAP_UNKNOWN_IS_OBSTACLE = False
INFLATION_RADIUS_M = 0.18
A_STAR_ALLOW_DIAGONALS = True

LOOKAHEAD_M = 0.45
WAYPOINT_REACHED_M = 0.18
MAX_NAV_TIME_S = 180.0
CONTROL_DT_S = 0.05

NAV_BASE_SPEED = 32
NAV_SLOW_SPEED = 20
DOCK_BASE_SPEED = 16
DOCK_MIN_SPEED = 10

# Straight-drive mode.
STRAIGHT_DRIVE_SPEED = 50
STRAIGHT_DRIVE_STEER = -3

# LiDAR side-wall centering for straight-drive mode.
LIDAR_CENTERING_ENABLE = True
LIDAR_CENTER_LEFT_DEG = 90.0
LIDAR_CENTER_RIGHT_DEG = 270.0
LIDAR_CENTER_HALF_WINDOW_DEG = 14.0
LIDAR_CENTER_MIN_RANGE_M = 0.15
LIDAR_CENTER_MAX_RANGE_M = 2.50

# If centering makes wall-hugging worse, flip this from +1.0 to -1.0.
LIDAR_CENTERING_SIGN = 1.0

# Larger = stronger left/right correction.
LIDAR_CENTERING_K_CMD_PER_M = 12.0
LIDAR_CENTERING_MAX_CORR_CMD = 7

K_STEER_HEADING = 18.0
K_STEER_DOCK_CENTER = 22.0

LOSS_TIME_W = 0.05
LOSS_POS_W = 10.0
LOSS_HEADING_W = 0.05
LOSS_COLLISION_W = 50.0


# =============================================================================
# Data containers
# =============================================================================
@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass
class TagObservation:
    tag_id: int
    x_robot: float
    y_robot: float
    z_cam: float
    range_m: float
    bearing_rad: float
    rvec: np.ndarray
    tvec: np.ndarray


@dataclass
class NavMetrics:
    target_tag: int
    start_time_s: float
    end_time_s: Optional[float] = None
    success: bool = False
    fail_reason: str = ""
    final_position_error_m: Optional[float] = None
    final_heading_error_deg: Optional[float] = None
    collision_count: int = 0
    loss: Optional[float] = None


# =============================================================================
# Basic helpers
# =============================================================================
def angle_wrap(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(a_rad: float, b_rad: float) -> float:
    return angle_wrap(a_rad - b_rad)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def nearest_manhattan_heading(angle_rad: float) -> float:
    return angle_wrap(round(angle_rad / (math.pi / 2.0)) * (math.pi / 2.0))


def rot2(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


# =============================================================================
# Map + tag loading
# =============================================================================
def parse_simple_map_yaml(yaml_path: Path) -> Dict[str, object]:
    out: Dict[str, object] = {}

    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key == "origin":
            value = value.strip("[]")
            out[key] = [float(v.strip()) for v in value.split(",")]
        elif key in {"resolution", "occupied_thresh", "free_thresh"}:
            out[key] = float(value)
        elif key == "negate":
            out[key] = int(value)
        else:
            out[key] = value

    return out


def find_latest_file(folder: Path, pattern: str) -> Optional[Path]:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_map(map_yaml_path: Optional[str]) -> Tuple[np.ndarray, float, float, float, Path]:
    if map_yaml_path is None:
        latest_yaml = find_latest_file(Path("occupancy_maps"), "map_*.yaml")

        if latest_yaml is None:
            raise FileNotFoundError("No map YAML found. Pass --map-yaml explicitly.")

        yaml_path = latest_yaml
    else:
        yaml_path = Path(map_yaml_path)

    meta = parse_simple_map_yaml(yaml_path)

    image_name = str(meta["image"])
    resolution = float(meta["resolution"])
    origin = meta.get("origin", [0.0, 0.0, 0.0])

    origin_x = float(origin[0])
    origin_y = float(origin[1])

    pgm_path = yaml_path.parent / image_name
    img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not read map image: {pgm_path}")

    return img, resolution, origin_x, origin_y, yaml_path


def normalize_tag_record(record: dict) -> dict:
    out = dict(record)

    if "x_m" in out and "y_m" in out:
        return out

    if "x" in record and "y" in record:
        out["x_m"] = record["x"]
        out["y_m"] = record["y"]
        return out

    if "position" in record and isinstance(record["position"], dict):
        pos = record["position"]
        out["x_m"] = pos.get("x_m", pos.get("x", 0.0))
        out["y_m"] = pos.get("y_m", pos.get("y", 0.0))
        return out

    raise ValueError(f"Unsupported tag_map record shape: {record}")


def load_tag_map(tag_map_path: Optional[str]) -> Tuple[Dict[int, dict], Path]:
    if tag_map_path is None:
        candidates = [
            Path("occupancy_maps") / "tag_map.json",
            find_latest_file(Path("occupancy_maps"), "map_*_tags.json"),
        ]

        json_path = next((p for p in candidates if p is not None and p.exists()), None)

        if json_path is None:
            raise FileNotFoundError("No tag_map JSON found. Pass --tag-map explicitly.")
    else:
        json_path = Path(tag_map_path)

    data = json.loads(json_path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "tags" in data:
        raw_tags = data["tags"]
    else:
        raw_tags = data

    tags: Dict[int, dict] = {}

    if isinstance(raw_tags, dict):
        for k, v in raw_tags.items():
            tags[int(k)] = normalize_tag_record(v)
    elif isinstance(raw_tags, list):
        for rec in raw_tags:
            tag_id = int(rec.get("id", rec.get("tag_id")))
            tags[tag_id] = normalize_tag_record(rec)
    else:
        raise ValueError("Unsupported tag_map JSON structure.")

    return tags, json_path


# =============================================================================
# Occupancy grid + A*
# =============================================================================
def world_to_grid(x_m: float, y_m: float, origin_x: float, origin_y: float, resolution: float) -> Tuple[int, int]:
    gx = int(math.floor((x_m - origin_x) / resolution))
    gy = int(math.floor((y_m - origin_y) / resolution))
    return gx, gy


def grid_to_world(gx: int, gy: int, origin_x: float, origin_y: float, resolution: float) -> Tuple[float, float]:
    x = origin_x + (gx + 0.5) * resolution
    y = origin_y + (gy + 0.5) * resolution
    return x, y


def grid_to_row_col(gx: int, gy: int, height: int) -> Tuple[int, int]:
    return height - 1 - gy, gx


def build_obstacle_grid(pgm: np.ndarray, resolution: float) -> np.ndarray:
    occupied = pgm < 128

    if MAP_UNKNOWN_IS_OBSTACLE:
        unknown = (pgm >= 128) & (pgm <= 230)
        occupied = occupied | unknown

    inflate_cells = max(1, int(math.ceil(INFLATION_RADIUS_M / resolution)))
    kernel_size = 2 * inflate_cells + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    inflated = cv2.dilate(occupied.astype(np.uint8), kernel, iterations=1).astype(bool)
    return inflated


def is_free_grid(obstacle: np.ndarray, gx: int, gy: int) -> bool:
    h, w = obstacle.shape

    if gx < 0 or gx >= w or gy < 0 or gy >= h:
        return False

    row, col = grid_to_row_col(gx, gy, h)
    return not bool(obstacle[row, col])


def nearest_free_cell(obstacle: np.ndarray, start: Tuple[int, int], max_radius: int = 20) -> Tuple[int, int]:
    sx, sy = start

    if is_free_grid(obstacle, sx, sy):
        return start

    best = None
    best_d2 = 1e18

    for r in range(1, max_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r:
                    continue

                gx = sx + dx
                gy = sy + dy

                if is_free_grid(obstacle, gx, gy):
                    d2 = dx * dx + dy * dy

                    if d2 < best_d2:
                        best_d2 = d2
                        best = (gx, gy)

        if best is not None:
            return best

    raise RuntimeError(f"No free cell near {start}")


def astar(obstacle: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    import heapq

    if A_STAR_ALLOW_DIAGONALS:
        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
        ]
    else:
        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        ]

    def h(cell: Tuple[int, int]) -> float:
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1])

    open_heap = [(h(start), 0.0, start)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    visited = set()

    while open_heap:
        _, g_cur, cur = heapq.heappop(open_heap)

        if cur in visited:
            continue

        visited.add(cur)

        if cur == goal:
            path = [cur]

            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)

            path.reverse()
            return path

        cx, cy = cur

        for dx, dy, step_cost in neighbors:
            nb = (cx + dx, cy + dy)
            nx, ny = nb

            if not is_free_grid(obstacle, nx, ny):
                continue

            new_g = g_cur + step_cost

            if new_g < g_score.get(nb, 1e18):
                g_score[nb] = new_g
                came_from[nb] = cur
                heapq.heappush(open_heap, (new_g + h(nb), new_g, nb))

    raise RuntimeError("A* failed: no path found.")


def grid_path_to_world_path(
    path_cells: List[Tuple[int, int]],
    origin_x: float,
    origin_y: float,
    resolution: float,
    simplify_every: int = 3,
) -> List[Tuple[float, float]]:
    points = []

    for i, (gx, gy) in enumerate(path_cells):
        if i % simplify_every == 0 or i == len(path_cells) - 1:
            points.append(grid_to_world(gx, gy, origin_x, origin_y, resolution))

    return points


# =============================================================================
# UDP command + telemetry
# =============================================================================
def parse_udp_packet(packet_bytes: bytes):
    if len(packet_bytes) != PACKET_SIZE:
        return None

    unpacked = struct.unpack(PACKET_FORMAT, packet_bytes)

    encoder_count = int(unpacked[0])
    steering_cmd = float(unpacked[1])
    num_bins = int(unpacked[2])
    distances_mm = list(unpacked[3:])

    if num_bins != EXPECTED_NUM_BINS:
        return None

    return encoder_count, steering_cmd, distances_mm


class RobotIO:
    def __init__(self, arduino_ip: str, arduino_port: int, local_udp_port: int):
        self.arduino_ip = arduino_ip
        self.arduino_port = arduino_port
        self.local_udp_port = local_udp_port
        self.last_error = ""

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.sock.bind(("0.0.0.0", local_udp_port))
        self.sock.settimeout(0.001)

    def send_command(self, speed: int, steer: int, auto_mode: int = 0) -> bool:
        speed = int(clamp(speed, SPEED_MIN, SPEED_MAX))
        steer = int(clamp(steer, STEER_CMD_MIN, STEER_CMD_MAX))

        msg = f"{speed},{steer},{auto_mode}".encode("utf-8")

        try:
            self.sock.sendto(msg, (self.arduino_ip, self.arduino_port))
            self.last_error = ""
            return True
        except OSError as exc:
            self.last_error = str(exc)
            return False

    def stop(self):
        self.send_command(0, STRAIGHT_STEER_CMD, 0)

    def read_latest_sensor(self):
        latest = None

        while True:
            try:
                data, _addr = self.sock.recvfrom(8192)
                parsed = parse_udp_packet(data)

                if parsed is not None:
                    latest = parsed

            except socket.timeout:
                break
            except BlockingIOError:
                break
            except OSError as exc:
                self.last_error = str(exc)
                break

        return latest

    def close(self):
        try:
            self.stop()
        except Exception:
            pass

        try:
            self.sock.close()
        except Exception:
            pass


# =============================================================================
# AprilTag detection
# =============================================================================
def build_apriltag_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector


def tag_object_points() -> np.ndarray:
    half = APRILTAG_SIZE_M / 2.0

    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def detect_tags(frame: np.ndarray, detector, object_points: np.ndarray) -> List[TagObservation]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)

    observations: List[TagObservation] = []

    if ids is None:
        return observations

    for i in range(len(ids)):
        tag_id = int(ids[i][0])
        image_points = corners[i][0].astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            CAMERA_MATRIX,
            DIST_COEFFS,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not success:
            continue

        x_cam = float(tvec[0][0])
        z_cam = float(tvec[2][0])

        x_robot = z_cam
        y_robot = CAMERA_Y_SIGN * x_cam

        range_m = math.hypot(x_robot, y_robot)
        bearing_rad = math.atan2(y_robot, x_robot)

        observations.append(
            TagObservation(
                tag_id=tag_id,
                x_robot=x_robot,
                y_robot=y_robot,
                z_cam=z_cam,
                range_m=range_m,
                bearing_rad=bearing_rad,
                rvec=rvec,
                tvec=tvec,
            )
        )

        cv2.polylines(frame, [image_points.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.03)
        cv2.putText(
            frame,
            f"ID {tag_id} xr={x_robot:.2f} yr={y_robot:.2f}",
            tuple(image_points[0].astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return observations


# =============================================================================
# Navigation controller
# =============================================================================
class TagNavigator:
    def __init__(
        self,
        tag_map: Dict[int, dict],
        obstacle_grid: np.ndarray,
        resolution: float,
        origin_x: float,
        origin_y: float,
        target_tag: int,
        start_heading_deg: float,
        arduino_ip: str,
        arduino_port: int,
        local_udp_port: int,
        start_tag: Optional[int] = None,
        no_camera: bool = False,
        straight_drive: bool = False,
    ):
        self.tag_map = tag_map
        self.obstacle_grid = obstacle_grid
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        self.target_tag = int(target_tag)
        self.start_tag = None if start_tag is None else int(start_tag)
        self.no_camera = bool(no_camera)
        self.straight_drive = bool(straight_drive)

        self.arduino_ip = arduino_ip
        self.arduino_port = arduino_port
        self.local_udp_port = local_udp_port
        self.io: Optional[RobotIO] = None

        self.pose = Pose2D(theta=math.radians(start_heading_deg))
        self.pose_initialized = False
        self.prev_encoder: Optional[int] = None
        self.last_steer_cmd = STRAIGHT_STEER_CMD

        self.path: List[Tuple[float, float]] = []
        self.path_index = 0

        self.metrics = NavMetrics(target_tag=self.target_tag, start_time_s=time.time())
        self.last_collision_time_s = -1e9

        self.phase = "LOCALIZE"
        self.status_message = "Waiting for camera localization..."

        self.last_centering_message = "LiDAR centering: no data yet"
        self.last_front_message = "Front: no data yet"

        if self.target_tag not in self.tag_map:
            raise ValueError(f"Target tag {self.target_tag} not found in tag_map.")

        if self.start_tag is not None:
            self.initialize_from_start_tag(start_heading_deg)

    def initialize_from_start_tag(self, start_heading_deg: float):
        if self.start_tag not in self.tag_map:
            raise ValueError(f"Start tag {self.start_tag} not found in tag_map.")

        rec = self.tag_map[self.start_tag]

        tag_x = float(rec["x_m"])
        tag_y = float(rec["y_m"])

        if "theta_rad" in rec:
            theta = float(rec["theta_rad"])
        elif "theta_deg" in rec:
            theta = math.radians(float(rec["theta_deg"]))
        else:
            theta = math.radians(start_heading_deg)

        stop_distance = float(rec.get("stop_distance_m", 0.55))

        self.pose.x = tag_x - stop_distance * math.cos(theta)
        self.pose.y = tag_y - stop_distance * math.sin(theta)
        self.pose.theta = theta
        self.pose_initialized = True

        self.phase = "READY"
        self.status_message = "Initialized from start tag. Connect to R2D2 and press G."

        print(
            f"[INIT] Start from tag {self.start_tag}: "
            f"x={self.pose.x:+.2f}, y={self.pose.y:+.2f}, "
            f"theta={math.degrees(self.pose.theta):+.1f} deg"
        )

    def open_robot_io(self) -> bool:
        if self.io is not None:
            return True

        try:
            self.io = RobotIO(
                arduino_ip=self.arduino_ip,
                arduino_port=self.arduino_port,
                local_udp_port=self.local_udp_port,
            )
            self.status_message = "UDP connected. Driving started."
            print("[UDP] Robot UDP opened successfully.")
            return True

        except PermissionError as exc:
            self.io = None
            self.status_message = "Could not open UDP port. Close mapper/GUI using port 4010."
            print(f"[UDP ERROR] Permission error: {exc}")
            return False

        except OSError as exc:
            self.io = None
            self.status_message = "Could not open UDP. Make sure laptop is on R2D2."
            print(f"[UDP ERROR] {exc}")
            return False

    def target_xy(self) -> Tuple[float, float]:
        rec = self.tag_map[self.target_tag]
        return float(rec["x_m"]), float(rec["y_m"])

    def target_heading(self) -> float:
        rec = self.tag_map[self.target_tag]

        if "theta_rad" in rec:
            return float(rec["theta_rad"])

        if "theta_deg" in rec:
            return math.radians(float(rec["theta_deg"]))

        tx, ty = self.target_xy()
        return math.atan2(ty - self.pose.y, tx - self.pose.x)

    def update_odometry(self, encoder_now: int, applied_steering_cmd: float):
        if self.prev_encoder is None:
            self.prev_encoder = encoder_now
            self.last_steer_cmd = applied_steering_cmd
            return

        delta_ticks = encoder_now - self.prev_encoder
        self.prev_encoder = encoder_now

        delta_s = delta_ticks * METERS_PER_TICK

        steering_from_straight = applied_steering_cmd + STEERING_OFFSET_CMD

        if abs(steering_from_straight) <= STRAIGHT_STEER_DEADBAND_CMD:
            steering_from_straight = 0.0

        steering_cmd = STEERING_SIGN * STEERING_ODOM_SCALE * steering_from_straight
        steering_wheel_rad = math.radians(steering_cmd * STEERING_DEG_PER_CMD)

        if abs(steering_wheel_rad) < 1e-6:
            delta_theta = 0.0
        else:
            delta_theta = (delta_s / WHEELBASE_M) * math.tan(steering_wheel_rad)

        theta_mid = self.pose.theta + 0.5 * delta_theta

        self.pose.x += delta_s * math.cos(theta_mid)
        self.pose.y += delta_s * math.sin(theta_mid)
        self.pose.theta = angle_wrap(self.pose.theta + delta_theta)

        self.last_steer_cmd = applied_steering_cmd

    def correct_pose_from_tags(self, observations: List[TagObservation]):
        known = [
            obs for obs in observations
            if obs.tag_id in self.tag_map and obs.range_m <= TAG_MAX_LOCALIZATION_RANGE_M
        ]

        if not known:
            return

        obs = sorted(known, key=lambda o: o.range_m)[0]
        tag = self.tag_map[obs.tag_id]

        tag_xy = np.array([float(tag["x_m"]), float(tag["y_m"])], dtype=np.float32)
        rel_robot = np.array([obs.x_robot, obs.y_robot], dtype=np.float32)

        if not self.pose_initialized:
            if "theta_rad" in tag:
                init_theta = float(tag["theta_rad"])
            elif "theta_deg" in tag:
                init_theta = math.radians(float(tag["theta_deg"]))
            else:
                init_theta = nearest_manhattan_heading(self.pose.theta)

            rel_map = rot2(init_theta) @ rel_robot
            estimated_robot_xy = tag_xy - rel_map

            self.pose.x = float(estimated_robot_xy[0])
            self.pose.y = float(estimated_robot_xy[1])
            self.pose.theta = angle_wrap(init_theta)
            self.pose_initialized = True

            self.phase = "READY"
            self.status_message = (
                f"Localized using tag {obs.tag_id}. "
                "Switch to R2D2, close mapper/GUI, then press G."
            )

            print(
                f"[LOCALIZED] using tag {obs.tag_id}: "
                f"x={self.pose.x:+.2f}, y={self.pose.y:+.2f}, "
                f"theta={math.degrees(self.pose.theta):+.1f} deg"
            )

            if not self.path:
                self.plan_path()

            return

        rel_map = rot2(self.pose.theta) @ rel_robot
        estimated_robot_xy = tag_xy - rel_map

        alpha = TAG_POSITION_CORRECTION_ALPHA
        self.pose.x = float((1.0 - alpha) * self.pose.x + alpha * estimated_robot_xy[0])
        self.pose.y = float((1.0 - alpha) * self.pose.y + alpha * estimated_robot_xy[1])

    def plan_path(self):
        tx, ty = self.target_xy()

        start_cell = world_to_grid(
            self.pose.x,
            self.pose.y,
            self.origin_x,
            self.origin_y,
            self.resolution,
        )

        goal_cell = world_to_grid(
            tx,
            ty,
            self.origin_x,
            self.origin_y,
            self.resolution,
        )

        start_cell = nearest_free_cell(self.obstacle_grid, start_cell)
        goal_cell = nearest_free_cell(self.obstacle_grid, goal_cell)

        path_cells = astar(self.obstacle_grid, start_cell, goal_cell)

        self.path = grid_path_to_world_path(
            path_cells,
            self.origin_x,
            self.origin_y,
            self.resolution,
        )

        self.path_index = 0

        print(f"[PLAN] path cells={len(path_cells)} waypoints={len(self.path)}")

    def raw_scan_index_to_angle_deg(self, raw_idx: int, bins: int) -> float:
        step = 360.0 / bins

        if FLIP_SCAN:
            logical_idx = (bins - raw_idx) % bins
        else:
            logical_idx = raw_idx

        return (logical_idx * step + ANGLE_ZERO_OFFSET_DEG) % 360.0

    def average_range_near_angle(
        self,
        distances_mm: List[int],
        target_deg: float,
        half_window_deg: float,
        min_range_m: float,
        max_range_m: float,
    ) -> Optional[float]:
        bins = len(distances_mm)

        if bins == 0:
            return None

        vals = []

        for raw_idx, d_mm in enumerate(distances_mm):
            if d_mm <= 0:
                continue

            r_m = d_mm / 1000.0

            if r_m < min_range_m or r_m > max_range_m:
                continue

            angle_deg = self.raw_scan_index_to_angle_deg(raw_idx, bins)
            delta = ((angle_deg - target_deg + 180.0) % 360.0) - 180.0

            if abs(delta) <= half_window_deg:
                vals.append(r_m)

        if not vals:
            return None

        return float(np.median(vals))

    def front_min_range_m(self, distances_mm: List[int]) -> Optional[float]:
        bins = len(distances_mm)

        if bins == 0:
            return None

        vals = []

        for raw_idx, d_mm in enumerate(distances_mm):
            if d_mm <= 0:
                continue

            angle_deg = self.raw_scan_index_to_angle_deg(raw_idx, bins)
            delta = ((angle_deg - 0.0 + 180.0) % 360.0) - 180.0

            if abs(delta) <= COLLISION_FRONT_WINDOW_DEG:
                vals.append(d_mm / 1000.0)

        return min(vals) if vals else None

    def centered_straight_drive_command(self, distances_mm: Optional[List[int]]) -> Tuple[int, int]:
        speed = STRAIGHT_DRIVE_SPEED
        steer = STRAIGHT_DRIVE_STEER

        if not LIDAR_CENTERING_ENABLE or distances_mm is None:
            self.last_centering_message = "LiDAR centering: no packet"
            return speed, steer

        left_m = self.average_range_near_angle(
            distances_mm,
            target_deg=LIDAR_CENTER_LEFT_DEG,
            half_window_deg=LIDAR_CENTER_HALF_WINDOW_DEG,
            min_range_m=LIDAR_CENTER_MIN_RANGE_M,
            max_range_m=LIDAR_CENTER_MAX_RANGE_M,
        )

        right_m = self.average_range_near_angle(
            distances_mm,
            target_deg=LIDAR_CENTER_RIGHT_DEG,
            half_window_deg=LIDAR_CENTER_HALF_WINDOW_DEG,
            min_range_m=LIDAR_CENTER_MIN_RANGE_M,
            max_range_m=LIDAR_CENTER_MAX_RANGE_M,
        )

        if left_m is None or right_m is None:
            self.last_centering_message = f"LiDAR centering: L={left_m} R={right_m}, using steer={steer}"
            return speed, steer

        # Positive error means right side is farther than left side.
        # The sign is tunable with LIDAR_CENTERING_SIGN.
        error_m = right_m - left_m

        correction = LIDAR_CENTERING_SIGN * LIDAR_CENTERING_K_CMD_PER_M * error_m
        correction = clamp(
            correction,
            -LIDAR_CENTERING_MAX_CORR_CMD,
            LIDAR_CENTERING_MAX_CORR_CMD,
        )

        steer = int(round(STRAIGHT_DRIVE_STEER + correction))
        steer = int(clamp(steer, STEER_CMD_MIN, STEER_CMD_MAX))

        self.last_centering_message = (
            f"LiDAR centering: L={left_m:.2f}m R={right_m:.2f}m "
            f"err={error_m:+.2f} corr={correction:+.1f} steer={steer}"
        )

        return speed, steer

    def apply_front_safety(self, distances_mm: Optional[List[int]], cmd_speed: int) -> Tuple[int, bool]:
        if distances_mm is None or cmd_speed <= 0:
            self.last_front_message = "Front: no LiDAR packet"
            return cmd_speed, False

        front_m = self.front_min_range_m(distances_mm)

        if front_m is None:
            self.last_front_message = "Front: no valid front range"
            return cmd_speed, False

        self.last_front_message = f"Front: {front_m:.2f}m"

        if front_m <= FRONT_STOP_RANGE_M:
            return 0, True

        if front_m <= FRONT_CRAWL_RANGE_M:
            return min(cmd_speed, 10), False

        if front_m <= FRONT_SLOW_RANGE_M:
            return min(cmd_speed, 25), False

        return cmd_speed, False

    def update_collision_metric(self, distances_mm: List[int], commanded_speed: int):
        front_m = self.front_min_range_m(distances_mm)
        now = time.time()

        if (
            commanded_speed > 0
            and front_m is not None
            and front_m <= COLLISION_RANGE_M
            and now - self.last_collision_time_s > COLLISION_EVENT_COOLDOWN_S
        ):
            self.metrics.collision_count += 1
            self.last_collision_time_s = now
            print(f"[COLLISION_APPROX] front={front_m:.2f} m count={self.metrics.collision_count}")

    def select_lookahead_point(self) -> Tuple[float, float]:
        if not self.path:
            return self.target_xy()

        while self.path_index < len(self.path) - 1:
            wx, wy = self.path[self.path_index]

            if math.hypot(wx - self.pose.x, wy - self.pose.y) > WAYPOINT_REACHED_M:
                break

            self.path_index += 1

        for i in range(self.path_index, len(self.path)):
            wx, wy = self.path[i]

            if math.hypot(wx - self.pose.x, wy - self.pose.y) >= LOOKAHEAD_M:
                return wx, wy

        return self.path[-1]

    def path_follow_command(self) -> Tuple[int, int]:
        lx, ly = self.select_lookahead_point()

        desired_heading = math.atan2(ly - self.pose.y, lx - self.pose.x)
        heading_error = angle_diff(desired_heading, self.pose.theta)

        steer = STRAIGHT_STEER_CMD + int(round(K_STEER_HEADING * heading_error))
        steer = int(clamp(steer, STEER_CMD_MIN, STEER_CMD_MAX))

        tx, ty = self.target_xy()
        dist_goal = math.hypot(tx - self.pose.x, ty - self.pose.y)

        speed = NAV_SLOW_SPEED if dist_goal < 1.0 else NAV_BASE_SPEED

        if abs(math.degrees(heading_error)) > 45.0:
            speed = min(speed, 16)

        return int(speed), steer

    def docking_command(self, target_obs: TagObservation) -> Tuple[int, int, bool]:
        if target_obs.range_m <= DOCK_STOP_RANGE_M and abs(target_obs.y_robot) <= DOCK_CENTER_TOL_M:
            return 0, STRAIGHT_STEER_CMD, True

        center_error_m = target_obs.y_robot

        steer = STRAIGHT_STEER_CMD + int(round(K_STEER_DOCK_CENTER * center_error_m))
        steer = int(clamp(steer, STEER_CMD_MIN, STEER_CMD_MAX))

        speed = DOCK_BASE_SPEED

        if target_obs.range_m < 0.7:
            speed = DOCK_MIN_SPEED

        return speed, steer, False

    def compute_final_metrics(self):
        tx, ty = self.target_xy()

        epos = math.hypot(self.pose.x - tx, self.pose.y - ty)

        desired_theta = self.target_heading()
        eth_deg = abs(math.degrees(angle_diff(self.pose.theta, desired_theta)))

        tgoal = (self.metrics.end_time_s or time.time()) - self.metrics.start_time_s

        self.metrics.final_position_error_m = epos
        self.metrics.final_heading_error_deg = eth_deg
        self.metrics.loss = (
            LOSS_TIME_W * tgoal
            + LOSS_POS_W * epos
            + LOSS_HEADING_W * eth_deg
            + LOSS_COLLISION_W * self.metrics.collision_count
        )

    def finish(self, success: bool, reason: str):
        if self.io is not None:
            self.io.stop()

        self.metrics.success = success
        self.metrics.fail_reason = reason
        self.metrics.end_time_s = time.time()
        self.compute_final_metrics()

    def save_metrics(self, metrics_dir: Path):
        metrics_dir.mkdir(parents=True, exist_ok=True)

        if self.metrics.end_time_s is None:
            self.metrics.end_time_s = time.time()
            self.compute_final_metrics()

        tgoal = self.metrics.end_time_s - self.metrics.start_time_s

        result = {
            "target_tag": self.metrics.target_tag,
            "success": self.metrics.success,
            "fail_reason": self.metrics.fail_reason,
            "time_to_goal_s": tgoal,
            "final_position_error_m": self.metrics.final_position_error_m,
            "final_heading_error_deg": self.metrics.final_heading_error_deg,
            "collision_count": self.metrics.collision_count,
            "loss": self.metrics.loss,
            "final_pose": {
                "x_m": self.pose.x,
                "y_m": self.pose.y,
                "theta_deg": math.degrees(self.pose.theta),
            },
        }

        out_path = metrics_dir / f"navigation_metrics_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        print("\n================ NAVIGATION METRICS ================")
        print(f"Target tag: {result['target_tag']}")
        print(f"Success: {result['success']}")

        if result["fail_reason"]:
            print(f"Reason: {result['fail_reason']}")

        print(f"Time to goal: {result['time_to_goal_s']:.2f} s")
        print(f"Final position error: {result['final_position_error_m']:.3f} m")
        print(f"Final heading error: {result['final_heading_error_deg']:.2f} deg")
        print(f"Collision count: {result['collision_count']}")
        print(f"Loss: {result['loss']:.3f}")
        print(f"Saved metrics: {out_path}")
        print("====================================================\n")

    def draw_status(self, frame: np.ndarray, cmd_speed: int, cmd_steer: int):
        tx, ty = self.target_xy()
        dist_goal = math.hypot(tx - self.pose.x, ty - self.pose.y)

        mode_text = "STRAIGHT+LIDAR" if self.straight_drive else "PATH"

        lines = [
            f"phase={self.phase} mode={mode_text} target={self.target_tag} dist={dist_goal:.2f}m",
            f"pose=({self.pose.x:+.2f},{self.pose.y:+.2f},{math.degrees(self.pose.theta):+.0f} deg)",
            f"cmd=({cmd_speed},{cmd_steer})",
            self.last_centering_message,
            self.last_front_message,
            self.status_message,
            "Keys: G=start driving after switching to R2D2 | Q=quit",
        ]

        y = 25

        for line in lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 25

    def run(self, camera_index: int, metrics_dir: Path):
        cap = None
        detector = None
        object_points = None

        if not self.no_camera:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera index {camera_index}")

            detector = build_apriltag_detector()
            object_points = tag_object_points()
            print("[CAM] Camera opened. Show a mapped AprilTag to localize.")
        else:
            print("[CAM] No-camera mode enabled.")

        if self.pose_initialized and not self.path:
            self.plan_path()

        if self.pose_initialized:
            self.phase = "READY"
            self.status_message = "Pose ready. Switch to R2D2 and press G."

        nav_started_s = time.time()
        last_control_s = 0.0
        docking_mode = False

        cmd_speed = 0
        cmd_steer = STRAIGHT_STEER_CMD

        cv2.namedWindow("Tag Navigation / Docking", cv2.WINDOW_NORMAL)

        try:
            while True:
                now = time.time()
                observations: List[TagObservation] = []
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                if cap is not None:
                    ret, cam_frame = cap.read()

                    if ret and cam_frame is not None:
                        frame = cam_frame
                        observations = detect_tags(frame, detector, object_points)
                        self.correct_pose_from_tags(observations)
                    else:
                        cv2.putText(
                            frame,
                            "Camera frame unavailable. This is okay after switching to R2D2.",
                            (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                if self.pose_initialized and not self.path:
                    try:
                        self.plan_path()
                        self.phase = "READY"
                        self.status_message = "Localized. Switch to R2D2, close mapper/GUI, press G."
                    except Exception as exc:
                        self.status_message = f"Planning failed: {exc}"

                if self.phase == "DRIVE":
                    if self.io is None:
                        self.status_message = "No UDP connection. Press G again after switching to R2D2."
                    else:
                        sensor = self.io.read_latest_sensor()
                        distances_mm = None

                        if sensor is not None:
                            encoder, applied_steer, distances_mm = sensor
                            self.update_odometry(encoder, applied_steer)

                        target_observations = [
                            obs for obs in observations
                            if obs.tag_id == self.target_tag
                        ]
                        target_obs = (
                            sorted(target_observations, key=lambda o: o.range_m)[0]
                            if target_observations
                            else None
                        )

                        tx, ty = self.target_xy()
                        dist_to_goal_xy = math.hypot(tx - self.pose.x, ty - self.pose.y)

                        if target_obs is not None and (
                            target_obs.range_m <= DOCK_START_RANGE_M
                            or dist_to_goal_xy <= 1.0
                        ):
                            docking_mode = True

                        if docking_mode and target_obs is not None:
                            cmd_speed, cmd_steer, docked = self.docking_command(target_obs)

                            if docked:
                                self.finish(True, "Docked visually at target tag.")
                                print("[DONE] Docked visually at target tag.")
                                break

                        else:
                            if dist_to_goal_xy <= GOAL_RADIUS_M:
                                self.finish(True, "Reached target coordinate.")
                                print("[DONE] Reached target coordinate.")
                                break

                            if self.straight_drive:
                                cmd_speed, cmd_steer = self.centered_straight_drive_command(distances_mm)
                            else:
                                cmd_speed, cmd_steer = self.path_follow_command()

                        if now - last_control_s >= CONTROL_DT_S:
                            cmd_speed, should_stop = self.apply_front_safety(distances_mm, cmd_speed)

                            if should_stop:
                                self.finish(
                                    False,
                                    f"Emergency stop: front obstacle/wall too close. {self.last_front_message}",
                                )
                                print(f"[EMERGENCY STOP] {self.last_front_message}")
                                break

                            ok = self.io.send_command(cmd_speed, cmd_steer, 0)

                            if not ok:
                                self.status_message = f"UDP send failed: {self.io.last_error}"

                            last_control_s = now

                            if distances_mm is not None:
                                self.update_collision_metric(distances_mm, cmd_speed)

                        if now - nav_started_s > MAX_NAV_TIME_S:
                            self.finish(False, "Navigation timeout.")
                            print("[STOP] Navigation timeout.")
                            break

                self.draw_status(frame, cmd_speed, cmd_steer)
                cv2.imshow("Tag Navigation / Docking", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    self.finish(False, "User aborted.")
                    print("[STOP] User aborted.")
                    break

                if key == ord("g"):
                    if not self.pose_initialized:
                        self.status_message = "Cannot start yet. Show a known AprilTag first."
                        print("[WAIT] Not localized yet.")
                    else:
                        if not self.path:
                            try:
                                self.plan_path()
                            except Exception as exc:
                                self.status_message = f"Planning failed: {exc}"
                                print(f"[PLAN ERROR] {exc}")
                                continue

                        if self.open_robot_io():
                            self.phase = "DRIVE"
                            self.status_message = "Driving. Camera may disappear; that is okay."
                            nav_started_s = time.time()
                            print("[START] Driving started.")

        finally:
            if self.io is not None:
                self.io.close()

            if cap is not None:
                cap.release()

            cv2.destroyAllWindows()
            self.save_metrics(metrics_dir)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Navigate to and dock at a mapped AprilTag.")

    parser.add_argument("--target-tag", type=int, required=True)
    parser.add_argument("--map-yaml", type=str, default=None)
    parser.add_argument("--tag-map", type=str, default=None)
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX_DEFAULT)
    parser.add_argument("--arduino-ip", type=str, default=ARDUINO_IP_DEFAULT)
    parser.add_argument("--arduino-port", type=int, default=ARDUINO_PORT_DEFAULT)
    parser.add_argument("--local-udp-port", type=int, default=LOCAL_UDP_PORT_DEFAULT)
    parser.add_argument("--start-heading-deg", type=float, default=0.0)
    parser.add_argument("--start-tag", type=int, default=None)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument(
        "--straight-drive",
        action="store_true",
        help="After localization and pressing G, drive at speed=50 with LiDAR side-wall centering around steer=-3.",
    )
    parser.add_argument("--metrics-dir", type=str, default="navigation_metrics")

    args = parser.parse_args()

    pgm, resolution, origin_x, origin_y, yaml_path = load_map(args.map_yaml)
    tag_map, tag_json_path = load_tag_map(args.tag_map)
    obstacle_grid = build_obstacle_grid(pgm, resolution)

    print("================ TAG NAVIGATION ================")
    print(f"Map YAML: {yaml_path}")
    print(f"Tag map: {tag_json_path}")
    print(f"Known tags: {sorted(tag_map.keys())}")
    print(f"Target tag: {args.target_tag}")
    print(f"Arduino: {args.arduino_ip}:{args.arduino_port}")
    print("Step 1: stay on WiFi/USB where camera works.")
    print("Step 2: show a known mapped AprilTag.")
    print("Step 3: after localization, switch to R2D2.")
    print("Step 4: close mapper/GUI, then press G in the OpenCV window.")
    print("================================================")

    navigator = TagNavigator(
        tag_map=tag_map,
        obstacle_grid=obstacle_grid,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        target_tag=args.target_tag,
        start_heading_deg=args.start_heading_deg,
        arduino_ip=args.arduino_ip,
        arduino_port=args.arduino_port,
        local_udp_port=args.local_udp_port,
        start_tag=args.start_tag,
        no_camera=args.no_camera,
        straight_drive=args.straight_drive,
    )

    navigator.run(camera_index=args.camera_index, metrics_dir=Path(args.metrics_dir))


if __name__ == "__main__":
    main()
