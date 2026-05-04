# ============================================================================
# Robot Teleoperation and Navigation GUI
# ============================================================================
# This script creates a NiceGUI interface for controlling the robot.
#
# Main tasks:
# - Sends speed, steering, and auto-corridor commands to the Arduino over UDP.
# - Provides manual teleoperation controls for speed and steering.
# - Supports emergency stop from the GUI.
# - Starts and stops the autonomous AprilTag navigation script.
# - Lets the user enter a target AprilTag ID and optional map/tag-map paths.
# - Shows live AprilTag landmark status from the mapper output file.
# - Can request a camera feed restart without restarting the mapper.
# ============================================================================

import socket
import json
from pathlib import Path
import subprocess
import sys
import cv2
import time
import numpy as np
from nicegui import ui, app, run
from fastapi import Response

import parameters

stream_video = False
STRAIGHT_STEER_CMD = -3
TAG_STATUS_PATH = Path("occupancy_maps") / "live_tag_status.json"
CAMERA_RESTART_REQUEST_PATH = Path("occupancy_maps") / "restart_camera_request.json"

def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()


@ui.page('/')
def main():
    dark = ui.dark_mode()
    dark.value = True

    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    nav_state = {'process': None}

    if stream_video:
        video_capture = cv2.VideoCapture(parameters.camera_id)

        @app.get('/video/frame')
        async def grab_video_frame() -> Response:
            if not video_capture.isOpened():
                return Response(content=b'', media_type='image/jpeg')
            _, frame = await run.io_bound(video_capture.read)
            if frame is None:
                return Response(content=b'', media_type='image/jpeg')
            jpeg = await run.cpu_bound(convert, frame)
            return Response(content=jpeg, media_type='image/jpeg')

    def send_command(speed: int, steer: int, auto_mode: int) -> None:
        msg = f'{speed},{steer},{auto_mode}'.encode('utf-8')
        send_sock.sendto(msg, (parameters.arduinoIP, parameters.arduinoPort))

    def nav_is_running() -> bool:
        proc = nav_state.get('process')
        return proc is not None and proc.poll() is None

    def start_navigation() -> None:
        if nav_is_running():
            status_label.set_text('Navigation is already running')
            return

        target_text = str(target_tag_input.value).strip()
        if not target_text:
            status_label.set_text('Enter a target AprilTag ID first')
            return

        # Stop manual teleop before launching autonomous navigation.
        speed_switch.value = False
        steering_switch.value = False
        auto_corridor_switch.value = False
        slider_speed.value = 0
        slider_steering.value = STRAIGHT_STEER_CMD
        stop_navigation()
        send_command(0, STRAIGHT_STEER_CMD, 0)

        cmd = [
            sys.executable,
            'tag_navigation.py',
            '--target-tag',
            target_text,
            '--camera-index',
            str(parameters.camera_id),
            '--arduino-ip',
            parameters.arduinoIP,
            '--arduino-port',
            str(parameters.arduinoPort),
        ]

        start_heading_text = str(start_heading_input.value).strip()
        if start_heading_text:
            cmd += ['--start-heading-deg', start_heading_text]

        if str(map_yaml_input.value).strip():
            cmd += ['--map-yaml', str(map_yaml_input.value).strip()]

        if str(tag_map_input.value).strip():
            cmd += ['--tag-map', str(tag_map_input.value).strip()]

        nav_state['process'] = subprocess.Popen(cmd)
        status_label.set_text(f'Navigation started to tag {target_text}')

    def stop_navigation() -> None:
        proc = nav_state.get('process')
        if proc is not None and proc.poll() is None:
            proc.terminate()
            status_label.set_text('Navigation process stopped')
        else:
            status_label.set_text('No navigation process running')

        nav_state['process'] = None
        send_command(0, STRAIGHT_STEER_CMD, 0)

    def stop_now():
        speed_switch.value = False
        steering_switch.value = False
        auto_corridor_switch.value = False
        slider_speed.value = 0
        slider_steering.value = STRAIGHT_STEER_CMD
        send_command(0, STRAIGHT_STEER_CMD, 0)
        status_label.set_text('STOP sent')

    def request_phone_feed_restart():
        """
        Ask udp_occupancy_mapper.py to release/reopen the OpenCV feed.

        This does not restart the whole mapper, so the LiDAR map and stored
        AprilTag observations stay alive.
        """
        CAMERA_RESTART_REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.time(),
            "reason": "manual GUI restart button",
        }
        CAMERA_RESTART_REQUEST_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        status_label.set_text('Requested phone camera feed restart')

    def get_command():
        cmd_speed = int(slider_speed.value) if connect_switch.value and speed_switch.value else 0
        auto_mode = 1 if (connect_switch.value and auto_corridor_switch.value) else 0

        # In auto corridor mode, default nominal straight steer is -3 with phone attached
        # even if the steering enable switch is off.
        if connect_switch.value and auto_mode:
            cmd_steer = int(slider_steering.value) if steering_switch.value else STRAIGHT_STEER_CMD
        else:
            cmd_steer = (
                int(slider_steering.value)
                if connect_switch.value and steering_switch.value
                else STRAIGHT_STEER_CMD
            )

        return cmd_speed, cmd_steer, auto_mode

    with ui.card().classes('w-full items-center'):
        ui.label('ROB-GY - 6213: Robot Teleop').style('font-size: 24px;')

    with ui.card().classes('w-full'):
        with ui.grid(columns=3).classes('w-full items-center'):
            with ui.card().classes('w-full items-center h-60'):
                if stream_video:
                    ui.interactive_image('/video/frame').classes('w-full h-full')
                else:
                    ui.image('./a_robot_image.jpg').props('height=2')

            with ui.card().classes('w-full items-center h-60'):
                ui.label('Mapping').style('font-size: 20px;')
                ui.label('Drive robot here. View live map in the mapper window.').style('color: lightgray;')
                ui.label('For AprilTag landmarks, run:').style('color: lightgray;')
                ui.label('python udp_occupancy_mapper.py --tags --camera-index 1').style('color: orange;')
                ui.button('RESTART PHONE FEED', on_click=request_phone_feed_restart).props('color=purple')
                tag_status_label = ui.label('AprilTags: mapper not running / no tags yet').style('color: violet;')

            with ui.card().classes('items-center h-60'):
                connect_switch = ui.switch('Robot Connect')
                auto_corridor_switch = ui.switch('Auto Corridor Centering')
                ui.button('EMERGENCY STOP', on_click=stop_now).props('color=red')
                status_label = ui.label('Disconnected')

    with ui.card().classes('w-full'):
        ui.label('AprilTag Navigation / Docking').style('font-size: 20px;')
        ui.label('Close the mapper before starting this. Navigation will send UDP commands directly.').style('color: lightgray;')

        with ui.grid(columns=4).classes('w-full items-center'):
            target_tag_input = ui.input('Target tag ID', value='')
            start_heading_input = ui.input('Start heading deg', value='0')
            ui.button('START TAG DOCK', on_click=start_navigation).props('color=green')
            ui.button('STOP NAV', on_click=stop_navigation).props('color=orange')

        with ui.grid(columns=2).classes('w-full items-center'):
            map_yaml_input = ui.input(
                'Optional map YAML path',
                value='',
                placeholder='Leave empty to use latest occupancy_maps/map_*.yaml'
            ).classes('w-full')
            tag_map_input = ui.input(
                'Optional tag_map JSON path',
                value='',
                placeholder='Leave empty to use latest occupancy_maps/map_*_tags.json'
            ).classes('w-full')

    with ui.card().classes('w-full'):
        with ui.grid(columns=4).classes('w-full'):
            with ui.card().classes('w-full items-center'):
                ui.label('SPEED:').style('text-align: center;')
            with ui.card().classes('w-full items-center'):
                slider_speed = ui.slider(min=0, max=100, value=0)
            with ui.card().classes('w-full items-center'):
                ui.label().bind_text_from(slider_speed, 'value').style('text-align: center;')
            with ui.card().classes('w-full items-center'):
                speed_switch = ui.switch('Enable')

    with ui.card().classes('w-full'):
        with ui.grid(columns=4).classes('w-full'):
            with ui.card().classes('w-full items-center'):
                ui.label('STEER:').style('text-align: center;')
            with ui.card().classes('w-full items-center'):
                slider_steering = ui.slider(min=-20, max=20, value=STRAIGHT_STEER_CMD)
            with ui.card().classes('w-full items-center'):
                ui.label().bind_text_from(slider_steering, 'value').style('text-align: center;')
            with ui.card().classes('w-full items-center'):
                steering_switch = ui.switch('Enable')

    async def control_loop():
        if nav_is_running():
            status_label.set_text('Autonomous navigation running; teleop disabled')
            return
        speed, steer, auto_mode = get_command()
        send_command(speed, steer, auto_mode)

        if connect_switch.value:
            mode_text = 'AUTO' if auto_mode else 'MANUAL'
            status_label.set_text(f'Sending cmd: speed={speed}, steer={steer}, mode={mode_text}')
        else:
            status_label.set_text('Disconnected / sending zeros')

    ui.timer(0.05, control_loop)

    async def tag_status_loop():
        if not TAG_STATUS_PATH.exists():
            tag_status_label.set_text('AprilTags: no live_tag_status.json yet')
            return

        try:
            data = json.loads(TAG_STATUS_PATH.read_text(encoding='utf-8'))
        except Exception as exc:
            tag_status_label.set_text(f'AprilTags: status read error: {exc}')
            return

        tags = data.get('tags', {})
        seen = data.get('last_seen_tags', [])

        if not tags:
            tag_status_label.set_text(
                f'AprilTags: none stored yet | currently seen: {seen if seen else "none"}'
            )
            return

        pieces = []
        for tag_id, tag in sorted(tags.items(), key=lambda kv: int(kv[0])):
            mark = '✓' if tag.get('confirmed') else '?'
            theta_deg = tag.get(
                "theta_deg",
                tag.get("approach_theta_deg", 0)
            )
            pieces.append(
                f'{mark} ID {tag_id}: n={tag.get("num_observations", 0)} '
                f'x={tag.get("x_m", tag.get("x", 0)):+.2f}, '
                f'y={tag.get("y_m", tag.get("y", 0)):+.2f}, '
                f'θ={theta_deg:+.0f}°'
            )

        tag_status_label.set_text('AprilTags: ' + ' | '.join(pieces[:3]))

    ui.timer(0.5, tag_status_loop)

ui.run(native=True)