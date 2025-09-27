import cv2
import time
import shutil
import argparse
import os, sys
import numpy as np
import pygame
from botconnect import BotConnect
import json
import heapq
from collections import deque

# ---- SLAM components (M2) ----
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import DriveMeasurement
from slam.ekf import EKF
from slam.robot import Robot
from slam.aruco_sensor import ArucoSensor

# ---- CV components (M3) ----
sys.path.insert(0, "{}/cv/".format(os.getcwd()))
from cv.detector import ObjectDetector


# ===============================
#        A* Pathfinder (from your working code)
# ===============================
class AStarPathfinder:
    """Standard A* pathfinding on 8-connected grid"""
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
        
    def heuristic(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)

    def get_neighbors(self, node):
        r, c = node
        nbrs = []
        for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.rows and 0 <= cc < self.cols and self.grid[rr, cc] == 0:
                nbrs.append((rr, cc))
        return nbrs

    def has_line_of_sight(self, a, b):
        r0, c0 = a
        r1, c1 = b
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 >= r0 else -1
        sc = 1 if c1 >= c0 else -1

        r, c = r0, c0
        if dc > dr:
            err = dc // 2
            while c != c1:
                if self.grid[r, c] == 1:
                    return False
                c += sc
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
        else:
            err = dr // 2
            while r != r1:
                if self.grid[r, c] == 1:
                    return False
                r += sr
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
        return self.grid[r1, c1] == 0

    def find_path(self, start, goal):
        if self.grid[start[0], start[1]] != 0 or self.grid[goal[0], goal[1]] != 0:
            return []

        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.heuristic(start, goal)}
        in_open = {start}

        while open_set:
            _, current = heapq.heappop(open_set)
            in_open.discard(current)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + self.heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    if neighbor not in in_open:
                        heapq.heappush(open_set, (f, neighbor))
                        in_open.add(neighbor)

        return []


# ===============================
#    Path Smoothing (from your working code)
# ===============================
def advanced_smooth_path(path_world, world_to_grid_fn, has_los_fn, iterations=3):
    if len(path_world) < 3:
        return path_world
    smoothed = shortcut_path_world(path_world, world_to_grid_fn, has_los_fn, trials=200)
    smoothed = remove_redundant_points(smoothed, world_to_grid_fn, has_los_fn)
    smoothed = chaikin_smooth(smoothed, iterations=iterations, alpha=0.25)
    smoothed = optimize_turning_points(smoothed, world_to_grid_fn, has_los_fn)
    return smoothed

def remove_redundant_points(path_world, world_to_grid_fn, has_los_fn, angle_threshold=0.1):
    if len(path_world) < 3:
        return path_world
    result = [path_world[0]]
    for i in range(1, len(path_world) - 1):
        p0 = np.array(path_world[i - 1])
        p1 = np.array(path_world[i])
        p2 = np.array(path_world[i + 1])
        v1 = p1 - p0
        v2 = p2 - p1
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            v1 = v1 / n1
            v2 = v2 / n2
            dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(dot_product)
            if angle > angle_threshold:
                result.append(path_world[i])
        else:
            result.append(path_world[i])
    result.append(path_world[-1])
    return result

def optimize_turning_points(path_world, world_to_grid_fn, has_los_fn):
    if len(path_world) < 3:
        return path_world
    result = [path_world[0]]
    i = 1
    while i < len(path_world):
        best_j = i
        max_j = min(i + 5, len(path_world) - 1)
        for j in range(max_j, i, -1):
            if j >= len(path_world):
                continue
            grid_i = world_to_grid_fn(result[-1])
            grid_j = world_to_grid_fn(path_world[j])
            if has_los_fn(grid_i, grid_j):
                best_j = j
                break
        if best_j > i:
            result.append(path_world[best_j])
            i = best_j + 1
        else:
            result.append(path_world[i])
            i += 1
    if result[-1] != path_world[-1]:
        result.append(path_world[-1])
    return result

def shortcut_path_world(path_world, world_to_grid_fn, has_los_fn, trials=200):
    if len(path_world) < 3:
        return path_world
    pts = path_world[:]
    for _ in range(trials):
        if len(pts) < 3:
            break
        i = np.random.randint(0, len(pts) - 2)
        j = np.random.randint(i + 2, len(pts))
        a_grid = world_to_grid_fn(pts[i])
        b_grid = world_to_grid_fn(pts[j])
        if has_los_fn(a_grid, b_grid):
            pts = pts[:i+1] + pts[j:]
    return pts

def chaikin_smooth(path_world, iterations=1, alpha=0.25):
    if len(path_world) < 3 or iterations <= 0:
        return path_world
    out = path_world[:]
    for _ in range(iterations):
        newp = [out[0]]
        for k in range(len(out) - 1):
            p = np.array(out[k], dtype=float)
            q = np.array(out[k+1], dtype=float)
            Q = (1 - alpha) * p + alpha * q
            R = alpha * p + (1 - alpha) * q
            newp.extend([tuple(Q), tuple(R)])
        newp.append(out[-1])
        out = newp
    return out


# ===============================
#     Operate (Combined System)
# ===============================
class Operate:
    def __init__(self, args):
        self.botconnect = BotConnect(args.ip)
        self.command = {
            'wheel_speed': [0, 0],
            'save_slam': False,
            'run_obj_detector': False,
            'save_obj_detector': False,
            'save_image': False
        }
        self.botconnect.set_pid(use_pid=1, kp=2.5, ki=0, kd=0)

        self.lab_output_dir = 'lab_output/'
        if not os.path.exists(self.lab_output_dir):
            os.makedirs(self.lab_output_dir)

        # SLAM
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_sensor = ArucoSensor(self.ekf.robot, marker_length=0.06)

        # True map
        self.true_map = self.load_true_map()
        self.grid_resolution = 0.05
        self.grid_size = int(2.5 / self.grid_resolution)
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size))
        self.path_grid = []
        self.obstacle_radius_m = 0.22
        self.create_occupancy_grid()

        # Pathfinder
        self.pathfinder = AStarPathfinder(self.occupancy_grid)

        # Navigation state
        self.autonomous_mode = False
        self.localization_complete = False
        self.target_point = None
        self.path = []
        self.current_path_index = 0

        # Localization
        self.localization_rotation_speed = 0.3
        self.localization_start_time = None
        self.relocalization_interval = 5.0
        self.last_relocalization_time = 0
        self.in_relocalization = False
        self.saved_path = []
        self.saved_path_index = 0
        self.saved_target = None

        # Step-wise rotation
        self.rotation_step_duration = 0.3
        self.scan_pause_duration = 0.8
        self.current_rotation_step = 0
        self.max_rotation_steps = 16
        self.step_start_time = None
        self.in_rotation_phase = False

        # Detector
        if args.ckpt == "":
            self.obj_detector = None
            self.cv_vis = cv2.imread('ui/8bit/detector_splash.png')
            if self.cv_vis is None:
                self.cv_vis = np.ones((480, 640, 3), dtype=np.uint8) * 50
        else:
            self.obj_detector = ObjectDetector(args.ckpt, use_gpu=False)
            self.cv_vis = np.ones((480, 640, 3), dtype=np.uint8) * 100

        self.raw_img_dir = 'raw_images/'
        if not os.path.exists(self.raw_img_dir):
            os.makedirs(self.raw_img_dir)
        else:
            shutil.rmtree(self.raw_img_dir)
            os.makedirs(self.raw_img_dir)

        # Smoothing
        self.enable_smoothing = True
        self.smoothing_iterations = 2

        # Pure pursuit
        self.use_pure_pursuit = True
        self.lookahead = 0.25
        self.pp_max_linear = 0.25  # Reduced from 0.5 to 0.25 (half speed)
        self.pp_max_angular = 0.8
        self.wp_reached_radius = 0.12

        # UI state
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.obj_detector_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press L to start localization'
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        self.img = np.zeros([480, 640, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([480, 640, 3], dtype=np.uint8)
        self.obj_detector_pred = np.zeros([480, 640], dtype=np.uint8)
        self.bg = pygame.image.load('ui/gui_mask.jpg')

    def load_true_map(self):
        try:
            with open('truemap.txt', 'r') as f:
                true_map = json.load(f)
            return true_map
        except FileNotFoundError:
            print("[WARN] truemap.txt not found!")
            return {}

    def create_occupancy_grid(self):
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        obstacle_radius = self.obstacle_radius_m
        grid_radius = max(1, int(obstacle_radius / self.grid_resolution))

        for _, pos in self.true_map.items():
            grid_x = int((pos['x'] + 1.25) / self.grid_resolution)
            grid_y = int((pos['y'] + 1.25) / self.grid_resolution)
            for dx in range(-grid_radius, grid_radius + 1):
                for dy in range(-grid_radius, grid_radius + 1):
                    if dx*dx + dy*dy <= grid_radius*grid_radius:
                        nx, ny = grid_x + dx, grid_y + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            self.occupancy_grid[ny, nx] = 1

    def world_to_grid(self, world_pos):
        grid_x = int((world_pos[0] + 1.25) / self.grid_resolution)
        grid_y = int((world_pos[1] + 1.25) / self.grid_resolution)
        grid_x = max(0, min(grid_x, self.grid_size - 1))
        grid_y = max(0, min(grid_y, self.grid_size - 1))
        return (grid_y, grid_x)

    def grid_to_world(self, grid_pos):
        world_x = grid_pos[1] * self.grid_resolution - 1.25
        world_y = grid_pos[0] * self.grid_resolution - 1.25
        return (world_x, world_y)

    def start_localization(self):
        self.autonomous_mode = True
        self.localization_complete = False
        self.localization_start_time = time.time()
        self.ekf_on = True
        self.current_rotation_step = 0
        self.step_start_time = time.time()
        self.in_rotation_phase = True
        self.notification = "Initial localization... Step 1/16"
        try:
            self.ekf.load_map('truemap.txt')
        except Exception:
            pass

    def start_relocalization(self):
        if self.autonomous_mode and self.localization_complete:
            self.in_relocalization = True
            self.current_rotation_step = 0
            self.step_start_time = time.time()
            self.in_rotation_phase = True
            self.saved_path = self.path.copy()
            self.saved_path_index = self.current_path_index
            self.saved_target = self.target_point
            self.notification = "Re-localizing position... Step 1/16"

    def perform_localization(self):
        if not self.autonomous_mode:
            return

        now = time.time()

        if not self.localization_complete:
            if self.current_rotation_step < self.max_rotation_steps:
                self.perform_step_wise_rotation(now, is_initial=True)
            else:
                self.command['wheel_speed'] = [0, 0]
                if len(self.ekf.taglist) >= 2:
                    self.localization_complete = True
                    self.last_relocalization_time = now
                    self.notification = f"Initial localization complete! Found {len(self.ekf.taglist)} markers"
                    self.reset_rotation_variables()
                else:
                    self.notification = f"Initial localization failed! Only {len(self.ekf.taglist)} markers found"
                    self.autonomous_mode = False
                    self.localization_complete = False
                    self.reset_rotation_variables()

        elif self.in_relocalization:
            if self.current_rotation_step < self.max_rotation_steps:
                self.perform_step_wise_rotation(now, is_initial=False)
            else:
                # Stop moving after re-localization completes
                self.command['wheel_speed'] = [0, 0]

                self.in_relocalization = False
                self.reset_rotation_variables()
                self.last_relocalization_time = time.time()

                resume_msg = f"Re-localized! Found {len(self.ekf.taglist)} markers."

                resumed = False

                if self.saved_target is not None:
                    if self.plan_path_to_target(self.saved_target):
                        self.autonomous_mode = True
                        self.target_point = self.saved_target
                        self.notification = resume_msg + " Re-planned route to target."
                        resumed = True
                    else:
                        self.notification = resume_msg + " Unable to re-plan path, trying saved route."

                if not resumed and self._restore_saved_path():
                    self.notification = resume_msg + " Resuming saved path."
                    resumed = True

                if not resumed:
                    self.autonomous_mode = False
                    self.path = []
                    self.current_path_index = 0
                    self.target_point = None
                    self.notification = resume_msg + " Awaiting new goal."

                self.saved_path = []
                self.saved_path_index = 0
                self.saved_target = None

    def perform_step_wise_rotation(self, now, is_initial=True):
        if self.step_start_time is None:
            self.step_start_time = now

        elapsed = now - self.step_start_time

        if self.in_rotation_phase:
            if elapsed < self.rotation_step_duration:
                rs = self.localization_rotation_speed * 0.8
                self.command['wheel_speed'] = [-rs, rs]
            else:
                self.command['wheel_speed'] = [0, 0]
                self.in_rotation_phase = False
                self.step_start_time = now
                step_num = self.current_rotation_step + 1
                tag_ct = len(self.ekf.taglist)
                self.notification = f"{'Initial' if is_initial else 'Re-'}localizing... Scanning step {step_num}/16 - Markers: {tag_ct}"
        else:
            if elapsed < self.scan_pause_duration:
                self.command['wheel_speed'] = [0, 0]
            else:
                self.current_rotation_step += 1
                self.in_rotation_phase = True
                self.step_start_time = now
                if self.current_rotation_step < self.max_rotation_steps:
                    step_num = self.current_rotation_step + 1
                    tag_ct = len(self.ekf.taglist)
                    self.notification = f"{'Initial' if is_initial else 'Re-'}localizing... Step {step_num}/16 - Markers: {tag_ct}"

    def reset_rotation_variables(self):
        self.current_rotation_step = 0
        self.step_start_time = None
        self.in_rotation_phase = False

    def should_relocalize(self):
        return (
            self.localization_complete and
            not self.in_relocalization and
            self.autonomous_mode and
            (time.time() - self.last_relocalization_time > self.relocalization_interval)
        )

    def plan_path_to_target(self, target_world):
        if not self.localization_complete:
            self.notification = "Complete localization first!"
            return False

        robot_pos = (self.ekf.robot.state[0, 0], self.ekf.robot.state[1, 0])
        start_grid = self.world_to_grid(robot_pos)
        goal_grid = self.world_to_grid(target_world)

        self.path_grid = self.pathfinder.find_path(start_grid, goal_grid)

        if not self.path_grid:
            self.notification = "No path found to target!"
            return False

        self.path = [self.grid_to_world(gp) for gp in self.path_grid]

        if self.enable_smoothing:
            original_length = len(self.path)
            self.path = advanced_smooth_path(
                self.path,
                self.world_to_grid,
                self.pathfinder.has_line_of_sight,
                iterations=self.smoothing_iterations
            )
            self.notification = f"A* Path: {original_length} → {len(self.path)} waypoints (smoothed)"
        else:
            self.notification = f"A* Path: {len(self.path)} waypoints"

        self.current_path_index = 0
        return True

    def _restore_saved_path(self):
        if not self.saved_path:
            return False

        self.path = self.saved_path.copy()

        if not self.path:
            return False

        rx = float(self.ekf.robot.state[0, 0])
        ry = float(self.ekf.robot.state[1, 0])

        start_idx = min(max(self.saved_path_index, 0), len(self.path) - 1)
        best_idx = start_idx
        best_dist = float('inf')

        for idx in range(start_idx, len(self.path)):
            px, py = self.path[idx]
            dist = np.hypot(px - rx, py - ry)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
            if best_dist <= self.wp_reached_radius * 0.5:
                break

        self.current_path_index = best_idx

        # Skip waypoints that we are already on top of after re-localization
        while (
            self.current_path_index < len(self.path) - 1 and
            np.hypot(self.path[self.current_path_index][0] - rx,
                     self.path[self.current_path_index][1] - ry) <= self.wp_reached_radius * 0.5
        ):
            self.current_path_index += 1

        if self.saved_target is not None:
            self.target_point = self.saved_target
        else:
            self.target_point = self.path[-1]

        self.autonomous_mode = True
        return True

    def _find_lookahead_target(self, pose, path, lookahead):
        rx, ry = pose[0], pose[1]
        start_idx = max(0, self.current_path_index)

        acc_dist = 0.0
        for i in range(start_idx, len(path) - 1):
            p = np.array(path[i], dtype=float)
            q = np.array(path[i + 1], dtype=float)
            seg = q - p
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-6:
                continue

            d_to_p = np.hypot(rx - p[0], ry - p[1])
            d_to_q = np.hypot(rx - q[0], ry - q[1])

            if d_to_p <= lookahead <= d_to_q:
                return float(q[0]), float(q[1]), i + 1

            acc_dist += seg_len
            if acc_dist >= lookahead:
                return float(q[0]), float(q[1]), i + 1

        gx, gy = path[-1]
        return float(gx), float(gy), len(path) - 1

    def follow_path(self):
        if self.in_relocalization:
            return

        if not self.path or self.current_path_index >= len(self.path):
            self.command['wheel_speed'] = [0, 0]
            if self.autonomous_mode and not self.in_relocalization:
                self.notification = "Reached target!"
                self.autonomous_mode = False
            return

        robot_x = float(self.ekf.robot.state[0, 0])
        robot_y = float(self.ekf.robot.state[1, 0])
        robot_theta = float(self.ekf.robot.state[2, 0])

        tx, ty = self.path[self.current_path_index]
        if np.hypot(tx - robot_x, ty - robot_y) < self.wp_reached_radius:
            self.current_path_index += 1
            if self.current_path_index >= len(self.path):
                self.command['wheel_speed'] = [0, 0]
                self.notification = "Reached target!"
                self.autonomous_mode = False
                return

        if not self.use_pure_pursuit:
            target_x, target_y = self.path[self.current_path_index]
            dx = target_x - robot_x
            dy = target_y - robot_y
            distance = np.hypot(dx, dy)
            desired_theta = np.arctan2(dy, dx)
            angle_error = (desired_theta - robot_theta + np.pi) % (2 * np.pi) - np.pi
            linear_speed = min(0.15, distance)  # Reduced from 0.3 to 0.15 (half speed)
            angular_speed = np.clip(angle_error * 1.0, -0.8, 0.8)
            baseline = float(self.ekf.robot.baseline)
            left = np.clip(linear_speed - angular_speed * baseline / 2.0, -0.6, 0.6)
            right = np.clip(linear_speed + angular_speed * baseline / 2.0, -0.6, 0.6)
            self.command['wheel_speed'] = [left, right]
        else:
            lx, ly, idx = self._find_lookahead_target(
                (robot_x, robot_y, robot_theta),
                self.path,
                self.lookahead
            )
            if idx > self.current_path_index:
                self.current_path_index = idx

            dx = lx - robot_x
            dy = ly - robot_y

            x_r = np.cos(robot_theta) * dx + np.sin(robot_theta) * dy
            y_r = -np.sin(robot_theta) * dx + np.cos(robot_theta) * dy

            Ld = max(0.05, self.lookahead)
            kappa = (2.0 * y_r) / (Ld * Ld)

            v = self.pp_max_linear
            omega = np.clip(v * kappa, -self.pp_max_angular, self.pp_max_angular)

            baseline = float(self.ekf.robot.baseline)
            left = np.clip(v - omega * baseline / 2.0, -self.pp_max_linear, self.pp_max_linear)
            right = np.clip(v + omega * baseline / 2.0, -self.pp_max_linear, self.pp_max_linear)
            self.command['wheel_speed'] = [left, right]

        t_rl = self.relocalization_interval - (time.time() - self.last_relocalization_time)
        self.notification = (
            f"Following path: waypoint {self.current_path_index + 1}/{len(self.path)} | "
            f"Next relocalize: {max(0, t_rl):.1f}s"
        )

    def handle_slam_click(self, pos):
        if not self.localization_complete:
            self.notification = "Complete localization first!"
            return

        h_pad, v_pad = 20, 40
        slam_x = pos[0] - (2 * h_pad + 320)
        slam_y = pos[1] - v_pad

        if slam_x < 0 or slam_x > 520 or slam_y < 0 or slam_y > 480 + v_pad:
            return

        m2pixel = 100
        world_x = -(slam_x - 520 / 2.0) / m2pixel
        world_y = (slam_y - (480 + v_pad) / 2.0) / m2pixel

        robot_x = self.ekf.robot.state[0, 0]
        robot_y = self.ekf.robot.state[1, 0]
        target_world = (robot_x + world_x, robot_y + world_y)

        if abs(target_world[0]) > 1.25 or abs(target_world[1]) > 1.25:
            self.notification = "Target outside arena bounds!"
            return

        if self.plan_path_to_target(target_world):
            self.autonomous_mode = True
            self.target_point = target_world
            self.last_relocalization_time = time.time()

    def control(self):
        left_speed, right_speed = self.botconnect.set_velocity(self.command['wheel_speed'])
        dt = time.time() - self.control_clock
        drive_measurement = DriveMeasurement(left_speed, right_speed, dt)
        self.control_clock = time.time()
        return drive_measurement

    def take_pic(self):
        self.img = self.botconnect.get_image()

    def init_ekf(self, calib_dir, ip):
        fileK = os.path.join(calib_dir, 'intrinsic.txt')
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = os.path.join(calib_dir, 'distCoeffs.txt')
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = os.path.join(calib_dir, 'scale.txt')
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = os.path.join(calib_dir, 'baseline.txt')
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    def perform_slam(self, drive_measurement):
        sensor_measurement, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(sensor_measurement)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on:
            self.ekf.predict(drive_measurement)
            self.ekf.add_landmarks(sensor_measurement)
            self.ekf.update(sensor_measurement)

    def save_result(self):
        if self.command['save_slam']:
            self.ekf.save_map(fname=os.path.join(self.lab_output_dir, 'slam.txt'))
            self.notification = 'Map is saved'
            self.command['save_slam'] = False

        if self.command['save_obj_detector']:
            if self.obj_detector_output is not None:
                self.pred_fname = self.obj_detector.write_image(
                    self.obj_detector_output[0],
                    self.obj_detector_output[1],
                    self.lab_output_dir
                )
                self.notification = f'Prediction is saved to {self.pred_fname}'
            else:
                self.notification = 'No prediction in buffer, save ignored'
            self.command['save_obj_detector'] = False

        if self.command['save_image']:
            image = self.botconnect.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            f_ = os.path.join(self.raw_img_dir, f'img_{self.image_id}.png')
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    def detect_object(self):
        if self.command['run_obj_detector'] and self.obj_detector is not None:
            self.obj_detector_pred, self.cv_vis = self.obj_detector.detect_single_image(self.img)
            self.command['run_obj_detector'] = False
            self.obj_detector_output = (self.obj_detector_pred, self.ekf.robot.state.tolist())
            self.notification = f'{len(np.unique(self.obj_detector_pred)) - 1} object type(s) detected'

    def draw(self, canvas):
        width, height = 900, 660
        canvas = pygame.display.set_mode((width, height))
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad, h_pad = 40, 20

        ekf_view = self.ekf.draw_slam_state(res=(520, 480 + v_pad), not_pause=self.ekf_on)

        if self.path and hasattr(self, 'path_grid'):
            self.draw_path_on_slam(ekf_view)

        canvas.blit(ekf_view, (2 * h_pad + 320, v_pad))

        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, position=(h_pad, v_pad))

        detector_view = cv2.resize(self.cv_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, position=(h_pad, 240 + 2 * v_pad))

        self.put_caption(canvas, caption='SLAM', position=(2 * h_pad + 320, v_pad))
        self.put_caption(canvas, caption='Detector', position=(h_pad, 240 + 2 * v_pad))
        self.put_caption(canvas, caption='Robot Cam', position=(h_pad, v_pad))

        if self.autonomous_mode:
            if not self.localization_complete or self.in_relocalization:
                notification_text = f"{self.notification} - Markers: {len(self.ekf.taglist)}"
            else:
                notification_text = self.notification
        else:
            notification_text = self.notification

        notification = TEXT_FONT.render(notification_text, False, text_colour)
        canvas.blit(notification, (h_pad + 10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain) % 2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2 * h_pad + 320 + 5, 530))
        return canvas

    def draw_path_on_slam(self, surface):
        if not self.path or not self.ekf_on:
            return

        m2pixel = 100
        res = (520, 480 + 40)

        robot_xy = self.ekf.robot.state[:2, 0].reshape((2, 1))

        path_points = []
        for world_pos in self.path:
            rel_x = world_pos[0] - robot_xy[0, 0]
            rel_y = world_pos[1] - robot_xy[1, 0]
            x_im = int(-rel_x * m2pixel + res[0] / 2.0)
            y_im = int(rel_y * m2pixel + res[1] / 2.0)
            if 0 <= x_im < res[0] and 0 <= y_im < res[1]:
                path_points.append((x_im, y_im))

        if len(path_points) > 1:
            if self.in_relocalization:
                pygame.draw.lines(surface, (255, 165, 0), False, path_points, 3)
            else:
                pygame.draw.lines(surface, (0, 255, 0), False, path_points, 3)

        if (self.current_path_index < len(path_points) and
                self.current_path_index < len(self.path)):
            target_point = path_points[self.current_path_index]
            color = (255, 165, 0) if self.in_relocalization else (255, 0, 0)
            pygame.draw.circle(surface, color, target_point, 8)

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)

    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption, False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1] - 25))

    def update_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.localization_complete:
                    self.handle_slam_click(pygame.mouse.get_pos())

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                if not self.autonomous_mode:
                    self.start_localization()

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                if self.autonomous_mode:
                    self.autonomous_mode = False
                    self.in_relocalization = False
                    self.reset_rotation_variables()
                    self.command['wheel_speed'] = [0, 0]
                    self.notification = "Autonomous mode stopped"

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                if self.localization_complete and not self.in_relocalization:
                    self.start_relocalization()

            elif not self.autonomous_mode:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    self.command['wheel_speed'] = [0.3, 0.3]
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                    self.command['wheel_speed'] = [-0.3, -0.3]
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    self.command['wheel_speed'] = [-0.3, 0.3]
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                    self.command['wheel_speed'] = [0.3, -0.3]

            if event.type == pygame.KEYUP or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                if not self.autonomous_mode:
                    self.command['wheel_speed'] = [0, 0]

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '>2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '>2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    self.notification = 'SLAM is running' if self.ekf_on else 'SLAM is paused'

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['save_slam'] = True

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm += 1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
                    self.localization_complete = False
                    self.autonomous_mode = False
                    self.in_relocalization = False
                    self.reset_rotation_variables()

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['run_obj_detector'] = True

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_obj_detector'] = True

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True

            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True

        if self.quit:
            pygame.quit()
            sys.exit()

    def run_autonomous_system(self):
        if self.autonomous_mode:
            if self.should_relocalize() and not self.in_relocalization:
                self.start_relocalization()

            if not self.localization_complete:
                self.perform_localization()
            elif self.in_relocalization:
                self.perform_localization()
            else:
                self.follow_path()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--ckpt", default='cv/model/model.best.pt')
    args, _ = parser.parse_known_args()

    pygame.font.init()
    TITLE_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 40)

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 Lab - Combined: Good Localization + A* Pathfinding')
    pygame.display.set_icon(pygame.image.load('ui/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('ui/loading.png')
    pibot_animate = [pygame.image.load('ui/8bit/pibot1.png'),
                     pygame.image.load('ui/8bit/pibot2.png'),
                     pygame.image.load('ui/8bit/pibot3.png'),
                     pygame.image.load('ui/8bit/pibot4.png'),
                     pygame.image.load('ui/8bit/pibot5.png')]
    pygame.display.update()

    start = False
    counter = 40
    clock = pygame.time.Clock()

    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter % 10 // 2], (x_, 565))
            pygame.display.update()
            counter += 2
        clock.tick(60)

    operate = Operate(args)

    print("\n=== COMBINED SYSTEM: GOOD LOCALIZATION + YOUR A* PATHFINDING ===")
    print("L - Start initial localization (robot will rotate to find markers)")
    print("Left Click - Click on SLAM map to set target destination")
    print("T - Force immediate re-localization")
    print("X - Stop autonomous mode")
    print("Arrow Keys - Manual control (when not in autonomous mode)")
    print("SPACE - Stop robot")
    print("ENTER - Start/pause SLAM")
    print("S - Save SLAM map")
    print("R - Reset SLAM map (press twice)")
    print("ESC - Quit")
    print("\n=== FEATURES ===")
    print("✓ Step-wise localization with marker scanning")
    print("✓ Your A* pathfinding with octile distance heuristic")
    print("✓ Multi-pass path smoothing from your code")
    print("✓ Pure Pursuit controller for smooth following")
    print("✓ Automatic re-localization every 5 seconds")
    print("✓ Path visualization: GREEN (normal) / ORANGE (re-localization)")
    print("✓ From document 7: Robust localization system")
    print("✓ From document 6: Your working A* + smoothing algorithms")
    print("================================================\n")

    while start:
        operate.update_keyboard()
        operate.take_pic()
        dm = operate.control()
        operate.perform_slam(dm)
        operate.run_autonomous_system()
        operate.save_result()
        operate.detect_object()
        operate.draw(canvas)
        pygame.display.update()
        clock.tick(60)