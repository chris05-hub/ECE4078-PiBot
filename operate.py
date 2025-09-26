import cv2
import time
import json
import math
import heapq
import shutil
import argparse
import os, sys
import numpy as np
import pygame # python package for GUI
from botconnect import BotConnect # access the robot communication

# import SLAM components (M2)
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import DriveMeasurement
from slam.ekf import EKF
from slam.robot import Robot
from slam.aruco_sensor import ArucoSensor

# import CV components (M3)
sys.path.insert(0,"{}/cv/".format(os.getcwd()))
from cv.detector import ObjectDetector


class GridAStar:
    """Grid-based A* planner that keeps obstacles inflated by a safety margin."""

    def __init__(self, arena_size=2.5, resolution=0.05, obstacles=None):
        self.arena_size = float(arena_size)
        self.resolution = float(resolution)
        self.half_extent = self.arena_size / 2.0
        self.min_coord = -self.half_extent
        self.max_coord = self.half_extent
        self.width = int(math.ceil(self.arena_size / self.resolution)) + 1
        self.height = self.width
        self.obstacle_grid = [[False for _ in range(self.height)] for _ in range(self.width)]
        self.obstacles = []
        if obstacles:
            for ox, oy, radius in obstacles:
                self.add_obstacle(ox, oy, radius)

    def in_bounds(self, idx):
        x, y = idx
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, idx):
        x, y = idx
        return self.obstacle_grid[x][y]

    def world_to_index(self, pt):
        x, y = pt
        if not self.world_in_bounds(pt):
            return None
        ix = int(math.floor((x - self.min_coord) / self.resolution))
        iy = int(math.floor((y - self.min_coord) / self.resolution))
        if 0 <= ix < self.width and 0 <= iy < self.height:
            return (ix, iy)
        return None

    def index_to_world(self, idx):
        x, y = idx
        wx = self.min_coord + (x + 0.5) * self.resolution
        wy = self.min_coord + (y + 0.5) * self.resolution
        return (wx, wy)

    def world_in_bounds(self, pt):
        x, y = pt
        return (self.min_coord <= x <= self.max_coord and
                self.min_coord <= y <= self.max_coord)

    def add_obstacle(self, x, y, radius):
        self.obstacles.append((x, y, radius))
        if not self.world_in_bounds((x, y)):
            return
        min_x = max(self.min_coord, x - radius)
        max_x = min(self.max_coord, x + radius)
        min_y = max(self.min_coord, y - radius)
        max_y = min(self.max_coord, y + radius)
        ix_min = max(0, int(math.floor((min_x - self.min_coord) / self.resolution)))
        ix_max = min(self.width - 1, int(math.ceil((max_x - self.min_coord) / self.resolution)))
        iy_min = max(0, int(math.floor((min_y - self.min_coord) / self.resolution)))
        iy_max = min(self.height - 1, int(math.ceil((max_y - self.min_coord) / self.resolution)))
        radius_sq = radius * radius
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                wx, wy = self.index_to_world((ix, iy))
                if (wx - x) ** 2 + (wy - y) ** 2 <= radius_sq:
                    self.obstacle_grid[ix][iy] = True

    def plan(self, start, goal):
        start_idx = self.world_to_index(start)
        goal_idx = self.world_to_index(goal)
        if start_idx is None or goal_idx is None:
            return []
        if self.is_obstacle(goal_idx):
            return []
        # allow start index even if marked as obstacle (robot may start inside inflated radius)
        open_set = []
        heapq.heappush(open_set, (0.0, start_idx))
        came_from = {}
        g_cost = {start_idx: 0.0}
        neighbor_moves = [
            ((1, 0), 1.0), ((-1, 0), 1.0), ((0, 1), 1.0), ((0, -1), 1.0),
            ((1, 1), math.sqrt(2)), ((1, -1), math.sqrt(2)),
            ((-1, 1), math.sqrt(2)), ((-1, -1), math.sqrt(2))
        ]

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_idx:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return [self.index_to_world(idx) for idx in path]
            for move, move_cost in neighbor_moves:
                neighbor = (current[0] + move[0], current[1] + move[1])
                if not self.in_bounds(neighbor):
                    continue
                if neighbor != start_idx and self.is_obstacle(neighbor):
                    continue
                tentative_g = g_cost[current] + move_cost
                if tentative_g < g_cost.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_cost[neighbor] = tentative_g
                    f_cost = tentative_g + heuristic(neighbor, goal_idx)
                    heapq.heappush(open_set, (f_cost, neighbor))
        return []

class Operate:
    def __init__(self, args):
        
        # Initialise robot controller object
        self.botconnect = BotConnect(args.ip)
        self.command = {'wheel_speed':[0, 0], # left wheel speed, right wheel speed
                        'save_slam': False,
                        'run_obj_detector': False,                       
                        'save_obj_detector': False,
                        'save_image': False}
                        
        # TODO: Tune PID parameters here. If you don't want to use PID, set use_pid=0
        self.botconnect.set_pid(use_pid=1, kp=2.5, ki=0.2, kd=0)
        
        # Create a folder "lab_output" that stores the results of the lab
        self.lab_output_dir = 'lab_output/'
        if not os.path.exists(self.lab_output_dir):
            os.makedirs(self.lab_output_dir)
    
        # Initialise SLAM parameters (M2)
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_sensor = ArucoSensor(self.ekf.robot, marker_length=0.06) # size of the ARUCO markers (6cm)
        
        # Initialise detector (M3)
        if args.ckpt == "":
            self.obj_detector = None
            self.cv_vis = cv2.imread('ui/8bit/detector_splash.png')
        else:
            self.obj_detector = ObjectDetector(args.ckpt, use_gpu=False)
            self.cv_vis = np.ones((480,640,3))* 100
        
        # Create a folder to save raw camera images after pressing "i" (M3)
        self.raw_img_dir = 'raw_images/'
        if not os.path.exists(self.raw_img_dir):
            os.makedirs(self.raw_img_dir)
        else:
            # Delete the folder and create an empty one, i.e. every operate.py is run, this folder will be empty.
            shutil.rmtree(self.raw_img_dir)
            os.makedirs(self.raw_img_dir)

        # Autonomous navigation configuration
        self.true_map_path = args.true_map
        self.true_map = self.load_true_map(self.true_map_path)
        if self.true_map.get('markers'):
            try:
                self.ekf.load_map(self.true_map_path)
            except FileNotFoundError:
                pass
        self.obstacles = self.true_map.get('obstacles', [])
        self.planner = GridAStar(arena_size=2.5, resolution=0.05, obstacles=self.obstacles)
        self.autonomous_goal = None
        self.path_world = []
        self.remaining_path = []
        self.following_path = False
        self.slam_res = (520, 520)
        self.map_surface_rect = None
        self.max_linear_speed = 0.25
        self.max_angular_speed = 1.2
        self.linear_gain = 0.8
        self.angular_gain = 2.0
        self.waypoint_tolerance = 0.05


        # Other auxiliary objects/variables
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.obj_detector_output = None
        self.ekf_on = True
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Performing localization scan...'
        self.count_down = 300 # 5 min timer
        self.start_time = time.time()
        self.control_clock = time.time()
        self.img = np.zeros([480,640,3], dtype=np.uint8)
        self.aruco_img = np.zeros([480,640,3], dtype=np.uint8)
        self.obj_detector_pred = np.zeros([480,640], dtype=np.uint8)
        self.bg = pygame.image.load('ui/gui_mask.jpg')

        # Perform an initial localization spin to detect nearby markers
        self.initial_localization_scan()
        self.notification = 'Localization complete. Click the map to set a goal.'

    def load_true_map(self, path):
        true_map = {'markers': {}, 'fruits': {}, 'obstacles': []}
        if not path:
            return true_map
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return true_map
        for name, entry in data.items():
            x = float(entry.get('x', 0.0))
            y = float(entry.get('y', 0.0))
            if name.startswith('aruco'):
                true_map['markers'][name] = (x, y)
            else:
                true_map['fruits'][name] = (x, y)
            true_map['obstacles'].append((x, y, 0.2))
        return true_map

    def initial_localization_scan(self, duration=6.0, spin_speed=0.15):
        start_time = time.time()
        previous_command = list(self.command['wheel_speed'])
        try:
            self.command['wheel_speed'] = [-spin_speed, spin_speed]
            while time.time() - start_time < duration and not self.quit:
                drive_measurement = self.control()
                time.sleep(0.15)
                self.take_pic()
                self.perform_slam(drive_measurement)
        finally:
            self.command['wheel_speed'] = [0.0, 0.0]
            self.control()
            self.command['wheel_speed'] = previous_command

    # wheel control
    def control(self):
        left_speed, right_speed = self.botconnect.set_velocity(self.command['wheel_speed'])
        dt = time.time() - self.control_clock
        drive_measurement = DriveMeasurement(left_speed, right_speed, dt)
        self.control_clock = time.time()
        return drive_measurement
    
    # camera control
    def take_pic(self):
        self.img = self.botconnect.get_image()
        
    # wheel and camera calibration for SLAM
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

    # SLAM with ARUCO markers       
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
        # save slam map after pressing "s"
        if self.command['save_slam']:
            self.ekf.save_map(fname=os.path.join(self.lab_output_dir, 'slam.txt'))
            self.notification = 'Map is saved'
            self.command['save_slam'] = False
        
        # save obj_detector result with the matching robot pose and detector labels
        if self.command['save_obj_detector']:
            if self.obj_detector_output is not None:
                pred, state, detections, W, H = self.obj_detector_output
                self.pred_fname = self.obj_detector.write_image(
                    pred, state, self.lab_output_dir,
                    detections=detections, im_w=W, im_h=H
                )
                self.notification = f'Prediction is saved to {self.pred_fname}'
            else:
                self.notification = 'No prediction in buffer, save ignored'
            self.command['save_obj_detector'] = False

        
        # save raw images taken by the camera after pressing "i"
        if self.command['save_image']:
            image = self.botconnect.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            f_ = os.path.join(self.raw_img_dir, f'img_{self.image_id}.png')
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # using computer vision to detect objects
    def detect_object(self):
        if self.command['run_obj_detector'] and self.obj_detector is not None:
            # self.img is RGB -> model (OpenCV/YOLO) expects BGR
            img_bgr = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

            # UPDATED: detector returns (mask, annotated_vis_bgr, detections)
            pred_mask, vis_bgr, detections = self.obj_detector.detect_single_image(img_bgr)

            # Bring visualization back to RGB for Pygame display
            self.obj_detector_pred = pred_mask
            self.cv_vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

            # Stash everything needed for saving to pred.txt later
            H, W = img_bgr.shape[:2]
            self.obj_detector_output = (
                self.obj_detector_pred,           # image to save (mask/vis)
                self.ekf.robot.state.tolist(),    # robot pose (JSON-friendly)
                detections,                       # list of {name, bbox_xyxy, conf}
                W,                                # im_w
                H                                 # im_h
            )

            # Nice notification
            n_types = len({d['name'] for d in detections}) if detections else 0
            self.notification = f'{n_types} object type(s) detected'

            # Reset the command latch
            self.command['run_obj_detector'] = False


    def screen_to_world(self, screen_pos):
        if not self.map_surface_rect:
            return None
        local_x = screen_pos[0] - self.map_surface_rect.left
        local_y = screen_pos[1] - self.map_surface_rect.top
        if local_x < 0 or local_y < 0 or local_x >= self.map_surface_rect.width or local_y >= self.map_surface_rect.height:
            return None
        w, h = self.slam_res
        x_canvas = w - 1 - local_y + 0.5
        y_canvas = h - 1 - local_x + 0.5
        m2pixel = 100.0
        rel_x = (w / 2.0 - x_canvas) / m2pixel
        rel_y = (y_canvas - h / 2.0) / m2pixel
        robot_state = self.ekf.robot.state
        world_x = robot_state[0, 0] + rel_x
        world_y = robot_state[1, 0] + rel_y
        world_point = (world_x, world_y)
        if not self.planner.world_in_bounds(world_point):
            return None
        return world_point

    def plan_path_to(self, goal_world):
        start = (self.ekf.robot.state[0, 0], self.ekf.robot.state[1, 0])
        path = self.planner.plan(start, goal_world)
        if len(path) < 2:
            self.following_path = False
            self.command['wheel_speed'] = [0.0, 0.0]
            self.remaining_path = []
            if len(path) == 1:
                self.notification = 'Already at the selected location.'
                self.autonomous_goal = goal_world
                self.path_world = path
            else:
                self.notification = 'Unable to find a collision-free path.'
                self.autonomous_goal = None
                self.path_world = []
            return
        self.autonomous_goal = goal_world
        self.path_world = path
        self.remaining_path = path[1:]
        self.following_path = True
        self.notification = 'Autonomous navigation engaged.'

    def update_autonomy(self):
        if not self.following_path:
            return
        if not self.remaining_path:
            self.command['wheel_speed'] = [0.0, 0.0]
            self.following_path = False
            self.notification = 'Destination reached.'
            return
        self._drive_towards_next_waypoint()

    def _drive_towards_next_waypoint(self):
        pose = self.ekf.robot.state
        robot_xy = np.array([pose[0, 0], pose[1, 0]])
        target_xy = np.array(self.remaining_path[0])
        distance = np.linalg.norm(target_xy - robot_xy)
        if distance < self.waypoint_tolerance:
            self.remaining_path.pop(0)
            if not self.remaining_path:
                self.command['wheel_speed'] = [0.0, 0.0]
                self.following_path = False
                self.notification = 'Destination reached.'
            return
        heading = math.atan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0])
        heading_error = self._normalize_angle(heading - pose[2, 0])
        linear_velocity = max(-self.max_linear_speed,
                              min(self.max_linear_speed, self.linear_gain * distance))
        angular_velocity = max(-self.max_angular_speed,
                               min(self.max_angular_speed, self.angular_gain * heading_error))
        if abs(heading_error) > math.pi / 4:
            linear_velocity *= 0.4
        self._set_wheel_speeds_from_twist(linear_velocity, angular_velocity)

    def _set_wheel_speeds_from_twist(self, linear_velocity, angular_velocity):
        baseline = self.ekf.robot.baseline
        scale = self.ekf.robot.scale if self.ekf.robot.scale != 0 else 1.0
        left_m = linear_velocity - angular_velocity * baseline / 2.0
        right_m = linear_velocity + angular_velocity * baseline / 2.0
        left_cmd = max(-1.0, min(1.0, left_m / scale))
        right_cmd = max(-1.0, min(1.0, right_m / scale))
        self.command['wheel_speed'] = [left_cmd, right_cmd]

    @staticmethod
    def _normalize_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def cancel_autonomy(self, reason=None):
        self.following_path = False
        self.remaining_path = []
        self.path_world = []
        self.autonomous_goal = None
        self.command['wheel_speed'] = [0.0, 0.0]
        if reason:
            self.notification = reason

    def set_goal_from_click(self, position):
        self.cancel_autonomy()
        goal_world = self.screen_to_world(position)
        if goal_world is None:
            self.notification = 'Selected point is outside the valid map area.'
            return
        self.plan_path_to(goal_world)


    # paint the GUI
    def draw(self, canvas):
        width, height = 900, 660
        canvas = pygame.display.set_mode((width, height))
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad, h_pad = 40, 20

        # paint SLAM outputs
        res = self.slam_res
        ekf_view = self.ekf.draw_slam_state(
            res=res,
            not_pause=self.ekf_on,
            path=self.path_world if self.path_world else None,
            goal=self.autonomous_goal
        )
        map_position = (2*h_pad+320, v_pad)
        canvas.blit(ekf_view, map_position)
        self.map_surface_rect = pygame.Rect(map_position[0], map_position[1],
                                            ekf_view.get_width(), ekf_view.get_height())
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, position=(h_pad, v_pad))

        # for object detector (M3)
        detector_view = cv2.resize(self.cv_vis, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, position=(h_pad, 240+2*v_pad))

        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector', position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='Robot Cam', position=(h_pad, v_pad))
        notification = TEXT_FONT.render(self.notification, False, text_colour)
        canvas.blit(notification, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption, False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # Keyboard teleoperation
    # For pibot motion, set two numbers for the self.command['wheel_speed']. Eg self.command['wheel_speed'] = [0.6, 0.6]
    # These numbers specify how fast to power the left and right wheels
    # The numbers must be between -1 (full speed backward) and 1 (full speed forward). 0 means stop.
    # Study the code in connect.py for more information
    def handle_event(self, event):
        # drive forward
        if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
            self.cancel_autonomy('Manual override enabled.')
            self.command['wheel_speed'] = [0.3, 0.3]
        # drive backward
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
            self.cancel_autonomy('Manual override enabled.')
            self.command['wheel_speed'] = [-0.3, -0.3]
        # turn left
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
            self.cancel_autonomy('Manual override enabled.')
            self.command['wheel_speed'] = [-0.3, 0.3]
        # turn right
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
            self.cancel_autonomy('Manual override enabled.')
            self.command['wheel_speed'] = [0.3, -0.3]
        # stop via keyboard
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self.cancel_autonomy('Manual override enabled.')
            self.command['wheel_speed'] = [0, 0]
        elif event.type == pygame.KEYUP:
            self.command['wheel_speed'] = [0, 0]
        # run SLAM
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
                if self.ekf_on:
                    self.notification = 'SLAM is running'
                else:
                    self.notification = 'SLAM is paused'
        # save SLAM map
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
            self.command['save_slam'] = True
        # reset SLAM map
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            if self.double_reset_comfirm == 0:
                self.notification = 'Press again to confirm CLEAR MAP'
                self.double_reset_comfirm +=1
            elif self.double_reset_comfirm == 1:
                self.notification = 'SLAM Map is cleared'
                self.double_reset_comfirm = 0
                self.ekf.reset()
        # run object/fruit detector
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            self.command['run_obj_detector'] = True
        # save object detection outputs
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
            self.command['save_obj_detector'] = True
        # capture and save raw image
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
            self.command['save_image'] = True
        # load SLAM map (true map) into EKF
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
            try:
                self.ekf.load_map(os.path.join(self.lab_output_dir, 'slam.txt'))
                self.notification = 'Loaded map from lab_output/slam.txt'
            except Exception as e:
                self.notification = f'Load map failed: {e}'

        # toggle localization-only (freeze map) mode
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
            self.ekf.set_localization_only(not self.ekf.localization_only)
            self.notification = ('Mode: LOCALIZATION-ONLY (map frozen)'
                                if self.ekf.localization_only else
                                'Mode: SLAM (map can update)')
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.set_goal_from_click(event.pos)
        # quit
        elif event.type == pygame.QUIT:
            self.quit = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.quit = True

    def update_keyboard(self):
        for event in pygame.event.get():
            self.handle_event(event)
        if self.quit:
            pygame.quit()
            sys.exit()

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost') # you can hardcode ip here, but it may change from time to time.
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--ckpt", default='cv/model/model.best.pt')
    parser.add_argument("--true_map", type=str, default='truemap.txt')
    args, _ = parser.parse_known_args()
    
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 Lab')
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
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)
    while start:
        operate.update_keyboard()
        operate.update_autonomy()
        operate.take_pic()
        drive_measurement = operate.control()
        operate.perform_slam(drive_measurement)
        operate.save_result()
        operate.detect_object()
        operate.draw(canvas)
        pygame.display.update()
