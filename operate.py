import cv2
import time
import shutil
import argparse
import os, sys
import math
import json
import csv
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
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

# import object pose utilities (M3)
from object_pose_est import CLASS_NAME_MAPPING


MAP_V_PAD = 40
MAP_H_PAD = 20
MAP_VIEW_SIZE = (520, 520)
MAP_VIEW_POS = (2 * MAP_H_PAD + 320, MAP_V_PAD)


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
        self.ekf, self.camera_matrix = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_sensor = ArucoSensor(self.ekf.robot, marker_length=0.06) # size of the ARUCO markers (6cm)

        # Load true map (aruco + prior fruit layout)
        self.true_map = self.load_true_map('truemap.txt')
        try:
            self.ekf.load_map('truemap.txt')
        except Exception:
            # Map loading is optional for development; failures are non-fatal
            pass

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
        

        # Other auxiliary objects/variables      
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.obj_detector_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        self.count_down = 300 # 5 min timer
        self.start_time = time.time()
        self.control_clock = time.time()
        self.img = np.zeros([480,640,3], dtype=np.uint8)
        self.aruco_img = np.zeros([480,640,3], dtype=np.uint8)
        self.obj_detector_pred = np.zeros([480,640], dtype=np.uint8)
        self.bg = pygame.image.load('ui/gui_mask.jpg')
        self.latest_sensor_measurement = []
        self.latest_detections = None

        # Object metadata for pose estimation
        self.object_list, self.object_dimensions = self.load_object_metadata('object_list.csv')
        if not self.object_list:
            fallback_names = sorted({v for v in CLASS_NAME_MAPPING.values()})
            self.object_list = fallback_names
            self.object_dimensions = [[0.1, 0.1, 0.1] for _ in fallback_names]

        # Search list & fruit bookkeeping
        self.search_targets = self.load_search_targets('search_list.txt')

        # Autonomous navigation manager
        self.autonomy = AutonomyManager(self)
        self.autonomy.start_localization_scan()

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

    @staticmethod
    def load_true_map(fname: str) -> Dict[str, Dict[str, float]]:
        try:
            with open(fname, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def load_object_metadata(fname: str) -> Tuple[List[str], List[List[float]]]:
        object_names: List[str] = []
        dimensions: List[List[float]] = []
        try:
            with open(fname, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    name = row.get('object')
                    if not name:
                        continue
                    object_names.append(name)
                    dims: List[float] = []
                    for key in ('length(m)', 'width(m)', 'height(m)'):
                        try:
                            dims.append(float(row.get(key, 0.0)))
                        except (TypeError, ValueError):
                            dims.append(0.0)
                    dimensions.append(dims)
        except Exception:
            pass
        return object_names, dimensions

    @staticmethod
    def load_search_targets(fname: str) -> List[str]:
        targets: List[str] = []
        try:
            with open(fname, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    targets.append(line)
        except Exception:
            pass
        return targets
        
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
        return EKF(robot), camera_matrix

    # SLAM with ARUCO markers       
    def perform_slam(self, drive_measurement):
        sensor_measurement, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)
        self.latest_sensor_measurement = sensor_measurement
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(sensor_measurement)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        else:
            self.ekf.predict(drive_measurement)
            if self.ekf_on:
                self.ekf.add_landmarks(sensor_measurement)
                self.ekf.update(sensor_measurement)
        return sensor_measurement
            
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
        auto_trigger = self.autonomy.should_run_ai_scan() if hasattr(self, 'autonomy') else False
        if (self.command['run_obj_detector'] or auto_trigger) and self.obj_detector is not None:
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
            self.latest_detections = detections
            if not self.autonomy.is_active():
                self.autonomy.on_auto_detections(detections)
            return detections
        else:
            self.latest_detections = None
        return None
    # paint the GUI            
    def draw(self, canvas):
        width, height = 900, 660
        canvas = pygame.display.set_mode((width, height))
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad, h_pad = 40, 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(520, 480+v_pad), not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
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
    def update_keyboard(self):
        manual_allowed = not self.autonomy.should_ignore_manual_drive()
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                if manual_allowed:
                    self.command['wheel_speed'] = [0.3, 0.3]
                 # TODO
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                if manual_allowed:
                    self.command['wheel_speed'] = [-0.3,-0.3]
                 # TODO
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                if manual_allowed:
                    self.command['wheel_speed'] = [-0.3,0.3]
                 # TODO
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                if manual_allowed:
                    self.command['wheel_speed'] = [0.3,-0.3]
                 # TODO
            # stop
            elif event.type == pygame.KEYUP or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
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
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.autonomy.start_search_sequence()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                self.autonomy.autonomous_enabled = False
                self.autonomy.current_goal = None
                self.autonomy.manual_goal_active = False
                self.notification = 'Autonomy cancelled'
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                goal = self.map_click_to_world(event.pos)
                if goal is not None:
                    self.autonomy.set_manual_goal(goal)
        if self.quit:
            pygame.quit()
            sys.exit()


    def update_autonomy(self, sensor_measurement, detections):
        command = self.autonomy.update(sensor_measurement, detections)
        if command is not None:
            self.command['wheel_speed'] = command

    def map_click_to_world(self, mouse_pos: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        x, y = mouse_pos
        origin_x, origin_y = MAP_VIEW_POS
        width, height = MAP_VIEW_SIZE
        if not (origin_x <= x <= origin_x + width and origin_y <= y <= origin_y + height):
            return None
        rel_x = x - origin_x
        rel_y = y - origin_y
        meter_per_pixel = 1.0 / 100.0
        world_x = float(self.ekf.robot.state[0, 0]) + (-(rel_x - width / 2.0) * meter_per_pixel)
        world_y = float(self.ekf.robot.state[1, 0]) + ((rel_y - height / 2.0) * meter_per_pixel)
        return (world_x, world_y)


class AutonomyManager:
    """Handles localization, autonomous goal tracking, and fruit mapping."""

    SCAN_SEGMENTS = 12  # 360 / 30 degrees
    SCAN_PAUSE_DURATION = 0.4
    SCAN_SPEED = 0.15
    SCAN_ANGLE_TOL = math.radians(3.0)
    GOAL_REACH_RADIUS = 0.4
    GOAL_HOLD_TIME = 2.0
    DETECTION_INTERVAL = 1.0
    AVOID_RADIUS = 0.45
    AVOID_GAIN = 0.6

    def __init__(self, operate: 'Operate'):
        self.op = operate
        self.localization_started = False
        self.localization_done = False
        self.scan_phase = 'idle'
        self.scan_step = 0
        self.phase_end_time = 0.0
        self.segment_start_yaw = 0.0
        self.segment_target_delta = (2.0 * math.pi) / self.SCAN_SEGMENTS
        self.pending_pose_estimates: List[Tuple[float, float, float]] = []
        self.current_command: Optional[List[float]] = None
        self.autonomous_enabled = False
        self.current_goal: Optional[Dict[str, object]] = None
        self.goal_queue: deque = deque()
        self.goal_name_queue: deque = deque()
        self.goal_wait_start: Optional[float] = None
        self.last_detection_ts = 0.0
        self.fruit_estimates: Dict[str, Tuple[float, float]] = {}
        self.fruit_samples: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.marker_positions = self._extract_marker_positions(operate.true_map)
        self.manual_goal_active = False

    @staticmethod
    def _extract_marker_positions(true_map: Dict[str, Dict[str, float]]) -> List[Tuple[float, float]]:
        positions: List[Tuple[float, float]] = []
        for key, value in true_map.items():
            if key.startswith('aruco'):
                positions.append((float(value.get('x', 0.0)), float(value.get('y', 0.0))))
        return positions

    def _get_robot_yaw(self) -> float:
        return float(self.op.ekf.robot.state[2, 0])

    @staticmethod
    def _positive_angle_diff(angle: float, reference: float) -> float:
        diff = math.atan2(math.sin(angle - reference), math.cos(angle - reference))
        if diff < 0:
            diff += 2.0 * math.pi
        return diff

    def start_localization_scan(self):
        self.localization_started = True
        self.localization_done = False
        self.scan_phase = 'rotate'
        self.scan_step = 0
        self.phase_end_time = 0.0
        self.segment_start_yaw = self._get_robot_yaw()
        self.pending_pose_estimates.clear()
        self.op.notification = 'Initial localization scan in progress...'
        self.current_command = [-self.SCAN_SPEED, self.SCAN_SPEED]

    def start_search_sequence(self):
        if not self.localization_done:
            return
        self.goal_queue.clear()
        self.goal_name_queue.clear()
        for target in self.op.search_targets:
            self.goal_name_queue.append(target)
        self.autonomous_enabled = True
        self.manual_goal_active = False
        self._advance_goal()
        if self.current_goal:
            self.op.notification = f'Autonomous search: heading to {self.current_goal["name"]}'

    def set_manual_goal(self, goal_pos: Tuple[float, float]):
        self.goal_queue.clear()
        self.goal_name_queue.clear()
        self.current_goal = {'name': 'manual', 'point': goal_pos}
        self.autonomous_enabled = True
        self.manual_goal_active = True
        self.goal_wait_start = None
        self.op.notification = f'Autonomous drive to selected point ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})'

    def should_ignore_manual_drive(self) -> bool:
        return self.localization_started and not self.localization_done or self.autonomous_enabled

    def is_active(self) -> bool:
        return self.localization_started and not self.localization_done or self.autonomous_enabled

    def should_run_ai_scan(self) -> bool:
        if not self.op.obj_detector:
            return False
        now = time.time()
        if (self.autonomous_enabled or (self.localization_started and not self.localization_done)) and now - self.last_detection_ts >= self.DETECTION_INTERVAL:
            self.last_detection_ts = now
            return True
        return False

    def update(self, sensor_measurement, detections: Optional[List[dict]]):
        self.current_command = None
        if not self.localization_done:
            self.current_command = self._update_localization(sensor_measurement)
            if detections:
                self.on_auto_detections(detections)
            return self.current_command

        if self.autonomous_enabled:
            if detections:
                self.on_auto_detections(detections)
            if self.current_goal is None:
                self._advance_goal()
            if self.current_goal is not None:
                self.current_command = self._drive_to_goal()
            else:
                self.op.notification = 'Autonomous sequence complete'
                self.autonomous_enabled = False
        return self.current_command

    def _update_localization(self, sensor_measurement):
        now = time.time()
        if self.scan_phase == 'rotate':
            self.current_command = [-self.SCAN_SPEED, self.SCAN_SPEED]
            delta = self._positive_angle_diff(self._get_robot_yaw(), self.segment_start_yaw)
            if delta >= self.segment_target_delta - self.SCAN_ANGLE_TOL:
                self.scan_phase = 'pause'
                self.phase_end_time = now + self.SCAN_PAUSE_DURATION
                self.current_command = [0.0, 0.0]
        elif self.scan_phase == 'pause':
            self.current_command = [0.0, 0.0]
            if now >= self.phase_end_time:
                self.scan_step += 1
                if self.scan_step >= self.SCAN_SEGMENTS:
                    self._finalize_localization()
                    return [0.0, 0.0]
                self.scan_phase = 'rotate'
                self.segment_start_yaw = self._get_robot_yaw()
                self.current_command = [-self.SCAN_SPEED, self.SCAN_SPEED]
        else:
            self.current_command = [0.0, 0.0]

        if sensor_measurement:
            pose = self._estimate_pose_from_markers(sensor_measurement)
            if pose:
                self.pending_pose_estimates.append(pose)
        return self.current_command

    def _finalize_localization(self):
        if self.pending_pose_estimates:
            xs, ys, thetas = zip(*self.pending_pose_estimates)
            x = sum(xs) / len(xs)
            y = sum(ys) / len(ys)
            mean_sin = sum(math.sin(t) for t in thetas) / len(thetas)
            mean_cos = sum(math.cos(t) for t in thetas) / len(thetas)
            theta = math.atan2(mean_sin, mean_cos)
            self.op.ekf.robot.state = np.array([[x], [y], [theta]])
            self.op.ekf_on = True
        self.localization_done = True
        self.localization_started = False
        self.current_command = [0.0, 0.0]
        self.op.notification = 'Localization complete. Ready for commands.'
        if self.op.search_targets:
            self.start_search_sequence()

    def _estimate_pose_from_markers(self, sensor_measurement) -> Optional[Tuple[float, float, float]]:
        if not sensor_measurement:
            return None
        world_points: List[Tuple[float, float]] = []
        robot_points: List[Tuple[float, float]] = []
        for marker in sensor_measurement:
            tag = getattr(marker, 'tag', None)
            key = f'aruco{tag}_0'
            if tag is None or key not in self.op.true_map:
                continue
            obs = marker.position
            robot_points.append((float(obs[0, 0]), float(obs[1, 0])))
            world = self.op.true_map[key]
            world_points.append((float(world['x']), float(world['y'])))
        if len(world_points) < 2:
            return None
        theta_estimates: List[float] = []
        for i in range(len(world_points)):
            for j in range(i + 1, len(world_points)):
                dwx = world_points[j][0] - world_points[i][0]
                dwy = world_points[j][1] - world_points[i][1]
                drx = robot_points[j][0] - robot_points[i][0]
                dry = robot_points[j][1] - robot_points[i][1]
                angle_world = math.atan2(dwy, dwx)
                angle_robot = math.atan2(dry, drx)
                theta_estimates.append(angle_world - angle_robot)
        if not theta_estimates:
            return None
        mean_sin = sum(math.sin(t) for t in theta_estimates)
        mean_cos = sum(math.cos(t) for t in theta_estimates)
        theta = math.atan2(mean_sin, mean_cos)
        positions: List[Tuple[float, float]] = []
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        for (wx, wy), (rx, ry) in zip(world_points, robot_points):
            robot_x = wx - (cos_t * rx - sin_t * ry)
            robot_y = wy - (sin_t * rx + cos_t * ry)
            positions.append((robot_x, robot_y))
        if not positions:
            return None
        avg_x = sum(p[0] for p in positions) / len(positions)
        avg_y = sum(p[1] for p in positions) / len(positions)
        return avg_x, avg_y, theta

    def _advance_goal(self):
        if self.manual_goal_active:
            return
        while self.goal_queue:
            goal = self.goal_queue.popleft()
            if goal is not None:
                self.current_goal = goal
                self.goal_wait_start = None
                return
        while self.goal_name_queue:
            name = self.goal_name_queue.popleft()
            goal = self._build_goal_from_name(name)
            if goal:
                self.current_goal = goal
                self.goal_wait_start = None
                self.op.notification = f'Heading to {goal["name"]}'
                return
        self.current_goal = None

    def _build_goal_from_name(self, name: str) -> Optional[Dict[str, object]]:
        point = self.fruit_estimates.get(name)
        if point is None:
            key = f'{name}_0'
            if key in self.op.true_map:
                data = self.op.true_map[key]
                point = (float(data.get('x', 0.0)), float(data.get('y', 0.0)))
        if point is None:
            return None
        return {'name': name, 'point': point}

    def on_auto_detections(self, detections: List[dict]):
        if not detections:
            return
        robot_state = self.op.ekf.robot.state.flatten()
        robot_x, robot_y, robot_theta = float(robot_state[0]), float(robot_state[1]), float(robot_state[2])
        focal_length = float(self.op.camera_matrix[0][0]) if self.op.camera_matrix is not None else 1.0
        for det in detections:
            name = det.get('name')
            if not name:
                continue
            mapped = CLASS_NAME_MAPPING.get(name, name.lower())
            if mapped not in self.op.object_list:
                continue
            idx = self.op.object_list.index(mapped)
            dims = self.op.object_dimensions[idx] if idx < len(self.op.object_dimensions) else [0.1, 0.1, 0.1]
            bbox = det.get('bbox_xywh')
            if bbox is None:
                bbox_xyxy = det.get('bbox_xyxy')
                if bbox_xyxy is None:
                    continue
                x1, y1, x2, y2 = bbox_xyxy
                w = max(1.0, float(x2) - float(x1))
                h = max(1.0, float(y2) - float(y1))
                cx = float(x1) + w / 2.0
                cy = float(y1) + h / 2.0
            else:
                cx, cy, w, h = bbox
                w = max(1.0, float(w))
                h = max(1.0, float(h))
                cx = float(cx)
                cy = float(cy)
            conf = float(det.get('conf', 0.0))
            if conf < 0.4:
                continue
            true_width = dims[1] if len(dims) > 1 else dims[0]
            true_height = dims[2] if len(dims) > 2 else dims[0]
            distances = []
            if true_height > 0:
                distances.append((true_height * focal_length) / h)
            if true_width > 0:
                distances.append((true_width * focal_length) / w)
            if not distances:
                continue
            if len(distances) == 1:
                distance = distances[0]
            else:
                distance = math.sqrt(distances[0] * distances[1])
            distance = max(0.1, min(3.0, distance))
            cx_cam = (cx - self.op.img.shape[1] / 2.0) * distance / focal_length
            robot_frame_x = distance
            robot_frame_y = -cx_cam
            cos_t = math.cos(robot_theta)
            sin_t = math.sin(robot_theta)
            world_x = robot_x + (robot_frame_x * cos_t - robot_frame_y * sin_t)
            world_y = robot_y + (robot_frame_x * sin_t + robot_frame_y * cos_t)
            self.fruit_samples[mapped].append((world_x, world_y))
            if len(self.fruit_samples[mapped]) >= 3:
                avg_x = sum(p[0] for p in self.fruit_samples[mapped]) / len(self.fruit_samples[mapped])
                avg_y = sum(p[1] for p in self.fruit_samples[mapped]) / len(self.fruit_samples[mapped])
                self.fruit_estimates[mapped] = (avg_x, avg_y)

    def _drive_to_goal(self) -> List[float]:
        robot_state = self.op.ekf.robot.state.flatten()
        robot_x, robot_y, robot_theta = float(robot_state[0]), float(robot_state[1]), float(robot_state[2])
        goal_point = self.current_goal['point'] if self.current_goal else (robot_x, robot_y)
        goal_x, goal_y = goal_point
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        distance = math.hypot(dx, dy)
        if distance < self.GOAL_REACH_RADIUS:
            if self.goal_wait_start is None:
                self.goal_wait_start = time.time()
                self.op.command['wheel_speed'] = [0.0, 0.0]
                print(f'[AUTO] Target reached: {self.current_goal["name"]} at ({goal_x:.2f}, {goal_y:.2f})')
                self.op.notification = f'Reached {self.current_goal["name"]}'
            elif time.time() - self.goal_wait_start >= self.GOAL_HOLD_TIME:
                if self.current_goal and self.current_goal['name'] != 'manual':
                    print(f'[AUTO] Confirmed fruit: {self.current_goal["name"]}')
                if self.manual_goal_active:
                    self.autonomous_enabled = False
                    self.current_goal = None
                    self.manual_goal_active = False
                else:
                    self.current_goal = None
                    self._advance_goal()
                self.goal_wait_start = None
            return [0.0, 0.0]

        target_theta = math.atan2(dy, dx)
        heading_error = math.atan2(math.sin(target_theta - robot_theta), math.cos(target_theta - robot_theta))
        # Potential-field obstacle avoidance
        avoid_x, avoid_y = 0.0, 0.0
        for obs in self.marker_positions:
            avoid_vec = self._repulsive_force(robot_x, robot_y, obs)
            avoid_x += avoid_vec[0]
            avoid_y += avoid_vec[1]
        for name, pos in self.fruit_estimates.items():
            if self.current_goal and name == self.current_goal.get('name'):
                continue
            avoid_vec = self._repulsive_force(robot_x, robot_y, pos)
            avoid_x += avoid_vec[0]
            avoid_y += avoid_vec[1]
        combined_x = dx + avoid_x
        combined_y = dy + avoid_y
        target_theta = math.atan2(combined_y, combined_x)
        heading_error = math.atan2(math.sin(target_theta - robot_theta), math.cos(target_theta - robot_theta))
        linear = max(-0.4, min(0.4, 0.4 * distance))
        angular = max(-0.4, min(0.4, 0.8 * heading_error))
        left = linear - angular
        right = linear + angular
        left = max(-0.5, min(0.5, left))
        right = max(-0.5, min(0.5, right))
        return [left, right]

    def _repulsive_force(self, robot_x: float, robot_y: float, obstacle: Tuple[float, float]) -> Tuple[float, float]:
        ox, oy = obstacle
        dx = robot_x - ox
        dy = robot_y - oy
        dist = math.hypot(dx, dy)
        if dist < 1e-3 or dist > self.AVOID_RADIUS:
            return (0.0, 0.0)
        scale = self.AVOID_GAIN * (1.0 / dist - 1.0 / self.AVOID_RADIUS) / (dist ** 2)
        return (scale * dx, scale * dy)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost') # you can hardcode ip here, but it may change from time to time.
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--ckpt", default='cv/model/model.best.pt')
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
        operate.take_pic()
        drive_measurement = operate.control()
        sensor_measurement = operate.perform_slam(drive_measurement)
        detections = operate.detect_object()
        operate.update_autonomy(sensor_measurement, detections)
        operate.save_result()
        operate.draw(canvas)
        pygame.display.update()
