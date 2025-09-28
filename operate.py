import cv2
import time
import math
import json
import csv
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

# object pose estimation helpers
import object_pose_est as obj_pose
from object_pose_est import get_image_info, estimate_pose


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

        # Load object metadata for pose estimation (fruit mapping)
        self.object_list = []
        self.object_dimensions = []
        try:
            with open('object_list.csv', 'r') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                self.object_list = [row['object'] for row in rows]
                self.object_dimensions = [
                    [float(row['length(m)']), float(row['width(m)']), float(row['height(m)'])]
                    for row in rows
                ]
            obj_pose.object_list = self.object_list
            obj_pose.object_dimensions = self.object_dimensions
        except FileNotFoundError:
            self.notification = 'object_list.csv missing - fruit mapping disabled'
            self.object_list = []
            self.object_dimensions = []
        except KeyError:
            self.notification = 'object_list.csv malformed - fruit mapping disabled'
            self.object_list = []
            self.object_dimensions = []

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
        self.map_font = pygame.font.Font('ui/8-BitMadness.ttf', 18)
        self.slam_view_rect = None

        # Autonomous navigation state
        self.autonomous_mode = False
        self.target_point = None
        self.navigation_tolerance = 0.08
        self.heading_tolerance = 0.2
        self.forward_speed = 0.25
        self.turn_speed = 0.18
        self.correction_gain = 0.6
        self.auto_detection_period = 1.0
        self.last_auto_detection = 0.0

        # Localization and mapping caches
        self.true_marker_map, self.true_object_map = self.load_true_map('truemap.txt')
        self.object_pose_estimates = {}
        self.object_pose_counts = {}
        self.localization_complete = False
        self.fruit_colours = {
            'redapple': (220, 20, 60),
            'greenapple': (0, 155, 0),
            'orange': (255, 140, 0),
            'capsicum': (178, 34, 34),
            'yellowlemon': (250, 250, 50),
            'greenlemon': (34, 139, 34),
            'mango': (255, 215, 0)
        }

        # Load known marker map into EKF for localization reference
        if self.true_marker_map:
            try:
                self.ekf.load_map('truemap.txt')
            except FileNotFoundError:
                pass

        # Perform initial localization routine with slow rotational scan
        self.perform_initial_localization()

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
        self.camera_matrix = camera_matrix
        return EKF(robot)

    def load_true_map(self, fname):
        marker_map = {}
        object_map = {}
        if not os.path.exists(fname):
            return marker_map, object_map
        try:
            with open(fname, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return marker_map, object_map

        for key, value in data.items():
            if key.startswith('aruco'):
                try:
                    tag_id = int(key.split('_')[0].replace('aruco', ''))
                except ValueError:
                    continue
                marker_map[tag_id] = np.array([[float(value['x'])], [float(value['y'])]])
            else:
                name = key.replace('_0', '')
                object_map[name] = {'x': float(value['x']), 'y': float(value['y'])}
        return marker_map, object_map

    def perform_initial_localization(self):
        if self.localization_complete or not self.true_marker_map:
            return

        self.notification = 'Initial localization: scanning for markers'
        pygame.event.pump()

        detections = {}
        scan_steps = 12
        rotate_duration = 0.45
        pause_duration = 0.35

        for step in range(scan_steps):
            self.command['wheel_speed'] = [-self.turn_speed * 0.8, self.turn_speed * 0.8]
            end_time = time.time() + rotate_duration
            while time.time() < end_time:
                self.control()
                time.sleep(0.05)
            self.command['wheel_speed'] = [0, 0]
            self.control()
            time.sleep(pause_duration)

            self.take_pic()
            sensor_measurement, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)
            for measurement in sensor_measurement:
                tag = int(measurement.tag)
                if tag not in self.true_marker_map:
                    continue
                detections.setdefault(tag, []).append(measurement.position)

            if self.obj_detector is not None:
                self.detect_object(force=True, autop=True)

            pygame.event.pump()

        success = self.estimate_robot_pose_from_markers(detections)
        if success:
            self.localization_complete = True
            self.ekf_on = True
            self.notification = 'Localization complete'
        else:
            self.notification = 'Localization failed: insufficient markers'

        self.command['wheel_speed'] = [0, 0]
        self.control()

    def estimate_robot_pose_from_markers(self, detections):
        if len(detections) < 2:
            return False

        local_points = []
        world_points = []
        for tag, samples in detections.items():
            if tag not in self.true_marker_map or not samples:
                continue
            average_local = np.mean(np.hstack(samples), axis=1, keepdims=True)
            local_points.append(average_local)
            world_points.append(self.true_marker_map[tag])

        if len(local_points) < 2:
            return False

        local_arr = np.hstack(local_points)
        world_arr = np.hstack(world_points)

        try:
            R, t = self.ekf.umeyama(local_arr, world_arr)
        except Exception:
            return False

        theta = math.atan2(R[1, 0], R[0, 0])
        self.ekf.robot.state[0, 0] = t[0]
        self.ekf.robot.state[1, 0] = t[1]
        self.ekf.robot.state[2, 0] = (theta + np.pi) % (2 * np.pi) - np.pi

        return True

    def maybe_run_auto_detector(self):
        if self.obj_detector is None:
            return
        now = time.time()
        if self.autonomous_mode and (now - self.last_auto_detection >= self.auto_detection_period):
            self.detect_object(force=True, autop=True)
            self.last_auto_detection = now

    def update_object_map(self, detections):
        if not detections or not self.object_list or self.camera_matrix is None:
            return

        robot_pose = self.ekf.robot.state.flatten()
        detection_payload = {'detections': detections}
        completed = get_image_info(detection_payload, robot_pose)
        if not completed:
            return

        estimates = estimate_pose(self.camera_matrix, completed)
        if not estimates:
            return

        for name, pose in estimates.items():
            key = name.lower()
            prev = self.object_pose_estimates.get(key)
            if prev:
                count = self.object_pose_counts.get(key, 1)
                new_count = count + 1
                avg_x = (prev['x'] * count + pose['x']) / new_count
                avg_y = (prev['y'] * count + pose['y']) / new_count
                self.object_pose_estimates[key] = {'x': avg_x, 'y': avg_y}
                self.object_pose_counts[key] = new_count
            else:
                self.object_pose_estimates[key] = {'x': pose['x'], 'y': pose['y']}
                self.object_pose_counts[key] = 1

    def set_autonomous_target(self, world_point):
        if not self.localization_complete:
            self.notification = 'Localization incomplete - cannot navigate yet'
            return
        self.target_point = world_point
        self.autonomous_mode = True
        self.last_auto_detection = 0.0
        self.notification = f'Navigating to ({world_point[0]:.2f}, {world_point[1]:.2f})'

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def update_autonomous_navigation(self):
        if not self.autonomous_mode or self.target_point is None:
            return

        robot_x = self.ekf.robot.state[0, 0]
        robot_y = self.ekf.robot.state[1, 0]
        robot_theta = self.ekf.robot.state[2, 0]

        dx = self.target_point[0] - robot_x
        dy = self.target_point[1] - robot_y
        distance = math.hypot(dx, dy)

        if distance < self.navigation_tolerance:
            self.command['wheel_speed'] = [0, 0]
            self.autonomous_mode = False
            self.target_point = None
            self.notification = 'Arrived at destination'
            return

        desired_heading = math.atan2(dy, dx)
        heading_error = self._wrap_to_pi(desired_heading - robot_theta)

        if abs(heading_error) > self.heading_tolerance:
            turn = self.turn_speed if heading_error > 0 else -self.turn_speed
            self.command['wheel_speed'] = [-turn, turn]
        else:
            correction = self.correction_gain * heading_error
            left = np.clip(self.forward_speed - correction, -0.4, 0.4)
            right = np.clip(self.forward_speed + correction, -0.4, 0.4)
            self.command['wheel_speed'] = [left, right]

    def world_to_surface(self, world_xy, surface):
        if surface is None:
            return None
        size = surface.get_size()
        return self.relative_to_surface((world_xy[0] - self.ekf.robot.state[0, 0],
                                         world_xy[1] - self.ekf.robot.state[1, 0]), size)

    def relative_to_surface(self, rel_xy, size):
        if size is None:
            return None
        w, h = size
        m2pixel = 100.0
        raw_x = -rel_xy[0] * m2pixel + w / 2.0
        raw_y = rel_xy[1] * m2pixel + h / 2.0
        screen_x = int((h - 1) - raw_y)
        screen_y = int((w - 1) - raw_x)
        return (screen_x, screen_y)

    def screen_to_world(self, pos):
        if self.slam_view_rect is None:
            return None
        local_x = pos[0] - self.slam_view_rect.left
        local_y = pos[1] - self.slam_view_rect.top
        if local_x < 0 or local_y < 0 or local_x >= self.slam_view_rect.width or local_y >= self.slam_view_rect.height:
            return None

        w = self.slam_view_rect.width
        h = self.slam_view_rect.height
        raw_x = (w - 1) - local_y
        raw_y = (h - 1) - local_x
        m2pixel = 100.0
        rel_x = -(raw_x - w / 2.0) / m2pixel
        rel_y = (raw_y - h / 2.0) / m2pixel
        world_x = rel_x + self.ekf.robot.state[0, 0]
        world_y = rel_y + self.ekf.robot.state[1, 0]
        return (world_x, world_y)

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

    def update_autonomous_commands(self):
        now = time.time()
        if self.localization_mode:
            self.handle_localization(now)
        elif self.autonomous_mode and self.target_point is not None:
            self.handle_navigation()

    def handle_localization(self, now):
        # ensure SLAM keeps running during localization
        if not self.localization_notification_shown:
            self.notification = 'Performing localization scan...'
            self.localization_notification_shown = True
        elapsed = now - self.localization_timer
        if self.localization_phase == 'rotate':
            self.command['wheel_speed'] = [-self.localization_rotation_speed, self.localization_rotation_speed]
            if elapsed >= self.localization_segment_duration:
                self.localization_phase = 'pause'
                self.localization_timer = now
        else:  # pause
            self.command['wheel_speed'] = [0.0, 0.0]
            if elapsed >= self.localization_pause_duration:
                self.localization_segments_done += 1
                if self.localization_segments_done >= self.localization_segments_target:
                    self.localization_mode = False
                    self.localization_complete = True
                    self.localization_phase = 'pause'
                    self.command['wheel_speed'] = [0.0, 0.0]
                    self.ekf.set_localization_only(False)
                    self.notification = 'Localization complete. Click map to navigate.'
                else:
                    self.localization_phase = 'rotate'
                    self.localization_timer = now

    def handle_navigation(self):
        state = self.ekf.robot.state.reshape(-1)
        target_x, target_y = self.target_point
        dx = target_x - state[0]
        dy = target_y - state[1]
        distance = math.hypot(dx, dy)
        if distance < self.target_tolerance:
            self.command['wheel_speed'] = [0.0, 0.0]
            self.autonomous_mode = False
            self.notification = 'Arrived at destination.'
            return

        desired_heading = math.atan2(dy, dx)
        heading_error = self.wrap_angle(desired_heading - state[2])

        linear = max(min(self.nav_linear_gain * distance, 0.4), -0.4)
        angular = max(min(self.nav_angular_gain * heading_error, 0.6), -0.6)

        left = linear - angular
        right = linear + angular

        max_mag = max(abs(left), abs(right), 1.0)
        left /= max_mag
        right /= max_mag

        self.command['wheel_speed'] = [left, right]

    def schedule_auto_detection(self):
        if self.obj_detector is None:
            return
        if not (self.localization_mode or self.autonomous_mode):
            return
        now = time.time()
        if now - self.last_auto_detection >= self.auto_detection_interval:
            self.command['run_obj_detector'] = True
            self.last_auto_detection = now

    def handle_object_mapping(self, detections):
        if not (self.object_pose_available and detections):
            return
        detection_payload = {'detections': []}
        for det in detections:
            x1, y1, x2, y2 = map(float, det['bbox_xyxy'])
            w = max(x2 - x1, 0.0)
            h = max(y2 - y1, 0.0)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            detection_payload['detections'].append({
                'name': det['name'],
                'bbox_xywh': [cx, cy, w, h],
                'conf': float(det['conf'])
            })

        robot_pose = self.ekf.robot.state.reshape(-1)
        completed = object_pose_est.get_image_info(detection_payload, robot_pose)
        if not completed:
            return
        estimated = object_pose_est.estimate_pose(self.camera_matrix, completed)
        if not estimated:
            return
        key = f'frame_{self.auto_detection_counter}'
        self.object_pose_all_img_dict[key] = estimated
        self.auto_detection_counter += 1
        self.object_estimations = object_pose_est.merge_estimations(self.object_pose_all_img_dict)

    def overlay_objects_on_map(self, surface):
        if not self.object_estimations:
            return
        robot_state = self.ekf.robot.state.reshape(-1)
        res_w, res_h = self.slam_res
        for obj_name, pose in self.object_estimations.items():
            if not pose:
                continue
            if pose.get('x', 0.0) == 0.0 and pose.get('y', 0.0) == 0.0:
                continue
            rel_x = pose['x'] - robot_state[0]
            rel_y = pose['y'] - robot_state[1]
            x_canvas, y_canvas = self.ekf.to_im_coor((rel_x, rel_y), (res_w, res_h), 100)
            x_final = (res_h - 1) - y_canvas
            y_final = (res_w - 1) - x_canvas
            if 0 <= x_final < surface.get_width() and 0 <= y_final < surface.get_height():
                base_name = obj_name.replace('_0', '')
                colour = self.object_colors.get(base_name, (255, 128, 0))
                pygame.draw.circle(surface, colour, (x_final, y_final), 6)
                if LABEL_FONT is not None:
                    label = LABEL_FONT.render(base_name, False, colour)
                    surface.blit(label, (x_final + 6, y_final - 6))

        if self.target_point is not None:
            self.draw_target_marker(surface, self.target_point, robot_state)

    def draw_target_marker(self, surface, target, robot_state=None):
        if robot_state is None:
            robot_state = self.ekf.robot.state.reshape(-1)
        res_w, res_h = self.slam_res
        rel_x = target[0] - robot_state[0]
        rel_y = target[1] - robot_state[1]
        x_canvas, y_canvas = self.ekf.to_im_coor((rel_x, rel_y), (res_w, res_h), 100)
        x_final = (res_h - 1) - y_canvas
        y_final = (res_w - 1) - x_canvas
        if 0 <= x_final < surface.get_width() and 0 <= y_final < surface.get_height():
            pygame.draw.circle(surface, (0, 0, 255), (x_final, y_final), 8, 2)

    @staticmethod
    def wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def map_click_to_world(self, pos):
        map_left = 2*self.gui_h_pad + 320
        map_top = self.gui_v_pad
        local_x = pos[0] - map_left
        local_y = pos[1] - map_top
        if not (0 <= local_x < self.slam_res[0] and 0 <= local_y < self.slam_res[1]):
            return None

        res_w, res_h = self.slam_res
        x_canvas = res_w - 1 - local_y
        y_canvas = (res_h - 1) - local_x

        x_rel = -(x_canvas - res_w / 2.0) / 100.0
        y_rel = (y_canvas - res_h / 2.0) / 100.0

        robot_state = self.ekf.robot.state.reshape(-1)
        world_x = robot_state[0] + x_rel
        world_y = robot_state[1] + y_rel
        return world_x, world_y
            
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
    def detect_object(self, force=False, autop=False):
        if self.obj_detector is None:
            return

        run_detector = self.command['run_obj_detector'] or force
        if not run_detector:
            return

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
        if autop:
            self.notification = f'Auto-detected {n_types} object type(s)'
        else:
            self.notification = f'{n_types} object type(s) detected'

        # Reset the command latch
        self.command['run_obj_detector'] = False
        self.last_auto_detection = time.time()

        # Update mapped fruit positions
        self.update_object_map(detections)


    # paint the GUI            
    def draw(self, canvas):
        width, height = 900, 660
        canvas = pygame.display.set_mode((width, height))
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad, h_pad = 40, 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(520, 480+v_pad), not_pause = self.ekf_on)
        self.annotate_slam_overlay(ekf_view)
        slam_pos = (2*h_pad+320, v_pad)
        self.slam_view_rect = pygame.Rect(slam_pos, ekf_view.get_size())
        canvas.blit(ekf_view, slam_pos)
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

    def annotate_slam_overlay(self, surface):
        if surface is None:
            return

        # Draw detected fruits
        for name, pose in self.object_pose_estimates.items():
            pt = self.world_to_surface((pose['x'], pose['y']), surface)
            if pt is None:
                continue
            colour = self.fruit_colours.get(name, (80, 80, 80))
            pygame.draw.circle(surface, colour, pt, 6)
            label = self.map_font.render(name[:1].upper(), False, colour)
            surface.blit(label, (pt[0] - label.get_width() // 2, pt[1] - label.get_height() // 2))

        # Draw current navigation target
        if self.target_point is not None:
            pt = self.world_to_surface(self.target_point, surface)
            if pt is not None:
                pygame.draw.circle(surface, (40, 40, 220), pt, 9, 2)
                pygame.draw.line(surface, (40, 40, 220), (pt[0] - 6, pt[1]), (pt[0] + 6, pt[1]), 2)
                pygame.draw.line(surface, (40, 40, 220), (pt[0], pt[1] - 6), (pt[0], pt[1] + 6), 2)

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
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.autonomous_mode = False
                self.target_point = None
                self.command['wheel_speed'] = [0.3, 0.3]
                 # TODO
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.autonomous_mode = False
                self.target_point = None
                self.command['wheel_speed'] = [-0.3,-0.3]
                 # TODO
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.autonomous_mode = False
                self.target_point = None
                self.command['wheel_speed'] = [-0.3,0.3]
                 # TODO
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.autonomous_mode = False
                self.target_point = None
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
                    self.localization_mode = False
                    self.localization_complete = False
                    self.autonomous_mode = False
                    self.target_point = None
                    self.object_pose_all_img_dict.clear()
                    self.object_estimations = {}
                    self.auto_detection_counter = 0
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
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                target = self.screen_to_world(event.pos)
                if target is not None:
                    self.set_autonomous_target(target)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                self.autonomous_mode = False
                self.target_point = None
                self.notification = 'Autonomous navigation cancelled'
        if self.quit:
            pygame.quit()
            sys.exit()

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost') # you can hardcode ip here, but it may change from time to time.
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--ckpt", default='cv/model/model.best.pt')
    args, _ = parser.parse_known_args()
    
    pygame.font.init()
    TITLE_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 40)
    LABEL_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 18)
    
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
        operate.update_autonomous_navigation()
        operate.take_pic()
        drive_measurement = operate.control()
        operate.perform_slam(drive_measurement)
        operate.maybe_run_auto_detector()
        operate.save_result()
        operate.detect_object()
        operate.draw(canvas)
        pygame.display.update()
