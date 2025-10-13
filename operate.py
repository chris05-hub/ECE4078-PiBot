import cv2 
import time
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
import threading
# import CV components (M3)
sys.path.insert(0,"{}/cv/".format(os.getcwd()))
from cv.detector import ObjectDetector
import csv
import math
import json

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
        self.botconnect.set_pid(use_pid=1, linear_kp=2.5, linear_ki=0.02, linear_kd=0.0, rotation_kp = 0.3345, rotation_ki = 0.008, rotation_kd = 0.05)
        
        # Create a folder "lab_output" that stores the results of the lab
        self.lab_output_dir = 'lab_output/'
        if not os.path.exists(self.lab_output_dir):
            os.makedirs(self.lab_output_dir)

        # Initialise SLAM parameters (M2)
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_sensor = ArucoSensor(self.ekf.robot, marker_length=0.07) # size of the ARUCO markers (6cm)
        
        # ADD THESE LINES - Real-time fruit pose estimation
        self.estimated_fruits = {}  # Dictionary: {fruit_name: (x, y)}
        self.true_fruits = {}  # Ground truth from loaded map (optional)
        
        # Define fruit colors for visualization
        self.fruit_colors = {
            'redapple': (255, 0, 0),
            'greenapple': (0, 200, 0),
            'orange': (255, 165, 0),
            'mango': (255, 200, 0),
            'capsicum': (0, 255, 0),
            'yellowlemon': (255, 255, 0),
            'greenlemon': (144, 238, 144)
        }
        with open('object_list.csv', 'r') as file:
            reader = csv.DictReader(file)
            data = list(reader)

        self.object_list = [row['object'] for row in data]
        self.object_dimensions = [[float(row['length(m)']), float(row['width(m)']), float(row['height(m)'])] for row in data]
        self.fruit_lock = threading.Lock()
        # ADD THESE LINES - Arena boundary settings
        self.show_boundaries = True  # Set to False to disable boundary display
        self.arena_bounds = {
            'min_x': -1.25,  # -1.25m (left boundary)
            'max_x': 1.25,   # 1.25m (right boundary)
            'min_y': -1.25,  # -1.25m (bottom boundary)
            'max_y': 1.25    # 1.25m (top boundary)
        }
        # Calculate meters to pixels conversion (assuming center of SLAM view is robot position)
        self.m2pixel = 208  # This should match: pixels_per_meter = 520 / 2.5 = 208
        self.estimated_fruits = {}
        
        # Outlier rejection parameters
        self.max_position_jump = 2  # 30cm - reject observations that jump too far
        self.min_observations = 3     # Need at least 3 observations before trusting position
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

    def estimate_fruit_pose_realtime(self, detections):
        """
        Estimate fruit positions in real-time from detections.
        Similar to object_pose_est.py but runs immediately after detection.
        """
        if not detections:
            return {}
        
        camera_matrix = self.ekf.robot.camera_matrix
        focal_length = camera_matrix[0][0]
        cx = camera_matrix[0][2]
        
        robot_x = self.ekf.robot.state[0, 0]
        robot_y = self.ekf.robot.state[1, 0]
        robot_theta = self.ekf.robot.state[2, 0]
        
        # Map class names
        CLASS_NAME_MAPPING = {
            'Red Apple': 'redapple',
            'Green Apple': 'greenapple',
            'Orange': 'orange',
            'Capsicum': 'capsicum',
            'Lemon': 'yellowlemon',
            'Green Lemon': 'greenlemon',
            'Mango': 'mango'
        }
        
        fruit_positions = {}
        
        for detection in detections:
            detection_name = detection['name']
            
            # Map detection name to standardized name
            if detection_name in CLASS_NAME_MAPPING:
                standardized_name = CLASS_NAME_MAPPING[detection_name]
                
                # Find object index in object_list
                if standardized_name in self.object_list:
                    object_idx = self.object_list.index(standardized_name)
                    
                    # Get bounding box in xywh format
                    bbox_xywh = detection['bbox_xywh']
                    confidence = detection['conf']
                    
                    # Only process high-confidence detections
                    if confidence >= 0.8:
                        box_x_center = bbox_xywh[0]
                        box_y_center = bbox_xywh[1]
                        box_width = bbox_xywh[2]
                        box_height = bbox_xywh[3]
                        
                        # Get object dimensions
                        true_width = self.object_dimensions[object_idx][1]
                        true_height = self.object_dimensions[object_idx][2]
                        
                        # Multi-dimensional distance estimation
                        distances = []
                        
                        if box_height > 20:
                            dist_height = (true_height * focal_length) / (box_height + 1e-6)
                            distances.append(dist_height)
                        
                        if box_width > 20:
                            dist_width = (true_width * focal_length) / (box_width + 1e-6)
                            distances.append(dist_width)
                        
                        # Use geometric mean for stable distance
                        if len(distances) == 2:
                            distance = math.sqrt(distances[0] * distances[1])
                        elif len(distances) == 1:
                            distance = distances[0]
                        else:
                            diagonal_pixels = math.sqrt(box_height**2 + box_width**2)
                            diagonal_real = math.sqrt(true_height**2 + true_width**2)
                            distance = (diagonal_real * focal_length) / (diagonal_pixels + 1e-6)
                        
                        # Apply bounds
                        distance = max(0.1, min(3.0, distance))
                        
                        # Convert to camera frame
                        camera_x = (box_x_center - cx) * distance / focal_length
                        camera_z = distance
                        
                        # Transform to robot frame
                        robot_frame_x = camera_z
                        robot_frame_y = -camera_x
                        
                        # Transform to world frame
                        cos_theta = math.cos(robot_theta)
                        sin_theta = math.sin(robot_theta)
                        
                        object_world_x = robot_x + (robot_frame_x * cos_theta - robot_frame_y * sin_theta)
                        object_world_y = robot_y + (robot_frame_x * sin_theta + robot_frame_y * cos_theta)
                        
                        # Store or update position
                        if standardized_name in fruit_positions:
                            # Average with existing estimate
                            old_x, old_y = fruit_positions[standardized_name]
                            fruit_positions[standardized_name] = (
                                (old_x + object_world_x) / 2,
                                (old_y + object_world_y) / 2
                            )
                        else:
                            fruit_positions[standardized_name] = (object_world_x, object_world_y)
        
        return fruit_positions

    def merge_fruit_estimates(self, new_estimates):
        """
        Merge new fruit estimates with existing estimates.
        Uses robust outlier rejection and weighted averaging.
        Each fruit type has only ONE position estimate.
        """
        with self.fruit_lock:
            for fruit_name, (new_x, new_y) in new_estimates.items():
                if fruit_name in self.estimated_fruits:
                    existing = self.estimated_fruits[fruit_name]
                    old_x = existing['x']
                    old_y = existing['y']
                    old_count = existing['count']
                    
                    # Calculate distance from existing estimate
                    position_jump = math.sqrt((new_x - old_x)**2 + (new_y - old_y)**2)
                    
                    # Outlier rejection: if new observation is too far from current estimate, reject it
                    # BUT: be lenient if we don't have many observations yet
                    if old_count >= self.min_observations and position_jump > self.max_position_jump:
                        # This observation is likely an outlier - skip it
                        print(f"[Outlier] Rejected {fruit_name} observation: jump={position_jump:.3f}m")
                        continue
                    
                    # Weighted average: give more weight to accumulated estimates
                    # Cap the weight to prevent old estimates from becoming too rigid
                    weight_old = min(old_count, 15)  # Cap at 15 observations
                    weight_new = 1.0
                    
                    # If position jump is large but acceptable, reduce weight of new observation
                    if position_jump > self.max_position_jump * 0.5:
                        weight_new = 0.3  # Reduce influence of suspicious observations
                    
                    total_weight = weight_old + weight_new
                    
                    merged_x = (old_x * weight_old + new_x * weight_new) / total_weight
                    merged_y = (old_y * weight_old + new_y * weight_new) / total_weight
                    
                    # Update variance estimate (for monitoring convergence)
                    new_variance = ((old_x - merged_x)**2 + (old_y - merged_y)**2 + 
                                   (new_x - merged_x)**2 + (new_y - merged_y)**2) / 2
                    
                    self.estimated_fruits[fruit_name] = {
                        'x': merged_x,
                        'y': merged_y,
                        'count': old_count + 1,
                        'variance': new_variance
                    }
                else:
                    # First observation of this fruit type
                    self.estimated_fruits[fruit_name] = {
                        'x': new_x,
                        'y': new_y,
                        'count': 1,
                        'variance': 0.0
                    }

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
            if detections and self.ekf_on:
            # Run pose estimation in a separate thread to avoid blocking
                def estimate_and_merge():
                    fruit_poses = self.estimate_fruit_pose_realtime(detections)
                    if fruit_poses:
                        self.merge_fruit_estimates(fruit_poses)
                
                # Start thread
                pose_thread = threading.Thread(target=estimate_and_merge, daemon=True)
                pose_thread.start()
            # Nice notification
            n_types = len({d['name'] for d in detections}) if detections else 0
            self.notification = f'{n_types} object type(s) detected'

            # Reset the command latch
            self.command['run_obj_detector'] = False


    # paint the GUI            
    def draw(self, canvas):
        width, height = 1000, 760
        canvas = pygame.display.set_mode((width, height))
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad, h_pad = 60, 40

        # Calculate the correct m2pixel for 2.5m arena
        slam_res = (520,520)  # (520, 520)
        self.m2pixel = min(slam_res) / 4  # 520 / 2.5 = 208
        
        with self.fruit_lock:
                        current_fruits = {
                        name: (data['x'], data['y']) 
                        for name, data in self.estimated_fruits.items()
            }

        # paint SLAM outputs with the correct scale
        ekf_view = self.ekf.draw_slam_state(
            res=slam_res, 
            not_pause=self.ekf_on,
            m2pixel=self.m2pixel,
            true_fruits=current_fruits,  # Pass estimated fruits
            fruit_colors=self.fruit_colors
        )
        
        # Add grid lines and boundary to ekf_view
        if self.show_boundaries and hasattr(self, 'arena_bounds'):
            ekf_view = self.draw_grid_on_map(
                ekf_view, 
                map_size_m=4, 
                res=slam_res,
                grid_spacing_m=0.5,
                arena_bounds=self.arena_bounds,
                robot_state=self.ekf.robot.state,
                m2pixel=self.m2pixel
            )
        else:
            # Just draw grid without dynamic boundary
            ekf_view = self.draw_grid_on_map(ekf_view, map_size_m=2.5, res=slam_res)
        
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

    def draw_grid_on_map(self, surface, map_size_m=2.5, res=(520, 520), grid_spacing_m=0.5, 
                     arena_bounds=None, robot_state=None, m2pixel=None):
        """
        Draw grid lines and boundary on the SLAM map surface with real-world scaling
        
        Args:
            surface: pygame surface to draw on
            map_size_m: real world map size in meters (default 2.5m x 2.5m)
            res: resolution of the surface (width, height)
            grid_spacing_m: spacing between grid lines in meters (default 0.5m)
            arena_bounds: dict with 'min_x', 'max_x', 'min_y', 'max_y' in meters
            robot_state: current robot state [x, y] for dynamic boundary positioning
            m2pixel: meters to pixel conversion factor
        """
        width, height = res
        
        # Calculate pixels per meter
        pixels_per_meter = min(width, height) / map_size_m
        
        # Calculate the actual map area in pixels (square area)
        map_pixels = int(map_size_m * pixels_per_meter)
        
        # Draw grid lines
        grid_spacing_px = grid_spacing_m * pixels_per_meter
        
        # Draw vertical grid lines
        x = 0
        grid_count = 0
        while x <= map_pixels:
            color = (100, 100, 100) if grid_count % 2 == 0 else (80, 80, 80)
            pygame.draw.line(surface, color, (int(x), 0), (int(x), map_pixels), 1)
            x += grid_spacing_px
            grid_count += 1
        
        # Draw horizontal grid lines
        y = 0
        grid_count = 0
        while y <= map_pixels:
            color = (100, 100, 100) if grid_count % 2 == 0 else (80, 80, 80)
            pygame.draw.line(surface, color, (0, int(y)), (map_pixels, int(y)), 1)
            y += grid_spacing_px
            grid_count += 1
        
        # Draw dynamic arena boundary if parameters provided
        if arena_bounds is not None and robot_state is not None and m2pixel is not None:
            rx, ry = robot_state[0, 0], robot_state[1, 0]
            center_x = width // 2
            center_y = height // 2
            
            # Calculate boundary corners in screen coordinates
            min_x_screen = center_x - int((arena_bounds['min_x'] - rx) * m2pixel)
            max_x_screen = center_x - int((arena_bounds['max_x'] - rx) * m2pixel)
            min_y_screen = center_y + int((arena_bounds['min_y'] - ry) * m2pixel)
            max_y_screen = center_y + int((arena_bounds['max_y'] - ry) * m2pixel)
            
            # Clip to surface bounds
            min_x_screen = max(min_x_screen, 0)
            max_x_screen = min(max_x_screen, width)
            min_y_screen = max(min_y_screen, 0)
            max_y_screen = min(max_y_screen, height)
            
            # Draw boundary rectangle
            boundary_points = [
                (min_x_screen, min_y_screen),
                (max_x_screen, min_y_screen),
                (max_x_screen, max_y_screen),
                (min_x_screen, max_y_screen)
            ]
            
            # Check if robot is out of bounds
            robot_out_of_bounds = (rx < arena_bounds['min_x'] or 
                                rx > arena_bounds['max_x'] or
                                ry < arena_bounds['min_y'] or 
                                ry > arena_bounds['max_y'])
            
            # Choose color based on robot position
            if robot_out_of_bounds:
                boundary_color = (255, 50, 50)  # Red if out of bounds
                line_width = 4
            else:
                boundary_color = (100, 255, 100)  # Green if within bounds
                line_width = 3
            
            # Draw the boundary rectangle
            pygame.draw.lines(surface, boundary_color, True, boundary_points, line_width)
            
            # Draw semi-transparent fill to show safe area
            if not robot_out_of_bounds:
                s = pygame.Surface(res, pygame.SRCALPHA)
                fill_points = [
                    (min_x_screen, min_y_screen),
                    (max_x_screen, min_y_screen),
                    (max_x_screen, max_y_screen),
                    (min_x_screen, max_y_screen)
                ]
                pygame.draw.polygon(s, (100, 255, 100, 20), fill_points)
                surface.blit(s, (0, 0))
        
        # Draw scale markers every meter
        font = pygame.font.SysFont('Arial', 12)
        for i in range(int(map_size_m) + 1):
            pos_px = i * pixels_per_meter
            label = font.render(f'{i}m', True, (200, 200, 200))
            if pos_px < map_pixels - 30:
                surface.blit(label, (int(pos_px) + 2, 2))
            if pos_px < map_pixels - 20:
                surface.blit(label, (2, int(pos_px) + 2))
        
        return surface

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
                self.command['wheel_speed'] = [0.3, 0.3]
                 # TODO
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['wheel_speed'] = [-0.3,-0.3]
                 # TODO
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['wheel_speed'] = [-0.3,0.3]
                 # TODO
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
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
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_o:
                with self.fruit_lock:
                    if self.estimated_fruits:
                        # Convert to the standard format (one fruit per type)
                        fruit_dict = {}
                        
                        for fruit_name, data in self.estimated_fruits.items():
                            # Only save if we have enough observations for confidence
                            if data['count'] >= 2:  # At least 2 observations
                                key = f"{fruit_name}_0"
                                fruit_dict[key] = {
                                    "x": float(data['x']),
                                    "y": float(data['y'])
                                }
                        
                        # Save to objects.txt
                        import json
                        with open(os.path.join(self.lab_output_dir, 'objects.txt'), 'w') as f:
                            json.dump(fruit_dict, f, indent=4)
                        
                        self.notification = f'Saved {len(fruit_dict)} fruit positions to objects.txt'
                    else:
                        self.notification = 'No fruits to save'

            # UPDATE clear to work with dictionary structure
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                with self.fruit_lock:
                    self.estimated_fruits.clear()
                self.notification = 'Cleared fruit estimates'
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
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
        operate.perform_slam(drive_measurement)
        operate.save_result()
        operate.detect_object()
        operate.draw(canvas)
        pygame.display.update()