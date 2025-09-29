import cv2
import time
import shutil
import argparse
import csv
import os, sys
from collections import deque
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

# object pose estimation utilities
import object_pose_est as obj_pose_est

# object pose estimation utilities
import object_pose_est as obj_pose_est


LOCALIZATION_SETTINGS = {
    # Speed (in wheel command units) used while spinning during the
    # localization routine. Adjust here to tune without command-line args.
    'wheel_speed': 0.25,
    # Number of full rotations to perform (each split into 12 thirty-degree
    # segments) when localizing at startup.
    'rotations': 1.0,
}


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

        # load prior map for localization if available
        self.true_map_path = os.path.join(os.getcwd(), 'truemap.txt')
        self.notification = 'Initialising...'
        try:
            if os.path.exists(self.true_map_path):
                self.ekf.load_map(self.true_map_path)
                self.notification = 'Loaded reference map. Performing localization scan.'
            else:
                self.notification = 'Reference map not found. Performing exploratory localization.'
        except Exception as exc:
            self.notification = f'Failed to load reference map: {exc}'

        # freeze landmarks while we localize against the provided map
        self.ekf.set_localization_only(True)

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
        self.ekf_on = True
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.count_down = 300 # 5 min timer
        self.start_time = time.time()
        self.control_clock = time.time()
        self.img = np.zeros([480,640,3], dtype=np.uint8)
        self.aruco_img = np.zeros([480,640,3], dtype=np.uint8)
        self.obj_detector_pred = np.zeros([480,640], dtype=np.uint8)
        self.bg = pygame.image.load('ui/gui_mask.jpg')

        # prepare object pose estimation assets
        self.object_meta = self._load_object_metadata('object_list.csv')
        self.fruit_observations = {name: [] for name in self.object_meta.keys()}
        obj_pose_est.object_list = list(self.object_meta.keys())
        obj_pose_est.object_dimensions = [self.object_meta[name]['dimensions'] for name in self.object_meta.keys()]
        self.ekf.set_object_colour_map({name: meta['colour'] for name, meta in self.object_meta.items()})
        self.ekf.set_object_display_names({name: meta['display'] for name, meta in self.object_meta.items()})

        # localisation scan setup
        self.localization_speed = LOCALIZATION_SETTINGS['wheel_speed']
        rotations = max(LOCALIZATION_SETTINGS['rotations'], 0.0)
        steps_per_rotation = 12
        total_steps = int(round(steps_per_rotation * rotations))
        if rotations > 0 and total_steps == 0:
            total_steps = 1
        self.localization_sequence = self._create_localization_sequence(steps=total_steps)
        self.localization_action = None
        self.localization_action_end = 0.0
        self.localization_complete = False
        self.auto_detection_pending = False


    # wheel control
    def control(self):
        left_speed, right_speed = self.botconnect.set_velocity(self.command['wheel_speed'])
        dt = time.time() - self.control_clock
        drive_measurement = DriveMeasurement(left_speed, right_speed, dt)
        self.control_clock = time.time()
        return drive_measurement

    def _load_object_metadata(self, csv_path):
        meta = {}
        csv_abspath = os.path.join(os.getcwd(), csv_path)
        if not os.path.exists(csv_abspath):
            return meta
        inverse_class_map = {v: k for k, v in obj_pose_est.CLASS_NAME_MAPPING.items()}
        with open(csv_abspath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['object'].strip()
                colour = tuple(int(c.strip()) for c in row['rgb display'].strip('()').split(','))
                length = float(row['length(m)'])
                width = float(row['width(m)'])
                height = float(row['height(m)'])
                display_name = inverse_class_map.get(name, name.replace('_', ' ').title())
                meta[name] = {
                    'colour': colour,
                    'dimensions': (length, width, height),
                    'display': display_name
                }
        return meta

    def _create_localization_sequence(self, steps, rotate_time=0.7, pause_time=0.5):
        sequence = deque()
        if steps <= 0:
            sequence.append(('final_pause', 0.5))
            return sequence
        for _ in range(steps):
            sequence.append(('rotate', rotate_time))
            sequence.append(('pause', pause_time))
        sequence.append(('final_pause', 0.5))
        return sequence

    def update_localization_sequence(self):
        if self.localization_complete or not self.localization_sequence:
            return
        now = time.time()
        if self.localization_action is None:
            action, duration = self.localization_sequence.popleft()
            self.localization_action = action
            self.localization_action_end = now + duration
            if action == 'rotate':
                self.command['wheel_speed'] = [self.localization_speed, -self.localization_speed]
            else:
                self.command['wheel_speed'] = [0, 0]
                self.auto_detection_pending = True
        elif now >= self.localization_action_end:
            self.localization_action = None
            if not self.localization_sequence:
                self.localization_complete = True
                self.command['wheel_speed'] = [0, 0]
                self.ekf.set_localization_only(False)
                self.notification = 'Localization complete. Ready for teleoperation.'

    def consume_auto_detection_request(self):
        if self.auto_detection_pending:
            self.auto_detection_pending = False
            return True
        return False

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
    def detect_object(self, force=False, notify=True):
        if self.obj_detector is None:
            if not force:
                return []
            return []

        should_run = force or self.command['run_obj_detector']
        if not should_run:
            return []

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

        if notify:
            n_types = len({d['name'] for d in detections}) if detections else 0
            self.notification = f'{n_types} object type(s) detected'

        # Reset the command latch regardless of trigger source
        self.command['run_obj_detector'] = False
        return detections

    def update_object_estimates(self, detections):
        if not detections:
            return

        detection_payload = {'detections': detections}
        robot_pose = self.ekf.robot.state.flatten().tolist()
        completed = obj_pose_est.get_image_info(detection_payload, robot_pose)
        if not completed:
            return

        camera_matrix = self.ekf.robot.camera_matrix
        object_pose = obj_pose_est.estimate_pose(camera_matrix, completed)

        for name, pose in object_pose.items():
            if name not in self.fruit_observations:
                self.fruit_observations[name] = []
            obs = self.fruit_observations[name]
            obs.append(np.array([pose['x'], pose['y']], dtype=float))
            if len(obs) > 20:
                obs.pop(0)
            avg = np.mean(np.stack(obs, axis=0), axis=0)
            self.ekf.update_object_estimate(name, avg)


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
        operate.update_localization_sequence()
        drive_measurement = operate.control()
        operate.perform_slam(drive_measurement)
        operate.save_result()
        force_detection = operate.consume_auto_detection_request()
        detections = operate.detect_object(force=force_detection, notify=not force_detection)
        operate.update_object_estimates(detections)
        operate.draw(canvas)
        pygame.display.update()