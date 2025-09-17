import cv2
import math
import json
import pygame
import numpy as np


# This class stores the wheel velocities of the robot, to be used in the EKF.
class DriveMeasurement:
    def __init__(self, left_speed, right_speed, dt, left_cov=1, right_cov=1):
        self.left_speed = left_speed
        self.right_speed = right_speed
        self.dt = dt
        self.left_cov = left_cov
        self.right_cov = right_cov



class EKF:
    # Implementation of an EKF for SLAM
    # The EKF state is composed of the robot position (x, y, theta) and the landmark position (x_lm1, y_lm1, x_lm2, y_lm2, ....)
    # lm stands for landmark, ie the aruco markers.

    def __init__(self, robot):
        # State components
        self.robot = robot
        self.markers = np.zeros((2,0))
        
        # Covariance matrix
        self.P = np.zeros((3,3)) # shape of this matrix changes as more landmarks are discovered
        
        self.taglist = []
        self.init_lm_cov = 1e3
        self.robot_init_state = None
        self.lm_pics = []
        for i in range(1, 11):
            f_ = f'./ui/8bit/lm_{i}.png'
            self.lm_pics.append(pygame.image.load(f_))
        f_ = f'./ui/8bit/lm_unknown.png'
        self.lm_pics.append(pygame.image.load(f_))
        self.pibot_pic = pygame.image.load(f'./ui/8bit/pibot_top.png')
        self.localization_only = False  # freeze map if True
        
    def reset(self):
        self.robot.state = np.zeros((3, 1))
        self.markers = np.zeros((2,0))
        self.P = np.zeros((3,3))
        self.taglist = []
        self.init_lm_cov = 1e3
        self.robot_init_state = None

    def set_localization_only(self, flag: bool):
        """Toggle localization-only mode (freeze map)."""
        self.localization_only = bool(flag)

    def load_map(self, fname="lab_output/slam.txt"):
        """Load saved (true) map into self.markers/taglist from JSON slam.txt."""
        import os, re, json, numpy as np
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Map file not found: {fname}")
        with open(fname, 'r') as f:
            data = json.load(f)

        triples = []
        for k, v in data.items():
            m = re.search(r"aruco(\d+)", k)
            if not m: 
                continue
            tag_id = int(m.group(1))
            x = float(v.get('x', 0.0))
            y = float(v.get('y', 0.0))
            triples.append((tag_id, x, y))
        triples.sort(key=lambda t: t[0])

        if not triples:
            self.markers = np.zeros((2,0))
            self.taglist = []
            self.P = np.zeros((3,3))
            return

        xs = [t[1] for t in triples]
        ys = [t[2] for t in triples]
        self.markers = np.vstack([np.array(xs), np.array(ys)])
        self.taglist = [t[0] for t in triples]

        # Expand/set covariance for landmarks and make them “stiff”
        n_lm = len(self.taglist)
        if self.P.shape[0] != 3 + 2*n_lm:
            P_new = np.zeros((3 + 2*n_lm, 3 + 2*n_lm))
            s = min(self.P.shape[0], 3)
            P_new[:s, :s] = self.P[:s, :s]
            self.P = P_new
        eps = 1e-6
        for i in range(n_lm):
            self.P[3+2*i, 3+2*i] = eps
            self.P[3+2*i+1, 3+2*i+1] = eps


    def number_landmarks(self):
        return int(self.markers.shape[1])

    def get_state_vector(self):
        state = np.concatenate(
            (self.robot.state, np.reshape(self.markers, (-1,1), order='F')), axis=0)
        return state
    
    def set_state_vector(self, state):
        self.robot.state = state[0:3,:]
        self.markers = np.reshape(state[3:,:], (2,-1), order='F')
    
    def save_map(self, fname="slam.txt"):
        if self.number_landmarks() > 0:
            d = {}
            for i, tag in enumerate(self.taglist):
                d["aruco" + str(tag) + "_0"] = {"x": self.markers[0,i], "y":self.markers[1,i]}
        with open(fname, 'w') as map_f:
            json.dump(d, map_f, indent=4)

    def recover_from_pause(self, sensor_measurement):
        if not sensor_measurement:
            return False
        else:
            lm_new = np.zeros((2,0))
            lm_prev = np.zeros((2,0))
            tag = []
            for lm in sensor_measurement:
                if lm.tag in self.taglist:
                    lm_new = np.concatenate((lm_new, lm.position), axis=1)
                    tag.append(int(lm.tag))
                    lm_idx = self.taglist.index(lm.tag)
                    lm_prev = np.concatenate((lm_prev,self.markers[:,lm_idx].reshape(2, 1)), axis=1)
            if int(lm_new.shape[1]) > 2:
                R,t = self.umeyama(lm_new, lm_prev)
                theta = math.atan2(R[1][0], R[0][0])
                self.robot.state[:2]=t[:2]
                self.robot.state[2]=theta
                return True
            else:
                return False
        
    ##########################################
    # EKF functions
    # Tune your SLAM algorithm here
    # ########################################

    def predict(self, drive_measurement):
        # Current state
        x = self.get_state_vector()
        theta = self.robot.state[2, 0]

        # Extract motion params
        v_l = drive_measurement.left_speed
        v_r = drive_measurement.right_speed
        dt = drive_measurement.dt

        # Predict pose
        lin_vel, ang_vel = self.robot.convert_wheel_speeds(v_l, v_r)
        
        if abs(ang_vel) < 1e-6:  # Straight motion (unchanged)
            dx = lin_vel * np.cos(theta) * dt
            dy = lin_vel * np.sin(theta) * dt
            dtheta = 0.0
        else:  # Arc motion (improved rotation prediction)
            # Use a more accurate rotation model
            actual_dtheta = ang_vel * dt * 0.6  # The key adjustment factor
            dx = (lin_vel/ang_vel) * (np.sin(theta + actual_dtheta) - np.sin(theta))
            dy = (lin_vel/ang_vel) * (-np.cos(theta + actual_dtheta) + np.cos(theta))
            dtheta = actual_dtheta
        
        # Update state with angle normalization
        self.robot.state[0, 0] += dx
        self.robot.state[1, 0] += dy
        self.robot.state[2, 0] = (theta + dtheta + np.pi) % (2 * np.pi) - np.pi

        # Covariance update (unchanged)
        F = self.state_transition(drive_measurement)
        Q = self.predict_covariance(drive_measurement) * 1.2
        self.P = F @ self.P @ F.T + Q

    # the update/correct step of EKF
    def update(self, sensor_measurement):
        if not sensor_measurement:
            return

        # Construct measurement index list
        tags = [lm.tag for lm in sensor_measurement]
        idx_list = [self.taglist.index(tag) for tag in tags]

        # Stack measurements and set covariance
        z = np.concatenate([lm.position.reshape(-1,1) for lm in sensor_measurement], axis=0)
        R = np.zeros((2*len(sensor_measurement),2*len(sensor_measurement)))
        for i in range(len(sensor_measurement)):
            R[2*i:2*i+2,2*i:2*i+2] = 1*np.eye(2)

        # Compute own measurements
        z_hat = self.robot.measure(self.markers, idx_list)
        z_hat = z_hat.reshape((-1,1), order="F")
        H = self.robot.derivative_measure(self.markers, idx_list)

        x = self.get_state_vector()

        # TODO: add your codes here to compute the updated x
        # Kalman gain
        # EKF equations
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Robust update with innovation clipping
        innovation = z - z_hat
        max_innovation = 0.5  # Tune this threshold
        innovation = np.clip(innovation, -max_innovation, max_innovation)

        x_new = x + K @ innovation
        if self.localization_only:
            x_freeze = x_new.copy()
            if self.number_landmarks() > 0:
                x_freeze[3:, 0] = x[3:, 0]  # lock landmarks
            self.set_state_vector(x_freeze)
        else:
            self.set_state_vector(x_new)

        # Joseph form for covariance update
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        # TODO ends


    def state_transition(self, drive_measurement):
        n = self.number_landmarks()*2 + 3
        F = np.eye(n)
        F[0:3,0:3] = self.robot.derivative_drive(drive_measurement)
        return F
    
    def predict_covariance(self, drive_measurement):
        n = self.number_landmarks()*2 + 3
        Q = np.zeros((n,n))
        Q[0:3,0:3] = self.robot.covariance_drive(drive_measurement)
        return Q

    def add_landmarks(self, sensor_measurement):
        if self.localization_only:
            return
        if not sensor_measurement:
            return

        th = self.robot.state[2]
        robot_xy = self.robot.state[0:2,:]
        R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])

        # Add new landmarks to the state
        for lm in sensor_measurement:
            if lm.tag in self.taglist:
                continue # ignore known tags
            
            lm_position = lm.position
            lm_state = robot_xy + R_theta @ lm_position

            self.taglist.append(int(lm.tag))
            self.markers = np.concatenate((self.markers, lm_state), axis=1)

            # Create a simple, large covariance to be fixed by the update step
            self.P = np.concatenate((self.P, np.zeros((2, self.P.shape[1]))), axis=0)
            self.P = np.concatenate((self.P, np.zeros((self.P.shape[0], 2))), axis=1)
            self.P[-2,-2] = self.init_lm_cov**2
            self.P[-1,-1] = self.init_lm_cov**2

    @staticmethod
    def umeyama(from_points, to_points):
        assert len(from_points.shape) == 2, \
            "from_points must be a m x n array"
        assert from_points.shape == to_points.shape, \
            "from_points and to_points must have the same shape"
        
        N = from_points.shape[1]
        m = 2
        
        mean_from = from_points.mean(axis = 1).reshape((2,1))
        mean_to = to_points.mean(axis = 1).reshape((2,1))
        delta_from = from_points - mean_from # N x m
        delta_to = to_points - mean_to       # N x m
        cov_matrix = delta_to @ delta_from.T / N
        
        U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
        cov_rank = np.linalg.matrix_rank(cov_matrix)
        S = np.eye(m)
        
        if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
            S[m-1, m-1] = -1
        elif cov_rank < m-1:
            raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
        
        R = U.dot(S).dot(V_t)
        t = mean_to - R.dot(mean_from)
        return R, t

    # Plotting functions
    @ staticmethod
    def to_im_coor(xy, res, m2pixel):
        w, h = res
        x, y = xy
        x_im = int(-x*m2pixel+w/2.0)
        y_im = int(y*m2pixel+h/2.0)
        return (x_im, y_im)

    def draw_slam_state(self, res = (320, 500), not_pause=True):
        # Draw landmarks
        m2pixel = 100
        if not_pause:
            bg_rgb = np.array([213, 213, 213]).reshape(1, 1, 3)
        else:
            bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((res[1], res[0], 3))*bg_rgb.astype(np.uint8)
        # in meters, 
        lms_xy = self.markers[:2, :]
        robot_xy = self.robot.state[:2, 0].reshape((2, 1))
        lms_xy = lms_xy - robot_xy
        robot_xy = robot_xy*0
        robot_theta = self.robot.state[2,0]
        # plot robot
        start_point_uv = self.to_im_coor((0, 0), res, m2pixel)
        
        p_robot = self.P[0:2,0:2]
        axes_len,angle = self.make_ellipse(p_robot)
        canvas = cv2.ellipse(canvas, start_point_uv, (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)), angle, 0, 360, (0, 30, 56), 1)
        # draw landmards
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                # plot covariance
                Plmi = self.P[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
                axes_len, angle = self.make_ellipse(Plmi)
                canvas = cv2.ellipse(canvas, coor_, (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)), angle, 0, 360, (244, 69, 96), 1)

        surface = pygame.surfarray.make_surface(np.rot90(canvas))
        surface = pygame.transform.flip(surface, True, False)
        surface.blit(self.rot_center(self.pibot_pic, robot_theta*57.3), (start_point_uv[0]-15, start_point_uv[1]-15))
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                try:
                    surface.blit(self.lm_pics[self.taglist[i]-1], (coor_[0]-5, coor_[1]-5))
                except IndexError:
                    surface.blit(self.lm_pics[-1], (coor_[0]-5, coor_[1]-5))
        return surface

    @staticmethod
    def rot_center(image, angle):
        """rotate an image while keeping its center and size"""
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image       

    @staticmethod
    def make_ellipse(P):
        e_vals, e_vecs = np.linalg.eig(P)
        idx = e_vals.argsort()[::-1]   
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        alpha = np.sqrt(4.605)
        axes_len = e_vals*2*alpha
        if abs(e_vecs[1, 0]) > 1e-3:
            angle = np.arctan(e_vecs[0, 0]/e_vecs[1, 0])
        else:
            angle = 0
        return (axes_len[0], axes_len[1]), angle