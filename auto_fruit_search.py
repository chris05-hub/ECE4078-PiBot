# M4 - Autonomous object searching
import os
import sys
import cv2
import ast
import json
import time
import argparse
import numpy as np
from botconnect import BotConnect # access the robot communication

botconnect = None

# import SLAM components (M2)
# sys.path.insert(0, "{}/slam".format(os.getcwd()))
# from slam.ekf import EKF
# from slam.robot import Robot
# from slam.aruco_detector import ArucoDetector


def read_true_map(fname):
    """
    Read the ground truth map and output the pose of the ArUco markers and objects to search
    @param fname: filename of the map
    @return:
        1) list of objects, e.g. ['redapple', 'greenapple', 'orange']
        2) positions of the objects, [[x1, y1], ..... [xn, yn]]
        3) positions of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())   
        object_list = []
        object_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                object_list.append(key[:-2])
                if len(object_true_pos) == 0:
                    object_true_pos = np.array([[x, y]])
                else:
                    object_true_pos = np.append(object_true_pos, [[x, y]], axis=0)

        return object_list, object_true_pos, aruco_true_pos


def read_search_list():
    """
    Read the search order of the objects
    @return: search order of the objects
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        objects = fd.readlines()

        for obj in objects:
            search_list.append(obj.strip())

    return search_list


def print_object_pos(search_list, object_list, object_true_pos):
    """
    Print out the objects pos in the search order
    @param search_list: search order of the objects
    @param object_list: list of objects
    @param object_true_pos: positions of the objects
    """
    print("Search order:")
    n_object = 1
    for obj in search_list:
        for i in range(len(search_list)):
            if obj == object_list[i]:
                print('{}) {} at [{}, {}]'.format(n_object, obj, np.round(object_true_pos[i][0], 1), np.round(object_true_pos[i][1], 1)))
        n_object += 1

# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # Turn toward the waypoint and then drive straight in that direction
    global botconnect

    if botconnect is None:
        raise RuntimeError('BotConnect is not initialised')

    dx = waypoint[0] - robot_pose[0]
    dy = waypoint[1] - robot_pose[1]
    distance = float(np.hypot(dx, dy))

    if distance < 1e-3:
        botconnect.set_velocity([0.0, 0.0])
        return

    target_heading = float(np.arctan2(dy, dx))
    heading_error = target_heading - robot_pose[2]
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

    # Set motion parameters (in metres per second)
    turn_speed_m = 0.15
    drive_speed_m = 0.25

    # Convert to wheel commands using calibration values
    turn_speed_cmd = float(turn_speed_m / scale)
    drive_speed_cmd = float(drive_speed_m / scale)

    # Rotate on the spot toward the waypoint
    omega = 2.0 * turn_speed_m / baseline
    if abs(heading_error) > 1e-3 and omega > 1e-6:
        turn_time = abs(heading_error) / omega
        direction = np.sign(heading_error) or 1.0
        botconnect.set_velocity([-direction * turn_speed_cmd, direction * turn_speed_cmd])
        time.sleep(turn_time)
        botconnect.set_velocity([0.0, 0.0])
        time.sleep(0.1)

    # Drive straight to the waypoint
    if drive_speed_m > 1e-6:
        drive_time = distance / drive_speed_m
        botconnect.set_velocity([drive_speed_cmd, drive_speed_cmd])
        time.sleep(drive_time)
        botconnect.set_velocity([0.0, 0.0])

    # Update local pose estimate
    robot_pose[0] = waypoint[0]
    robot_pose[1] = waypoint[1]
    robot_pose[2] = target_heading
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # Strongly recommend to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    robot_pose = [0.0,0.0,0.0] # replace with your calculation
    ####################################################

    return robot_pose

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Object searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    args, _ = parser.parse_known_args()

    botconnect = BotConnect(args.ip)

    # read in the true map
    object_list, object_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_object_pos(search_list, object_list, object_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    # The following code is only a skeleton code for semi-auto searching task
    while True:
        # enter the waypoints
        # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input)
        x,y = 0.0, 0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        # estimate the robot's pose
        robot_pose = get_robot_pose()

        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint,robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput.lower() == 'n':
            break

