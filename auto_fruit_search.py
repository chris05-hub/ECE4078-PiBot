# M4 - Autonomous object searching
import json
import ast
import math
import argparse
import heapq
from collections import deque

import numpy as np
import pygame

from operate import Operate
import operate as operate_module


def wrap_angle(angle):
    """Wrap angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def load_true_map(fname):
    """Load the ground-truth map file into a dictionary."""
    with open(fname, 'r') as f:
        try:
            return json.load(f)
        except ValueError:
            f.seek(0)
            return ast.literal_eval(f.readline())


def read_true_map(fname):
    """
    Read the ground truth map and output the pose of the ArUco markers and objects to search.

    Returns
    -------
    object_list : list[str]
        List of object classes without unique suffix.
    object_true_pos : np.ndarray
        Array of shape (N, 2) containing object positions in metres.
    aruco_true_pos : np.ndarray
        Array of shape (10, 2) containing ArUco marker positions (aruco1_0 -> aruco10_0).
    gt_dict : dict
        Raw dictionary describing the true map.
    """
    gt_dict = load_true_map(fname)
    object_list = []
    object_true_pos = []
    aruco_true_pos = np.empty([10, 2])

    for key, value in gt_dict.items():
        x = np.round(value['x'], 1)
        y = np.round(value['y'], 1)

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

    return object_list, object_true_pos, aruco_true_pos, gt_dict


def read_search_list():
    """
    Read the search order of the objects.
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        objects = fd.readlines()

        for obj in objects:
            search_list.append(obj.strip())

    return search_list


def print_object_pos(search_list, object_list, object_true_pos):
    """Print the objects' positions in the search order."""
    print("Search order:")
    n_object = 1
    for obj in search_list:
        for i in range(len(object_list)):
            if obj == object_list[i]:
                print(
                    '{}) {} at [{}, {}]'.format(
                        n_object,
                        obj,
                        np.round(object_true_pos[i][0], 1),
                        np.round(object_true_pos[i][1], 1)
                    )
                )
        n_object += 1


class OccupancyGrid:
    """Simple occupancy grid based on the provided true map."""

    def __init__(self, map_dict, bounds=(-1.6, 1.6, -1.6, 1.6), resolution=0.05,
                 default_obstacle_radius=0.12):
        self.resolution = float(resolution)
        self.x_min, self.x_max, self.y_min, self.y_max = map(float, bounds)
        self.width = int(math.ceil((self.x_max - self.x_min) / self.resolution))
        self.height = int(math.ceil((self.y_max - self.y_min) / self.resolution))
        self.grid = np.zeros((self.height, self.width), dtype=bool)
        self.default_obstacle_radius = float(default_obstacle_radius)
        self.obstacles = []  # list of (x, y, radius)

        self._moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
        ]

        # Populate grid from map entries marked as obstacles
        for key, value in map_dict.items():
            lname = key.lower()
            if lname.startswith('obstacle') or 'wall' in lname:
                radius = float(value.get('radius', self.default_obstacle_radius))
                self.add_obstacle(value['x'], value['y'], radius)

    def in_bounds(self, ix, iy):
        return 0 <= ix < self.width and 0 <= iy < self.height

    def world_to_index(self, xy, clamp=True):
        ix = int(math.floor((xy[0] - self.x_min) / self.resolution))
        iy = int(math.floor((xy[1] - self.y_min) / self.resolution))
        if clamp:
            ix = min(max(ix, 0), self.width - 1)
            iy = min(max(iy, 0), self.height - 1)
        return ix, iy

    def index_to_world(self, ix, iy):
        x = self.x_min + (ix + 0.5) * self.resolution
        y = self.y_min + (iy + 0.5) * self.resolution
        return x, y

    def add_obstacle(self, x, y, radius):
        self.obstacles.append((x, y, radius))
        ix, iy = self.world_to_index((x, y))
        cells = int(math.ceil(radius / self.resolution))
        for dx in range(-cells, cells + 1):
            for dy in range(-cells, cells + 1):
                if dx * dx + dy * dy > cells * cells:
                    continue
                cx, cy = ix + dx, iy + dy
                if self.in_bounds(cx, cy):
                    self.grid[cy, cx] = True

    def _nearest_free(self, ix, iy):
        visited = set()
        queue = deque()
        queue.append((ix, iy))
        visited.add((ix, iy))

        while queue:
            cx, cy = queue.popleft()
            if self.in_bounds(cx, cy) and not self.grid[cy, cx]:
                return cx, cy
            for dx, dy, _ in self._moves:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
        return None

    def ensure_free(self, ix, iy):
        if not self.in_bounds(ix, iy):
            ix = min(max(ix, 0), self.width - 1)
            iy = min(max(iy, 0), self.height - 1)
        if not self.grid[iy, ix]:
            return ix, iy
        return self._nearest_free(ix, iy)

    def heuristic(self, node, goal):
        dx = (goal[0] - node[0]) * self.resolution
        dy = (goal[1] - node[1]) * self.resolution
        return math.hypot(dx, dy)

    def astar(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            for dx, dy, cost in self._moves:
                neighbour = (current[0] + dx, current[1] + dy)
                if not self.in_bounds(*neighbour):
                    continue
                if self.grid[neighbour[1], neighbour[0]]:
                    continue

                tentative = g_score[current] + cost * self.resolution
                if tentative < g_score.get(neighbour, float('inf')):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative
                    priority = tentative + self.heuristic(neighbour, goal)
                    heapq.heappush(open_set, (priority, neighbour))

        return None

    @staticmethod
    def _reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def has_line_of_sight(self, start, end):
        sx, sy = start
        ex, ey = end
        dist = math.hypot(ex - sx, ey - sy)
        steps = max(int(dist / (self.resolution * 0.5)), 1)
        for i in range(steps + 1):
            t = i / steps
            x = sx + t * (ex - sx)
            y = sy + t * (ey - sy)
            ix, iy = self.world_to_index((x, y))
            if self.grid[iy, ix]:
                return False
        return True

    def simplify(self, path_world):
        if len(path_world) <= 2:
            return path_world
        simplified = [path_world[0]]
        anchor = path_world[0]
        prev = path_world[1]
        for point in path_world[2:]:
            if not self.has_line_of_sight(anchor, point):
                simplified.append(prev)
                anchor = prev
            prev = point
        simplified.append(path_world[-1])
        return simplified

    def plan_path(self, start_xy, goal_xy):
        start_idx = self.world_to_index(start_xy)
        goal_idx = self.world_to_index(goal_xy)
        start_idx = self.ensure_free(*start_idx)
        goal_idx = self.ensure_free(*goal_idx)

        if start_idx is None or goal_idx is None:
            return None

        path_idx = self.astar(start_idx, goal_idx)
        if not path_idx:
            return None

        path_world = [self.index_to_world(ix, iy) for ix, iy in path_idx]
        return self.simplify(path_world)


class AutoNavigator:
    """Path planning and autonomous navigation helper."""

    def __init__(self, operate, true_map_dict, occupancy_grid,
                 map_rect=pygame.Rect(360, 40, 520, 520), m2pixel=100.0):
        self.operate = operate
        self.true_map_dict = true_map_dict
        self.grid = occupancy_grid
        self.map_rect = map_rect
        self.m2pixel = m2pixel
        self.map_res = (map_rect.width, map_rect.height)

        self.scale = float(np.loadtxt('calibration/param/scale.txt', delimiter=','))
        self.baseline = float(np.loadtxt('calibration/param/baseline.txt', delimiter=','))

        self.max_speed = 0.4  # m/s
        self.max_ang_speed = 1.2  # rad/s
        self.dist_kp = 1.0
        self.heading_kp = 2.5
        self.goal_tolerance = 0.08

        self.path_world = []
        self.current_index = 0
        self.goal_world = None

        # Cache map elements for rendering
        self.map_objects = []
        for key, value in true_map_dict.items():
            lname = key.lower()
            if lname.startswith('aruco'):
                continue
            if lname.startswith('obstacle') or 'wall' in lname:
                continue
            self.map_objects.append((value['x'], value['y'], key.split('_')[0]))

    def process_events(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.map_rect.collidepoint(event.pos):
                    world_xy = self.screen_to_world(event.pos)
                    self.set_goal(world_xy)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                self.cancel_navigation(update_notification=True)

            self.operate.handle_event(event)

        if self.operate.quit:
            pygame.quit()
            sys.exit()

    def set_goal(self, goal_world):
        robot_state = self.operate.ekf.robot.state[:, 0]
        start_xy = (float(robot_state[0]), float(robot_state[1]))

        goal_xy = (
            min(max(goal_world[0], self.grid.x_min), self.grid.x_max),
            min(max(goal_world[1], self.grid.y_min), self.grid.y_max)
        )

        path = self.grid.plan_path(start_xy, goal_xy)
        if path is None or len(path) < 2:
            self.operate.notification = 'No valid path to the selected point'
            return

        self.path_world = path
        self.goal_world = goal_xy
        self.current_index = 0
        self.operate.notification = (
            f'Navigating to ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})'
        )

    def cancel_navigation(self, update_notification=False):
        if self.path_world:
            self.path_world = []
            self.goal_world = None
            self.current_index = 0
            self.operate.command['wheel_speed'] = [0.0, 0.0]
            if update_notification:
                self.operate.notification = 'Navigation cancelled'

    def screen_to_world(self, pos):
        u = pos[0] - self.map_rect.left
        v = pos[1] - self.map_rect.top
        w, h = self.map_res
        dx = -(u - w / 2.0) / self.m2pixel
        dy = (v - h / 2.0) / self.m2pixel
        robot_state = self.operate.ekf.robot.state[:, 0]
        world_x = float(robot_state[0] + dx)
        world_y = float(robot_state[1] + dy)
        return world_x, world_y

    def world_to_screen(self, point):
        robot_state = self.operate.ekf.robot.state[:, 0]
        dx = point[0] - float(robot_state[0])
        dy = point[1] - float(robot_state[1])
        u = -dx * self.m2pixel + self.map_res[0] / 2.0
        v = dy * self.m2pixel + self.map_res[1] / 2.0
        x = self.map_rect.left + int(u)
        y = self.map_rect.top + int(v)
        if self.map_rect.collidepoint((x, y)):
            return x, y
        return None

    def update_motion(self):
        if not self.path_world or self.current_index >= len(self.path_world):
            return

        robot_state = self.operate.ekf.robot.state[:, 0]
        robot_xy = (float(robot_state[0]), float(robot_state[1]))
        robot_theta = float(robot_state[2])

        target = self.path_world[self.current_index]
        dx = target[0] - robot_xy[0]
        dy = target[1] - robot_xy[1]
        distance = math.hypot(dx, dy)

        if distance < self.goal_tolerance:
            self.current_index += 1
            if self.current_index >= len(self.path_world):
                self.operate.command['wheel_speed'] = [0.0, 0.0]
                self.operate.notification = 'Goal reached'
                self.path_world = []
                self.goal_world = None
            return

        desired_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(desired_heading - robot_theta)

        linear_speed = max(min(self.dist_kp * distance, self.max_speed), -self.max_speed)
        heading_weight = max(0.0, 1.0 - abs(heading_error) / (math.pi / 2.0))
        linear_speed *= heading_weight

        angular_speed = max(min(self.heading_kp * heading_error, self.max_ang_speed), -self.max_ang_speed)

        v_left = linear_speed - 0.5 * angular_speed * self.baseline
        v_right = linear_speed + 0.5 * angular_speed * self.baseline

        left_cmd = float(np.clip(v_left / self.scale, -1.0, 1.0))
        right_cmd = float(np.clip(v_right / self.scale, -1.0, 1.0))
        self.operate.command['wheel_speed'] = [left_cmd, right_cmd]

    def draw_overlay(self, canvas):
        # Draw true-map obstacles
        for x, y, radius in self.grid.obstacles:
            pos = self.world_to_screen((x, y))
            if pos is None:
                continue
            pygame.draw.circle(canvas, (140, 40, 40), pos, int(radius * self.m2pixel), width=2)

        # Draw objects as references
        for x, y, _ in self.map_objects:
            pos = self.world_to_screen((x, y))
            if pos is None:
                continue
            pygame.draw.circle(canvas, (40, 140, 40), pos, 4)

        # Draw planned path
        if self.path_world:
            points = [self.world_to_screen(p) for p in self.path_world]
            points = [p for p in points if p is not None]
            if len(points) >= 2:
                pygame.draw.lines(canvas, (50, 120, 220), False, points, 3)
            for p in points:
                pygame.draw.circle(canvas, (50, 120, 220), p, 4)

        if self.goal_world is not None:
            goal_pixel = self.world_to_screen(self.goal_world)
            if goal_pixel is not None:
                pygame.draw.circle(canvas, (255, 140, 0), goal_pixel, 7, width=2)


def main():
    parser = argparse.ArgumentParser("Autonomous navigation")
    parser.add_argument("--map", type=str, default='truemap.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--ckpt", default='cv/model/model.best.pt')
    args, _ = parser.parse_known_args()

    pygame.font.init()
    operate_module.TITLE_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 35)
    operate_module.TEXT_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 40)

    width, height = 900, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 Autonomous Navigation')
    pygame.display.set_icon(pygame.image.load('ui/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('ui/loading.png')
    pibot_animate = [
        pygame.image.load('ui/8bit/pibot1.png'),
        pygame.image.load('ui/8bit/pibot2.png'),
        pygame.image.load('ui/8bit/pibot3.png'),
        pygame.image.load('ui/8bit/pibot4.png'),
        pygame.image.load('ui/8bit/pibot5.png'),
    ]
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
            canvas.blit(pibot_animate[counter % 10 // 2], (x_, 565))
            pygame.display.update()
            counter += 2

    object_list, object_true_pos, aruco_true_pos, gt_dict = read_true_map(args.map)
    search_list = read_search_list()
    print_object_pos(search_list, object_list, object_true_pos)

    occupancy_grid = OccupancyGrid(gt_dict)

    operate = Operate(args)
    navigator = AutoNavigator(operate, gt_dict, occupancy_grid)

    clock = pygame.time.Clock()
    while True:
        events = pygame.event.get()
        navigator.process_events(events)
        navigator.update_motion()

        operate.take_pic()
        drive_measurement = operate.control()
        operate.perform_slam(drive_measurement)
        operate.save_result()
        operate.detect_object()
        operate.draw(canvas)
        navigator.draw_overlay(canvas)
        pygame.display.update()

        clock.tick(30)


if __name__ == "__main__":
    main()
