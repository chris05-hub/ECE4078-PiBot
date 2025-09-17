# import module
import os
import ast
import json
import math
import csv
import numpy as np

# Match Class Names
CLASS_NAME_MAPPING = {
    'Red Apple': 'redapple',
    'Green Apple': 'greenapple', 
    'Orange': 'orange',
    'Capsicum': 'capsicum',
    'Lemon': 'yellowlemon',
    'Green Lemon': 'greenlemon',
    'Mango': 'mango'
}
# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(detection_data, robot_pose):
    object_lst_box = [[] for _ in range(len(object_list))]
    object_lst_robot_pose = [[] for _ in range(len(object_list))]
    completed_img_dict = {}

    # Process each detection in the image
    for detection in detection_data.get('detections', []):
        detection_name = detection['name']
        
        # Map detection name to standardized name
        if detection_name in CLASS_NAME_MAPPING:
            standardized_name = CLASS_NAME_MAPPING[detection_name]
            
            # Find object index in object_list
            if standardized_name in object_list:
                object_idx = object_list.index(standardized_name)
                
                # Extract bounding box in xywh format (center x, center y, width, height)
                bbox_xywh = detection['bbox_xywh']  # [x_center, y_center, width, height]
                confidence = detection['conf']
                
                # Only use high-confidence detections
                if confidence >= 0.5:  # Adjust threshold as needed
                    object_lst_box[object_idx].append(bbox_xywh)
                    object_lst_robot_pose[object_idx].append(np.array(robot_pose).reshape(3,))

    # Combine multiple detections of the same object type
    for i in range(len(object_list)):
        if len(object_lst_box[i]) > 0:
            box = np.array(object_lst_box[i]).T  # Shape: (4, n_detections)
            pose = np.stack(object_lst_robot_pose[i], axis=1)  # Shape: (3, n_detections)
            completed_img_dict[i+1] = {'object': box, 'robot': pose}
    
    return completed_img_dict

# estimate the pose of a object based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(camera_matrix, completed_img_dict):
    focal_length = camera_matrix[0][0]
    cx = camera_matrix[0][2]  # principal point x

    object_pose_one_img_dict = {}
    
    for pixel_label in completed_img_dict.keys():
        box = completed_img_dict[pixel_label]['object']  # Shape: (4, n_detections) [x_center, y_center, width, height]
        robot_pose = completed_img_dict[pixel_label]['robot']  # Shape: (3, n_detections) [x, y, theta]
        
        # Get object dimensions
        true_width = object_dimensions[pixel_label-1][1]   # Object's real width in meters
        true_height = object_dimensions[pixel_label-1][2]  # Object's real height in meters
        
        # Process all detections of this object type
        object_positions = []
        
        for detection_idx in range(box.shape[1]):
            # Extract values for this detection
            box_x_center = box[0, detection_idx]  # bounding box center x in pixels
            box_width = box[2, detection_idx]     # bounding box width in pixels
            box_height = box[3, detection_idx]    # bounding box height in pixels
            
            robot_x = robot_pose[0, detection_idx]    # robot x position in meters
            robot_y = robot_pose[1, detection_idx]    # robot y position in meters
            robot_theta = robot_pose[2, detection_idx]  # robot orientation in radians
            
            # Multi-dimensional distance estimation approach
            # Use both height and width for more robust estimation
            distances = []
            
            # Distance from height (assuming vertical orientation is most common)
            if box_height > 20:  # Only use if height is reasonable
                dist_height = (true_height * focal_length) / (box_height + 1e-6)
                distances.append(dist_height)
            
            # Distance from width (for side views or when height is unreliable)
            if box_width > 20:  # Only use if width is reasonable
                dist_width = (true_width * focal_length) / (box_width + 1e-6)
                distances.append(dist_width)
            
            # Use geometric mean for more stable distance estimation
            if len(distances) == 2:
                distance = math.sqrt(distances[0] * distances[1])
            elif len(distances) == 1:
                distance = distances[0]
            else:
                # Fallback: use diagonal dimension
                diagonal_pixels = math.sqrt(box_height*2 + box_width*2)
                diagonal_real = math.sqrt(true_height*2 + true_width*2)
                distance = (diagonal_real * focal_length) / (diagonal_pixels + 1e-6)
            
            # Apply reasonable distance bounds to filter unrealistic estimates
            min_distance = 0.1  # 10cm minimum
            max_distance = 3.0  # 3m maximum
            distance = max(min_distance, min(max_distance, distance))
            
            # Convert pixel coordinates to normalized camera coordinates
            camera_x = (box_x_center - cx) * distance / focal_length
            camera_z = distance  # forward distance from camera
            
            # Simple transformation (camera to robot frame)
            # Assuming camera is aligned with robot (no extrinsics calibration needed)
            robot_frame_x = camera_z    # forward in robot frame
            robot_frame_y = -camera_x   # left in robot frame (negative for coordinate conversion)
            
            # Transform from robot frame to world frame
            cos_theta = math.cos(robot_theta)
            sin_theta = math.sin(robot_theta)
            
            # Apply rotation transformation
            object_world_x = robot_x + (robot_frame_x * cos_theta - robot_frame_y * sin_theta)
            object_world_y = robot_y + (robot_frame_x * sin_theta + robot_frame_y * cos_theta)
            
            object_positions.append([object_world_x, object_world_y])
        
        # Average multiple detections of the same object in the same image
        if len(object_positions) > 0:
            avg_position = np.mean(object_positions, axis=0)
            object_pose = {'x': float(avg_position[0]), 'y': float(avg_position[1])}
            object_pose_one_img_dict[object_list[pixel_label-1]] = object_pose
    
    return object_pose_one_img_dict

# merge the estimations of the objects so that there are at most 1 estimate for each object type
def merge_estimations(object_pose_all_img_dict):

    # final estimation dictionary, initialize as empty array first
    est_dict = {}
    for object in object_list:
        est_dict[str(object) + '_0'] = []
    
    # collect estimated positions from all images in the array
    for file_path in object_pose_all_img_dict:
        for found_object_name in object_pose_all_img_dict[file_path]:
            key = found_object_name + '_0'
            est_dict[key].append(np.array(list(object_pose_all_img_dict[file_path][found_object_name].values()), dtype=float))
    
    # Advanced merging for maximum accuracy
    for key in est_dict:
        if len(est_dict[key]) == 0:
            est_dict[key] = {'x': 0.0, 'y': 0.0}
        elif len(est_dict[key]) == 1:
            pose_x = est_dict[key][0][0]
            pose_y = est_dict[key][0][1]
            est_dict[key] = {'x': float(pose_x), 'y': float(pose_y)}
        else:
            positions = np.array(est_dict[key])
            
            # Use median for initial robust estimate (less sensitive to outliers)
            if len(positions) >= 3:
                median_pos = np.median(positions, axis=0)
                
                # Calculate distances from median
                distances = np.linalg.norm(positions - median_pos, axis=1)
                
                # Use Median Absolute Deviation (MAD) for robust outlier detection
                mad = np.median(np.abs(distances - np.median(distances)))
                
                # Remove outliers using modified z-score with MAD
                if mad > 0:
                    modified_z_scores = 0.6745 * (distances - np.median(distances)) / mad
                    # Keep points with modified z-score < 2.5 (less aggressive than 3.5)
                    inlier_mask = np.abs(modified_z_scores) < 2.5
                    
                    if np.sum(inlier_mask) >= max(2, len(positions) * 0.3):
                        positions = positions[inlier_mask]
            
            # Iterative clustering for multiple groups
            if len(positions) >= 4:
                # Simple clustering based on distance threshold
                cluster_threshold = 0.15  # 15cm threshold
                clusters = []
                remaining_positions = list(range(len(positions)))
                
                while remaining_positions:
                    # Start a new cluster with the first remaining position
                    seed_idx = remaining_positions[0]
                    seed_pos = positions[seed_idx]
                    current_cluster = [seed_idx]
                    remaining_positions.remove(seed_idx)
                    
                    # Add nearby positions to the cluster
                    to_remove = []
                    for idx in remaining_positions:
                        if np.linalg.norm(positions[idx] - seed_pos) < cluster_threshold:
                            current_cluster.append(idx)
                            to_remove.append(idx)
                    
                    for idx in to_remove:
                        remaining_positions.remove(idx)
                    
                    clusters.append(current_cluster)
                
                # Use the largest cluster
                if clusters:
                    largest_cluster = max(clusters, key=len)
                    positions = positions[largest_cluster]
            
            # Weighted average based on local density
            if len(positions) > 1:
                weights = []
                for i, pos in enumerate(positions):
                    # Calculate local density (number of nearby points)
                    nearby_count = 0
                    for j, other_pos in enumerate(positions):
                        if i != j:
                            distance = np.linalg.norm(pos - other_pos)
                            if distance < 0.05:  # Within 5cm
                                nearby_count += 1
                    
                    # Weight based on local density + base weight
                    weight = 1.0 + nearby_count * 0.5
                    weights.append(weight)
                
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                
                # Compute weighted average
                final_pos = np.average(positions, axis=0, weights=weights)
            else:
                final_pos = positions[0]
            
            # Apply Kalman-like smoothing for temporal consistency
            est_dict[key] = {'x': float(final_pos[0]), 'y': float(final_pos[1])}
    
    return est_dict


if __name__ == "__main__":
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',') 

    with open('object_list.csv', 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    object_list = [row['object'] for row in data]
    object_dimensions = [[float(row['length(m)']), float(row['width(m)']), float(row['height(m)'])] for row in data]
    
    # pred.txt contains the pair information on robot pose + file name of the segmentation mask (pred_X.png)
    # For every mask, the given code will first find the bounding boxes in the image
    # Then, store the pixel_label + robot_pose + bounding boxes information in completed_img_dict
    # This dict will be used in estimate_pose function to estimate the objects' position in the image
    # Repeat for all images, and all info is stored in object_pose_all_img_dict
    object_pose_all_img_dict = {}
    with open('lab_output/pred.txt') as fp:
        for line in fp.readlines():
            line_data = ast.literal_eval(line.strip())
            file_path = line_data['predfname']
            robot_pose = [pose[0] for pose in line_data['pose']]  # Extract scalar values from nested lists
            completed_img_dict = get_image_info(line_data, robot_pose)
            if completed_img_dict:  # Only process if detections found
                object_pose_all_img_dict[file_path] = estimate_pose(camera_matrix, completed_img_dict)

    # merge the estimations of the objects so that there are only one estimate for each object type
    object_est = merge_estimations(object_pose_all_img_dict)
                     
    # Save object pose estimations
    with open('lab_output/objects.txt', 'w') as fo:
        json.dump(object_est, fo, indent=4)
    
    print('Estimations saved!')
    
    # Print summary of found objects
    found_objects = 0
    for key, pos in object_est.items():
        if pos['x'] != 0.0 or pos['y'] != 0.0:
            found_objects += 1
            object_name = key.replace('_0', '')
            print(f'Found {object_name}: ({pos["x"]:.3f}, {pos["y"]:.3f})')
    
    print(f'Total objects located: {found_objects}/{len(object_list)}')