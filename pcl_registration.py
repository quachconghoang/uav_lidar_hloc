#%% read rosbag files and return a list of messages
import cv2
import numpy as np
import open3d as o3d
import ros_numpy
from utils import read_rosbag, read_rosbag_topics
import rosbag
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import rotate_point

bag_lidar_file = './Data/CEE_DATA_2024-11-25_08-26-00.bag'
bag_traj_file = './Data/ros_uav.bag'

traj_messages = read_rosbag(bag_traj_file)

# gps_messages = read_rosbag(bag_lidar_file, topic_filter='/gps')
topics = read_rosbag_topics(bag_lidar_file)
lidar_messages = read_rosbag(bag_lidar_file, topic_filter='quanergy/points')
image_messages = read_rosbag(bag_lidar_file, topic_filter='/camera/color/image_raw')
# get time stamps from lidar messages
lidar_time = []
image_time = []
for topic, msg, t in lidar_messages:
    if topic == 'quanergy/points':
        lidar_time.append(t.to_sec())

for topic, msg, t in image_messages:
    if topic == '/camera/color/image_raw':
        raw_time = msg.header.stamp.to_sec()
        image_time.append(raw_time)
        # print('diff:', t.to_sec()-raw_time)


#%% get xyz from bag traj file
traj_xyz = []
traj_orientation = []
traj_time = []
with rosbag.Bag(bag_traj_file, 'r') as bag:
    for topic, msg, t in traj_messages:
        if topic == 'trajectory_0':
            traj_xyz.append([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
            traj_orientation.append([msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w])
            traj_time.append(t.to_sec())

#%% get nearest lidar and image timestamps for each trajectory point
def get_nearest_timestamps(traj_time, lidar_time, image_time):
    nearest_lidar = []
    nearest_image = []

    for t in traj_time:
        nearest_lidar.append(min(lidar_time, key=lambda x: abs(x - t)))
        nearest_image.append(min(image_time, key=lambda x: abs(x - t)))

    return nearest_lidar, nearest_image

nearest_lidar_time, nearest_image_time = get_nearest_timestamps(traj_time, lidar_time, image_time)

# for id, trt in enumerate(traj_time):
#     # print(f"Trajectory time: {trt}, Nearest Lidar time: {nearest_lidar_time[id]-trt}, Nearest Image time: {nearest_image_time[id]-trt}")
#     if (abs(nearest_lidar_time[id]-trt) > 0.1 or abs(nearest_image_time[id]-trt) > 0.1):
#         print(f"Warning: Large time difference for trajectory point {id}: Lidar: {nearest_lidar_time[id]}, Image: {nearest_image_time[id]}")

#%% extract images and point clouds at nearest timestamps
images_ref = []
for topic, msg, t in image_messages:
    # print(topic, t.to_sec())
    timestamp = msg.header.stamp.to_sec()
    if timestamp in nearest_image_time:
        # Save or process the image
        print(f"Image at time {msg.header.stamp.to_sec()} extracted.")
        # For example, convert to numpy array
        image_np = ros_numpy.image.image_to_numpy(msg)
        images_ref.append(image_np)
        # You can save the image or process it further

# save images to disk with index 0000.jpg, 0001.jpg, etc.
import cv2 as cv
for index, img in enumerate(images_ref):
    filename = f'./Data/images/{index:04d}.jpg'
    cv.imwrite(filename, cv.cvtColor(img, cv.COLOR_RGB2BGR))
    print(f"Image saved to {filename}")

pcd_ref = []
for topic, msg, t in lidar_messages:
    if topic == 'quanergy/points':
        timestamp = t.to_sec()
        if timestamp in nearest_lidar_time:
            # Convert the point cloud message to a numpy array
            points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
            pcd_ref.append(points)
            print(f"Point cloud at time {timestamp} extracted.")

# save point clouds to disk with index 0000.pcd, 0001.pcd, etc.
for index, img in enumerate(pcd_ref):
    filename = f'./Data/pointclouds/{index:04d}.pcd'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(img)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")

#%% visualize trajectory in open3d
def visualize_trajectory(traj_xyz):
    points = np.array(traj_xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axes], window_name='Trajectory Visualization')

visualize_trajectory(traj_xyz)

#%% open3d ICP for 0410.pcd and 0415.pcd
def apply_icp(source, target, threshold=5):
    # Apply ICP
    reg_icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_icp.transformation

import copy
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source.paint_uniform_color([1, 0, 0])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source, source_temp, target_temp,axes])

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

#%% Load point clouds
pcd_source = load_point_cloud('./Data/pointclouds/0200.pcd')
pcd_target = load_point_cloud('./Data/pointclouds/0210.pcd')

# pcl filer by distance and z-axis
def filter_point_cloud(pcd, distance_threshold=80, z_min=-20, z_max=10):
    points = np.asarray(pcd.points)
    # Filter points based on distance and z-axis
    mask = (np.linalg.norm(points[:, :2], axis=1) < distance_threshold) & (points[:, 2] > z_min) & (points[:, 2] < z_max)
    filtered_points = points[mask]
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    return pcd

# pcd_source = filter_point_cloud(pcd_source, z_min=-12)
# pcd_target = filter_point_cloud(pcd_target, z_min=-12)

# compute normals for point clouds
# pcd_source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=30))
# pcd_target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=30))
# Apply ICP

transformation = apply_icp(pcd_source, pcd_target, threshold=15)
# Visualize the registration result
draw_registration_result(pcd_source, pcd_target, transformation)

