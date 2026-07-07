#%% read rosbag files and return a list of messages
import cv2
import numpy as np
import open3d as o3d
import ros_numpy

from utils import read_rosbag, read_rosbag_topics, get_nearest_timestamps
import rosbag
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import rotate_point
import cv2 as cv

bag_lidar_file = './Data/2026/IAE_DATA_2026-01-10_15-30-26_fixed_timestamps.bag'
bag_traj_file = './Data/2026/iae_2026_cartographer.bag'

traj_messages = read_rosbag(bag_traj_file)
topics = read_rosbag_topics(bag_lidar_file)

EXPORT_BAG = False

if EXPORT_BAG:
    lidar_messages = read_rosbag(bag_lidar_file, topic_filter='/quanergy/points')
    image_messages = read_rosbag(bag_lidar_file, topic_filter='/camera/color/image_raw')
    # get time stamps from lidar messages
    lidar_time = []
    image_time = []
    for topic, msg, t in lidar_messages:
        if topic == '/quanergy/points':
            lidar_time.append(t.to_sec())

    for topic, msg, t in image_messages:
        if topic == '/camera/color/image_raw':
            raw_time = msg.header.stamp.to_sec()
            image_time.append(raw_time)
            # print('diff:', t.to_sec()-raw_time)

    traj_xyz = []
    traj_orientation = []
    traj_time = []
    with rosbag.Bag(bag_traj_file, 'r') as bag:
        for topic, msg, t in traj_messages:
            if topic == 'trajectory_0':
                traj_xyz.append([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
                traj_orientation.append([msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z,
                                         msg.transform.rotation.w])
                traj_time.append(t.to_sec())

    nearest_lidar_time, nearest_image_time = get_nearest_timestamps(traj_time, lidar_time, image_time)

    # for id, trt in enumerate(traj_time):
    #     # print(f"Trajectory time: {trt}, Nearest Lidar time: {nearest_lidar_time[id]-trt}, Nearest Image time: {nearest_image_time[id]-trt}")
    #     if (abs(nearest_lidar_time[id]-trt) > 0.1 or abs(nearest_image_time[id]-trt) > 0.1):
    #         print(f"Warning: Large time difference for trajectory point {id}: Lidar: {nearest_lidar_time[id]}, Image: {nearest_image_time[id]}")

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
    for index, img in enumerate(images_ref):
        filename = f'./Data/2026/images/{index:04d}.jpg'
        cv.imwrite(filename, cv.cvtColor(img, cv.COLOR_RGB2BGR))
        print(f"Image saved to {filename}")
    pcd_ref = []
    for topic, msg, t in lidar_messages:
        if topic == '/quanergy/points':
            timestamp = t.to_sec()
            if timestamp in nearest_lidar_time:
                # Convert the point cloud message to a numpy array
                points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
                pcd_ref.append(points)
                print(f"Point cloud at time {timestamp} extracted.")

    # save point clouds to disk with index 0000.pcd, 0001.pcd, etc.
    for index, img in enumerate(pcd_ref):
        filename = f'./Data/2026/pointclouds/{index:04d}.pcd'
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(img)
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Point cloud saved to {filename}")

#%% multiway registration
from pathlib import Path
pcd_dir = Path('./Data/2026/pointclouds')
#list all pcd files in the directory
files = list(pcd_dir.glob('*.pcd'))
files.sort()

def filter_point_cloud(pcd, distance_threshold=80, z_min=-20, z_max=10):
    points = np.asarray(pcd.points)
    # Filter points based on distance and z-axis
    mask = (np.linalg.norm(points[:, :2], axis=1) < distance_threshold) & (points[:, 2] > z_min) & (points[:, 2] < z_max)
    filtered_points = points[mask]
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    return pcd

pcds = []
for file in files:
    pcd = o3d.io.read_point_cloud(file)
    pcd_down = filter_point_cloud(pcd, distance_threshold=80, z_min=-12, z_max=30)
    pcds.append(pcd_down)

# %%

voxel_size = 0.02
max_correspondence_distance_coarse = 20
max_correspondence_distance_fine = 0.3

pcds_down = pcds[320:380].copy()  # Select a subset of point clouds for registration # 310-380 matters

o3d.visualization.draw_geometries(pcds_down)

def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

print("Full registration ...")

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcds_down,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)

print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

print("Transform points and display")
for point_id in range(len(pcds_down)):
    print(pose_graph.nodes[point_id].pose)
    pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)

# draw axes
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
o3d.visualization.draw_geometries([axis] + pcds_down)
# o3d.visualization.draw_geometries(pcds_down)

#%% load raw traj
bag_traj_file = './Data/2026/iae_2026_cartographer.bag'
traj_messages = read_rosbag(bag_traj_file)

traj_ref = traj_messages[320*2][1]
xyz = [traj_ref.transform.translation.x, traj_ref.transform.translation.y, traj_ref.transform.translation.z]
rot_quat = [traj_ref.transform.rotation.x, traj_ref.transform.rotation.y, traj_ref.transform.rotation.z, traj_ref.transform.rotation.w]

# convert to 4x4 transformation matrix
def quat_to_transformation_matrix(quat, translation):
    x, y, z, w = quat
    R = np.array([[1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T

trans_ref = quat_to_transformation_matrix(rot_quat, xyz)

#%% create pose graph and save to disk
traj_ends= []
for i, node in enumerate(pose_graph.nodes):
    print(f"Node {i}: {node.pose}")
    # multiply the pose with the reference transformation
    local_pose = np.asarray(node.pose)
    global_pose = np.dot(local_pose, trans_ref)
    traj_ends.append(global_pose)

traj_ends_xy = []
for traj_mat in traj_ends:
    # extract x, y from the transformation matrix
    x = traj_mat[0, 3]
    y = traj_mat[1, 3]
    traj_ends_xy.append([x, y])

#%% tranj_ends_xy nice &&static rotation for stupid IMU installation
tranj_ends_xy_nice = traj_ends_xy[:40]

# rotate 120 z axis
angle = 120 * math.pi / 180
# rotation origin at the first point of tranj_ends_xy_nice
origin_x = tranj_ends_xy_nice[0][0]
origin_y = tranj_ends_xy_nice[0][1]

for i in range(len(tranj_ends_xy_nice)):
    x = tranj_ends_xy_nice[i][0] - origin_x
    y = tranj_ends_xy_nice[i][1] - origin_y
    x_new = x * math.cos(angle) - y * math.sin(angle)
    y_new = x * math.sin(angle) + y * math.cos(angle)
    tranj_ends_xy_nice[i][0] = x_new + origin_x
    tranj_ends_xy_nice[i][1] = y_new + origin_y

#%% draw traj_end xy
plt.figure()
plt.plot([p[0] for p in tranj_ends_xy_nice], [p[1] for p in tranj_ends_xy_nice], marker='o', label='Pose Graph Ends')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Pose Graph End Points XY')
plt.axis('equal')
plt.grid()
plt.show()

#%% draw trajectory
traj_xy = []
traj_time = []

with rosbag.Bag(bag_traj_file, 'r') as bag:
    for topic, msg, t in traj_messages:
        if topic == 'trajectory_0':
            # print(f"Time: {t.to_sec()}, Position: {msg.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
            print(f"Orientation: {msg.transform.translation.x}, {msg.transform.translation.y}")
            traj_xy.append([msg.transform.translation.x, msg.transform.translation.y])
            traj_time.append(t.to_sec())

traj_xy_new = traj_xy[0:320]  # Use the first 400 points for the trajectory
# add 40 last points from the pose graph
for xy in tranj_ends_xy_nice:
    traj_xy_new.append(xy)
traj_xy = traj_xy[0:360]

# draw traj xy
plt.figure(figsize=(10, 10))
# keep x and y have the same scale
plt.plot([p[0] for p in traj_xy], [p[1] for p in traj_xy], marker='o', markersize=2, label='Old Trajectory')

plt.plot([p[0] for p in traj_xy_new], [p[1] for p in traj_xy_new], marker='o', markersize=2, label='Trajectory')
plt.axis('equal')
plt.title('Trajectory from Pose Graph')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid()
plt.legend()
plt.show()


#%% save traj_xy and traj_xy_new to csv
import pandas as pd
df_old = pd.DataFrame(traj_xy, columns=['x', 'y'])
df_old.to_csv('./Data/2026/traj_xy_cartographer.csv', index=False)
df_new = pd.DataFrame(traj_xy_new, columns=['x', 'y'])
df_new.to_csv('./Data/2026/traj_xy_ours.csv', index=False)

#%% load and compare with ground truth
import csv
csv_gt_file = './Data/2026/final_csv/gt_ardupilot_aligned.csv'
traj_xy_gt = []
with open(csv_gt_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip header
    for row in csvreader:
        x = float(row[0])
        y = float(row[1])
        traj_xy_gt.append([x, y])
traj_xy_gt = np.asarray(traj_xy_gt)
plt.figure(figsize=(10, 10))
plt.plot([p[0] for p in traj_xy_gt], [p[1] for p in traj_xy_gt], label='Ground Truth Trajectory XY')
plt.plot([p[0] for p in traj_xy], [p[1] for p in traj_xy], label='Cartographer Trajectory XY')
plt.plot([p[0] for p in traj_xy_new], [p[1] for p in traj_xy_new], label='Our Trajectory XY')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')