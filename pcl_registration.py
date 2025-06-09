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

bag_lidar_file = './Data/CEE_DATA_2024-11-25_08-26-00.bag'
bag_traj_file = './Data/ros_uav.bag'

traj_messages = read_rosbag(bag_traj_file)
# gps_messages = read_rosbag(bag_lidar_file, topic_filter='/gps')
topics = read_rosbag_topics(bag_lidar_file)

EXPORT_BAG = False

if EXPORT_BAG:
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


#%% multiway registration
from pathlib import Path
pcd_dir = Path('./Data/pointclouds')
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

pcds_down = pcds[400:440].copy()  # Select a subset of point clouds for registration

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
o3d.visualization.draw_geometries(pcds_down)

#%% load raw traj
bag_traj_file = './Data/ros_uav.bag'
traj_messages = read_rosbag(bag_traj_file)
traj_ref = traj_messages[400*2][1]
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

traj_xy_new = traj_xy[0:400]  # Use the first 400 points for the trajectory
# add 40 last points from the pose graph
for xy in traj_ends_xy:
    traj_xy_new.append(xy)

ref0 = traj_xy[0]
ref1 = traj_xy[220]
# rotate traj_xy around ref0 to align ref1 with x-axis
angle = math.atan2(ref1[1] - ref0[1], ref1[0] - ref0[0])
traj_xy_new = [rotate_point(p, -angle, ref0) for p in traj_xy_new]
traj_xy = [rotate_point(p, -angle, ref0) for p in traj_xy]

# draw traj xy
plt.figure(figsize=(10, 10))
# keep x and y have the same scale
plt.plot([p[0] for p in traj_xy_new], [p[1] for p in traj_xy_new], marker='o', markersize=2, label='Trajectory')
plt.plot([p[0] for p in traj_xy], [p[1] for p in traj_xy], marker='o', markersize=2, label='Old Trajectory')
plt.axis('equal')
plt.title('Trajectory from Pose Graph')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid()
plt.legend()
plt.show()

#%% pair with gps data
gps_messages = read_rosbag(bag_lidar_file, topic_filter='/gps')
gps_xy = []
gps_altitude = []
gps_time = []
with rosbag.Bag(bag_lidar_file, 'r') as bag:
    for topic, msg, t in gps_messages:
        if topic == '/gps/':
            print(f"Time: {t.to_sec()}, Latitude: {msg.latitude}, Longitude: {msg.longitude}")
            gps_xy.append([msg.longitude, msg.latitude])
            gps_altitude.append(msg.altitude)
            gps_time.append(t.to_sec())

# convert gps_xy lat/lon to metric coordinates
gps_xy = [[lat * 111320, lon * 111320] for lat, lon in gps_xy]  # Rough conversion, not accurate for large distances
gps_xy = np.asarray(gps_xy)
gps_xy = gps_xy - gps_xy[0]  # Normalize to start at origin
gps_altitude = np.asarray(gps_altitude)

ref0 = gps_xy[0]
ref1 = gps_xy[2400]
# rotate gps_xy around ref0 to align ref1 with x-axis
angle = math.atan2(ref1[1] - ref0[1], ref1[0] - ref0[0])
gps_xy = [rotate_point(p, -angle, ref0) for p in gps_xy]

# %% visualize adjusted gps_xy and traj_xy
draw_gps = []
draw_xy = []
draw_xy_old = []

plt.figure(figsize=(10, 10))
# keep x and y have the same scale
plt.axis('equal')
# set limits y from -50 to 10
# plt.ylim(-50, 10)
plt.plot([p[0]-20 for p in gps_xy[1250:]], [p[1] for p in gps_xy[1250:]], marker='o', markersize=1, linestyle='-', color='red')
plt.plot([p[0] for p in traj_xy_new[:430]], [p[1] for p in traj_xy_new[0:430]], marker='o', markersize=1, linestyle='-', color='green', alpha=0.5, label='New Trajectory')
# draw old traj_xy
plt.plot([p[0] for p in traj_xy], [p[1] for p in traj_xy], marker='o', markersize=1, linestyle='-', color='blue', alpha=0.5, label='Old Trajectory')
# add index text to each every 20th point in traj_xy
for i in range(100, len(traj_xy_new)-9, 10):
    plt.text(traj_xy_new[i][0], traj_xy_new[i][1], str(i), fontsize=8, color='green')
    draw_xy.append([traj_xy_new[i][0], traj_xy_new[i][1]])
    draw_xy_old.append([traj_xy[i][0], traj_xy[i][1]])
    t = traj_time[i]
    nearest_gps_time = min(gps_time, key=lambda x: abs(x - t))
    nearest_gps_index = gps_time.index(nearest_gps_time)
    print(nearest_gps_index)
    gt = gps_xy[nearest_gps_index]
    draw_gps.append([gt[0]-19.5, gt[1]])
    plt.text(gt[0]-19.5, gt[1], str(i), fontsize=8, color='red')
# add index to each point in gps_xy
plt.title('Adjusted GPS and Traj XY Plot')
# set units as meters
plt.xlabel('X (meters)')
# draw grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# add markers for every 100th point
# for i in range(0, len(gps_xy), 100):
#     plt.text(gps_xy[i][0], gps_xy[i][1], str(i), fontsize=8, color='green')
plt.show()

# redraw draw_gps and draw_xy
plt.figure(figsize=(10, 10))
plt.axis('equal')
plt.plot([p[0] for p in draw_gps], [p[1] for p in draw_gps], marker='o', markersize=2, label='GPS')
plt.plot([p[0] for p in draw_xy], [p[1] for p in draw_xy], marker='o', markersize=2, label='Trajectory')
plt.plot([p[0] for p in draw_xy_old], [p[1] for p in draw_xy_old], marker='o', markersize=2, label='Old Trajectory')
plt.title('GPS and Trajectory Points')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()

# save draw_gps and draw_xy to csv
import pandas as pd
draw_gps_df = pd.DataFrame(draw_gps, columns=['X', 'Y'])
draw_xy_df = pd.DataFrame(draw_xy, columns=['X', 'Y'])
draw_xy_old_df = pd.DataFrame(draw_xy_old, columns=['X', 'Y'])
draw_gps_df.to_csv('./Data/gps_points.csv', index=False)
draw_xy_df.to_csv('./Data/traj_points.csv', index=False)
draw_xy_old_df.to_csv('./Data/traj_points_old.csv', index=False)


#%% group and save pcds_down to disk

# full_pcd = o3d.geometry.PointCloud()
# for pcd in pcds_down:
#     full_pcd+= pcd
# full_pcd = full_pcd.voxel_down_sample(voxel_size=0.02)
# o3d.io.write_point_cloud('./Data/pointclouds/pcl_400to440.pcd', full_pcd)

