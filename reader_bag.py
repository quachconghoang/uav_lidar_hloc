#%% read rosbag files and return a list of messages
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
topics = read_rosbag_topics(bag_lidar_file)
gps_messages = read_rosbag(bag_lidar_file, topic_filter='/gps')

#%% get xyz from bag traj file
traj_xy = []
traj_time = []
with rosbag.Bag(bag_traj_file, 'r') as bag:
    for topic, msg, t in traj_messages:
        if topic == 'trajectory_0':
            # print(f"Time: {t.to_sec()}, Position: {msg.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
            print(f"Orientation: {msg.transform.translation.x}, {msg.transform.translation.y}")
            traj_xy.append([msg.transform.translation.x, msg.transform.translation.y])
            traj_time.append(t.to_sec())

#%% adjust traj_xy
ref0 = traj_xy[0]
ref1 = traj_xy[220]
# rotate traj_xy around ref0 to align ref1 with x-axis
angle = math.atan2(ref1[1] - ref0[1], ref1[0] - ref0[0])
traj_xy = [rotate_point(p, -angle, ref0) for p in traj_xy]


#%% get gps data
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

# %%
ref0 = gps_xy[0]
ref1 = gps_xy[2400]
# rotate gps_xy around ref0 to align ref1 with x-axis
angle = math.atan2(ref1[1] - ref0[1], ref1[0] - ref0[0])
gps_xy = [rotate_point(p, -angle, ref0) for p in gps_xy]

# %% visualize adjusted gps_xy and traj_xy
plt.figure(figsize=(10, 10))
# keep x and y have the same scale
plt.axis('equal')
# set limits y from -50 to 10
# plt.ylim(-50, 10)
plt.plot([p[0] for p in gps_xy], [p[1] for p in gps_xy], marker='o', markersize=1, linestyle='-', color='red')
plt.plot([p[0] for p in traj_xy], [p[1] for p in traj_xy], marker='o', markersize=1, linestyle='-', color='blue')
# add index text to each every 20th point in traj_xy
for i in range(100, len(traj_xy), 20):
    plt.text(traj_xy[i][0], traj_xy[i][1], str(i), fontsize=8, color='blue')
    t = traj_time[i]
    nearest_gps_time = min(gps_time, key=lambda x: abs(x - t))
    nearest_gps_index = gps_time.index(nearest_gps_time)
    gt = gps_xy[nearest_gps_index]
    plt.text(gt[0], gt[1], str(i), fontsize=8, color='red')
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

#%% get time stamps from gps messages
errors = []
for id, t in enumerate(traj_time):
    # find nearest gps message to each traj message
    nearest_gps_time = min(gps_time, key=lambda x: abs(x - t))
    print(f"Trajectory time: {t}, Nearest GPS time: {nearest_gps_time}, Difference: {abs(t - nearest_gps_time)}")
    # get index of nearest gps time
    nearest_gps_index = gps_time.index(nearest_gps_time)
    gt = gps_xy[nearest_gps_index]
    loc = traj_xy[id]
    print(f"Ground Truth: {gt}, Location: {loc}")
    # calculate error
    error = np.linalg.norm(np.array(gt) - np.array(loc))
    print(f"Error: {error}")
    errors.append(error)

#%% visualize errors
plt.figure(figsize=(10, 5))
plt.plot(errors, marker='o', linestyle='-', color='blue')
# vertical units as meters, horizontal units as seconds
plt.xlabel('Time (seconds)')
plt.ylabel('Error (meters)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

#%% get image messages at the same time of traj_time
image_messages = []
with rosbag.Bag(bag_lidar_file, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics='/camera/color/image_raw'):
        # find the nearest traj_time within 0.1 seconds
        nearest_traj_time = min(traj_time, key=lambda x: abs(x - t.to_sec()))
        if abs(nearest_traj_time - t.to_sec()) < 0.1:
            image_messages.append((topic, msg, t))
            # convert image message to numpy array
            img_array = ros_numpy.numpify(msg[1])
            # print image shape
            # print(f"Image shape: {img_array.shape}, Time: {t.to_sec()}")

#%% viz
# lidar_messages = read_rosbag(bag_lidar_file, topic_filter='quanergy/points')
#
# pc = o3d.geometry.PointCloud()
# # open3d non_block point cloud visualization
# viz = o3d.visualization.Visualizer()
# viz.create_window(window_name='Open3D Non-blocking')
#
# axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
# viz.add_geometry(axes)
# viz.add_geometry(pc)
#
# # convert lidar_messages to pointsclouds open3D
# for msg in lidar_messages:
#     pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg[1])
#     xyz = ros_numpy.point_cloud2.get_xyz_points(pc_array)
#     print(xyz.shape)
#     # pc_array to open3d point cloud
#     pc.points = o3d.utility.Vector3dVector(xyz)
#     viz.update_geometry(pc)
#     viz.poll_events()
#     viz.update_renderer()
#
# viz.destroy_window()




# lidar_messages = read_rosbag(bag_lidar_file)