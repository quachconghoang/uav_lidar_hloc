#%% read rosbag files and return a list of messages
import numpy as np
import open3d as o3d
import ros_numpy
from kornia.geometry import camtoworld_vision_to_graphics_Rt

from utils import read_rosbag, read_rosbag_topics
import rosbag
import numpy as np
import math
import matplotlib.pyplot as plt
from utils import rotate_point

bag_lidar_file = './Data/2026/IAE_DATA_2026-01-10_15-30-26.bag'
bag_traj_file = './Data/2026/iae_2026_cartographer.bag'
bag_traj_fastlivo_file = './Data/2026/iae_2026_fastlivo2.bag'

csv_carto_file = './Data/2026/traj_xy_cartographer.csv'
csv_ours_file = './Data/2026/traj_xy_ours.csv'

#%% traj messages
traj_messages = read_rosbag(bag_traj_file)
# topics = read_rosbag_topics(bag_lidar_file)

#%% get xyz from bag traj file
traj_xy_carto = []
traj_time_carto = []
with rosbag.Bag(bag_traj_file, 'r') as bag:
    for topic, msg, t in traj_messages:
        if topic == 'trajectory_0':
            # print(f"Time: {t.to_sec()}, Position: {msg.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
            print(f"Orientation: {msg.transform.translation.x}, {msg.transform.translation.y}")
            traj_xy_carto.append([msg.transform.translation.x, msg.transform.translation.y])
            traj_time_carto.append(t.to_sec())

#%% get xy from csv traj file
import csv
traj_xy_carto = []
with open(csv_carto_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip header
    for row in csvreader:
        x = float(row[0])
        y = float(row[1])
        traj_xy_carto.append([x, y])
traj_xy_carto = np.asarray(traj_xy_carto)

traj_xy_ours = []
with open(csv_ours_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip header
    for row in csvreader:
        x = float(row[0])
        y = float(row[1])
        traj_xy_ours.append([x, y])
traj_xy_ours = np.asarray(traj_xy_ours)

#%% bag fastlivo
bag_traj_file = './Data/2026/iae_2026_fastlivo2.bag'
traj_messages = read_rosbag(bag_traj_file)[0][1].poses

traj_xy_fastlivo = []
traj_time_fastlivo = []

for p in traj_messages:
    pose = p.pose
    t = p.header.stamp.to_sec()
    traj_xy_fastlivo.append([pose.position.x, pose.position.z])
    traj_time_fastlivo.append(t)
    # print(t, pose.position.x, pose.position.y, pose.position.z)

# get last 5400 points
# traj_xy_fastlivo = traj_xy_fastlivo[-5400:]
# traj_time_fastlivo = traj_time_fastlivo[-5400:]

#%% draw traj_xy_fastlivo and compare with carto
traj_xy_fastlivo = np.asarray(traj_xy_fastlivo)
plt.figure()
plt.plot(traj_xy_fastlivo[:, 0], traj_xy_fastlivo[:, 1], label='FastLIVO2 Trajectory XY')
plt.plot(traj_xy_carto[:, 0], traj_xy_carto[:, 1], label='Cartographer Trajectory XY')
plt.plot(traj_xy_ours[:, 0], traj_xy_ours[:, 1], label='Our Trajectory XY')

# for i in range(0, len(traj_xy_fastlivo), 540):
#     plt.scatter(traj_xy_fastlivo[i, 0], traj_xy_fastlivo[i, 1], label=f'Point {i}', s=50)

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('UAV Trajectory XY Comparison')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()



#%% gt from pixhawk (csv)
import pandas as pd
gt_file = './Data/2026/2060-01-10.csv'
gt_data = pd.read_csv(gt_file)

gt_ardupilot = []
gt_time = []
for index, row in gt_data.iterrows():
    date = row['Date']
    time = row['Time']
    x = row['POS.Lat']
    y = row['POS.Lng']
    z = row['POS.Alt']
    gt_ardupilot.append([x, y])
    # convert date time to UTC seconds

    gt_time.append(t)

# convert lat lon to meters using simple equirectangular approximation
gt_ardupilot = np.asarray(gt_ardupilot)
gt_ardupilot_meters = []
for lat, lon in gt_ardupilot:
    x = lon * 111320
    y = lat * 110540
    gt_ardupilot_meters.append([x, y])
gt_ardupilot_meters = np.asarray(gt_ardupilot_meters)
# normalize to start at origin
gt_ardupilot_meters = gt_ardupilot_meters - gt_ardupilot_meters[0]

# rotate to align traj_xy_fastlivo and gt with Ox axis
ref0 = gt_ardupilot_meters[0]
ref1 = gt_ardupilot_meters[1500]
angle = math.atan2(ref1[1] - ref0[1], ref1[0] - ref0[0])
gt_ardupilot_meters = [rotate_point(p, -angle, ref0) for p in gt_ardupilot_meters]
gt_ardupilot_meters = np.asarray(gt_ardupilot_meters)

ref0 = traj_xy_fastlivo[0]
ref1 = traj_xy_fastlivo[3000]
angle = math.atan2(ref1[1] - ref0[1], ref1[0] - ref0[0])
traj_xy_fastlivo = [rotate_point(p, -angle, ref0) for p in traj_xy_fastlivo]
traj_xy_fastlivo = np.asarray(traj_xy_fastlivo)

ref0 = traj_xy_carto[0]
ref1 = traj_xy_carto[220]
angle = math.atan2(ref1[1] - ref0[1], ref1[0] - ref0[0])
traj_xy_carto = [rotate_point(p, -angle, ref0) for p in traj_xy_carto]
traj_xy_carto = np.asarray(traj_xy_carto)

traj_xy_ours = [rotate_point(p, -angle, ref0) for p in traj_xy_ours]
traj_xy_ours = np.asarray(traj_xy_ours)

#%% truncate data
traj_xy_fastlivo = traj_xy_fastlivo[-5400:]
# sample to 360 points
traj_xy_fastlivo = traj_xy_fastlivo[::15]

#truncate ardupilot to 3600 points
gt_ardupilot_meters = gt_ardupilot_meters[-3600:]
# sample to 360 points
gt_ardupilot_meters = gt_ardupilot_meters[::10]

#%% plot gt_ardupilot_meters
plt.figure()
plt.plot(gt_ardupilot_meters[:, 0], gt_ardupilot_meters[:, 1], label='ArduPilot GT XY')
plt.plot(traj_xy_fastlivo[:, 0], traj_xy_fastlivo[:, 1], label='FastLIVO2 Trajectory XY')
plt.plot(traj_xy_carto[:, 0], traj_xy_carto[:, 1], label='Cartographer Trajectory XY')
plt.plot(traj_xy_ours[:, 0], traj_xy_ours[:, 1], label='Our Trajectory XY')
# add markers each 500 points
# for i in range(0, len(gt_ardupilot_meters), 500):
#     plt.scatter(gt_ardupilot_meters[i, 0], gt_ardupilot_meters[i, 1], label=f'Point {i}', s=50)

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('ArduPilot Ground Truth XY')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()

#%% export all trajs to csv
import csv
output_csv_file = './Data/2026/traj_xy_fastlivo2_aligned.csv'
with open(output_csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['X', 'Y'])
    for point in traj_xy_fastlivo:
        csvwriter.writerow(point)
output_csv_file = './Data/2026/traj_xy_carto_aligned.csv'
with open(output_csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['X', 'Y'])
    for point in traj_xy_carto:
        csvwriter.writerow(point)
output_csv_file = './Data/2026/traj_xy_ours_aligned.csv'
with open(output_csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['X', 'Y'])
    for point in traj_xy_ours:
        csvwriter.writerow(point)
output_csv_file = './Data/2026/gt_ardupilot_aligned.csv'
with open(output_csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['X', 'Y'])
    for point in gt_ardupilot_meters:
        csvwriter.writerow(point)
#%% end