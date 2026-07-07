from utils import read_rosbag, read_rosbag_topics
import rosbag
import numpy as np
import math
import matplotlib.pyplot as plt

bag_traj_fastlivo_file = './Data/2026/fastlivo_1.bag'

# %% traj messages list
traj = read_rosbag(bag_traj_fastlivo_file)[-1][1].poses
traj_xy_1_fastlivo = []

for p in traj:
    traj_xy_1_fastlivo.append([p.pose.position.x, p.pose.position.z])
traj_xy_fastlivo = np.asarray(traj_xy_1_fastlivo)
# rotate right 90 degrees
traj_xy_fastlivo = np.array([[y, -x] for x, y in traj_xy_fastlivo])

# save to csv
# import csv
# csv_ours_file = './Data/2026/traj_xy_fastlivo_1fps.csv'
# with open(csv_ours_file, 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(['x', 'y'])
#     for row in traj_xy_fastlivo:
#         csvwriter.writerow(row)

# %% plot traj
plt.figure()
plt.plot(traj_xy_fastlivo[:, 0], traj_xy_fastlivo[:, 1], label='FastLIOv2')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Trajectory from FastLIOv2')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()


#%% compare with ground truth
import pandas as pd
import csv
csv_gt_file = './Data/2026/final_csv/gt_ardupilot_aligned.csv'
csv_ours_file = './Data/2026/final_csv/traj_xy_ours_aligned.csv'

traj_xy_gt = []
with open(csv_gt_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip header
    for row in csvreader:
        x = float(row[0])
        y = float(row[1])
        traj_xy_gt.append([x, y])
traj_xy_gt = np.asarray(traj_xy_gt)

traj_xy_ours = []
with open(csv_ours_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip header
    for row in csvreader:
        x = float(row[0])
        y = float(row[1])
        traj_xy_ours.append([x, y])
traj_xy_ours = np.asarray(traj_xy_ours)
# scale x with 1.1
traj_xy_ours[:, 0] = traj_xy_ours[:, 0] * 1.18

#%% save traj_xy_ours to csv
# import csv
out_file = './Data/2026/final_csv/traj_xy_ours_aligned_adjusted.csv'
with open(out_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['x', 'y'])
    for row in traj_xy_ours:
        csvwriter.writerow(row)
#%% plot comparison


plt.figure()
plt.plot(traj_xy_fastlivo[:, 0], traj_xy_fastlivo[:, 1], label='FastLIOv2 Trajectory XY')
plt.plot(traj_xy_gt[:, 0], traj_xy_gt[:, 1], label='Ground Truth Trajectory XY')
plt.plot(traj_xy_ours[:, 0], traj_xy_ours[:, 1], label='Our Trajectory XY')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('UAV Trajectory XY Comparison')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()
