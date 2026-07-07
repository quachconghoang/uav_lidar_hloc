from utils import read_rosbag, read_rosbag_topics
import rosbag
import numpy as np
import math
import matplotlib.pyplot as plt
import gtsam

bag_traj_fastlivo_file = './Data/2026/fastlivo_1.bag'
bag_raw_data = './Data/2026/IAE_DATA_2026-01-10_15-30-26_fixed_timestamps.bag'
traj = read_rosbag(bag_traj_fastlivo_file)[-1][1].poses

#%% extract timestamps from traj
traj_time_fastlivo = []
traj_poses_fastlivo = []
for p in traj:
    traj_time_fastlivo.append(p.header.stamp.to_sec())
    orient = p.pose.orientation
    pos = p.pose.position
    # to 4x4 homogeneous transformation matrix
    R = gtsam.Rot3.Quaternion(orient.x, orient.y, orient.z, orient.w).matrix()
    t = np.array([pos.x, pos.y, pos.z])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    traj_poses_fastlivo.append(T)
traj_time_fastlivo = np.asarray(traj_time_fastlivo)


#%% get image timestamps from raw data bag
image_timestamps = []
with rosbag.Bag(bag_raw_data, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        if topic == '/camera/color/image_raw':
            image_timestamps.append(t.to_sec())

#%% get the closest image timestamp for each traj timestamp (image timestamp should be more than or equal to traj timestamp)
closest_image_timestamps = []
for traj_time in traj_time_fastlivo:
    closest_image_timestamp = min(image_timestamps, key=lambda x: abs(x - traj_time)) #
    time_diff = closest_image_timestamp - traj_time
    if time_diff < 0:
        # print(f"Warning: Closest image timestamp {closest_image_timestamps} is before traj timestamp {traj_time}. Time difference: {time_diff}")
        # get the next image timestamp that is greater than traj_time
        next_image_timestamp = [t for t in image_timestamps if t > traj_time]
        if next_image_timestamp:
            closest_image_timestamp = min(next_image_timestamp, key=lambda x: abs(x - traj_time))
            time_diff = closest_image_timestamp - traj_time
            # print(f"Next closest image timestamp: {closest_image_timestamps}, Time difference: {time_diff}")

    print(f"--- Traj time: {traj_time}, Closest image time: {closest_image_timestamp}, Time difference: {time_diff}")
    closest_image_timestamps.append(closest_image_timestamp)

#%% get images at the closest timestamps
import cv2
import os
import ros_numpy
import sensor_msgs


image_folder = './Data/2026/images_fastlivo'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

def get_img_from_ros_image_msg(msg):
    msg.__class__ = sensor_msgs.msg.Image
    return ros_numpy.numpify(msg)

# save images at the closest timestamps
with rosbag.Bag(bag_raw_data, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        if topic == '/camera/color/image_raw':
            if t.to_sec() in closest_image_timestamps:
                # get the index of the closest timestamp
                index = closest_image_timestamps.index(t.to_sec())
                # Convert ROS Image message to OpenCV image
                cv_image = get_img_from_ros_image_msg(msg)
                # from BGR to RGB
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                # Save the image with indexed (000*) of the closest timestamp
                image_filename = os.path.join(image_folder, f"image_{index:04d}_{t.to_sec():.6f}.png")
                # image_filename = os.path.join(image_folder, f"image_{index:04d}.png")
                cv2.imwrite(image_filename, cv_image)
                print(f"Saved image at timestamp {t.to_sec()} to {image_filename}")