# import hloc with superglue lightglue matcher
from utils import read_rosbag, read_rosbag_topics
import rosbag

import numpy as np
import matplotlib.pyplot as plt

import gtsam
from gtsam import Point2, Point3, Rot3, Pose3, Cal3_S2, Values
from gtsam import PinholeCameraCal3_S2

bag_traj_fastlivo_file = './Data/2026/fastlivo_1.bag'
bag_raw_data = './Data/2026/IAE_DATA_2026-01-10_15-30-26_fixed_timestamps.bag'
traj = read_rosbag(bag_traj_fastlivo_file)[-1][1].poses

#%% extract timestamps from traj
traj_time_fastlivo = []
traj_poses_fastlivo = []
S = np.asarray([[-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
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
    new_T = S @ T @ S
    traj_poses_fastlivo.append(new_T)
traj_time_fastlivo = np.asarray(traj_time_fastlivo)

#%% visualize traj in open3d
import open3d as o3d
traj_points = [Pose3(Rot3(T[:3, :3]), Point3(T[:3, 3])) for T in traj_poses_fastlivo]
traj_lines = [[i, i + 1] for i in range(len(traj_points) - 1)]
traj_colors = [[1, 0, 0] for _ in traj_lines]
traj_line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector([p.translation() for p in traj_points]),
    lines=o3d.utility.Vector2iVector(traj_lines),)
traj_line_set.colors = o3d.utility.Vector3dVector(traj_colors)

# draw origin
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

# draw camera points as small spheres for every 10th point with text labeling the index
camera_poses = []
for i, T in enumerate(traj_poses_fastlivo):
    if i % 10 == 0:

        camera_pose = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        axe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        axe.transform(T)
        camera_pose.translate(T[:3, 3])
        camera_poses.append(camera_pose)
        camera_poses.append(axe)

o3d.visualization.draw_geometries([traj_line_set, axis] + camera_poses)

#%% get image & poses
import os
img_dir = './Data/2026/images_fastlivo/'
img_list = os.listdir(img_dir)
img_list.sort()
img_paths = [os.path.join(img_dir, img) for img in img_list]

src_id = 260; tar_id= 280

src_img_path = img_paths[src_id]
tar_img_path = img_paths[tar_id]
src_pose = traj_poses_fastlivo[src_id]
tar_pose = traj_poses_fastlivo[tar_id]

motion = np.linalg.inv(tar_pose) @ src_pose
print("Motion from src to tar:\n", motion)
# convert motion to rotation in degree and translation
R = motion[:3, :3]
t = motion[:3, 3]
print("Rotation:\n", R)
print("Translation:\n", t)
print("Rotation in degree:\n", np.degrees(gtsam.Rot3(R).rpy()))

# draw point2D by motion
import cv2
src_img = cv2.imread(src_img_path)
tar_img = cv2.imread(tar_img_path)
h, w = src_img.shape[:2]

f = 615
cx = 320
cy = 240

p2d = np.array([w//2, h//2])
center_pt_3D = np.array([[(p2d[0]-cx)/f, (p2d[1]-cy)/f, 20 , 1]]).T
center_pt_3D_transformed = motion @ center_pt_3D
print("Center point in src image (3D):", center_pt_3D)
print("Center point transformed to tar image (3D):", center_pt_3D_transformed)
# convert only x,y to 2D pixel coordinates in tar image
center_pt_2D_transformed = np.array([center_pt_3D_transformed[0][0]*f/center_pt_3D_transformed[2][0]+cx,
                                     center_pt_3D_transformed[1][0]*f/center_pt_3D_transformed[2][0]+cy])
print("Center point transformed to tar image (2D):", center_pt_2D_transformed)

#%% draw circle at center point in src image and transformed center point in tar image
# src_img_with_circle = src_img.copy()
# cv2.circle(src_img_with_circle, (w//2, h//2), 10, (0, 255, 0), -1)
# tar_img_with_circle = cv2.imread(tar_img_path)
# pt = (int(center_pt_2D_transformed[0]), int(center_pt_2D_transformed[1]))
# cv2.circle(tar_img_with_circle, pt, 10, (0, 0, 255), -1)
# cv2.circle(tar_img_with_circle, (320,240), 10, (0, 255, 0), 2)
# # show images with circles
# cv2.imshow('src_img_with_circle', src_img_with_circle)
# cv2.imshow('tar_img_with_circle', tar_img_with_circle)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#%% superpoint extract and superglue match

import torch
from image_proc_2D import getSuperPoints_v2
torch.set_grad_enabled(False)

src_gray = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
tar_gray = cv2.cvtColor(tar_img, cv2.COLOR_RGB2GRAY)

pts0 = getSuperPoints_v2(src_gray)
pts1 = getSuperPoints_v2(tar_gray)

kp0 = pts0['pts']
kp1 = pts1['pts']
desc0 = pts0['desc']
desc1 = pts1['desc']

# discard all kp1 with x< 100
mask_x = kp1[:, 0] > 220
mask_y = kp1[:, 1] < 410
mask = mask_x & mask_y
kp1_original = kp1.copy()
kp1 = kp1[mask]
desc1 = desc1[:, mask]

#%% draw kp0 and kp1 on images
src_img_kp = src_img.copy()
tar_img_kp = tar_img.copy()
for pt in kp0:
    cv2.drawMarker(src_img_kp, (int(pt[0]), int(pt[1])), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10)
for pt in kp1:
    cv2.drawMarker(tar_img_kp, (int(pt[0]), int(pt[1])), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10)
# show images with keypoints
cv2.imshow('src_img_kp', src_img_kp)
cv2.imshow('tar_img_kp', tar_img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
### Get Sinkhorn Map
import ot
from scipy.optimize import linear_sum_assignment

norm_self_src = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc0), -1, 1))
norm_self_tar = np.sqrt(2 - 2 * np.clip(np.dot(desc1.T, desc1), -1, 1))
norm_cross = np.sqrt(2 - 2 * np.clip(np.dot(desc0.T, desc1), -1, 1))



a, b = [], []
for pt_his in norm_self_src:
    a.append(pt_his.mean() - 1)
for pt_his in norm_self_tar:
    b.append(pt_his.mean() - 1)

Gs = ot.sinkhorn(a, b, norm_cross, reg=1e-1, numItermax=20)
### Sinkhorn with 3D project map
match_id = linear_sum_assignment(cost_matrix=(1 - Gs))
match_score = norm_cross[match_id]
matches = np.stack((match_id[0], match_id[1], match_score), axis=1)

#%% draw matches on images
# merge src and tar images side by side
src_img_match = src_img.copy()
tar_img_match = tar_img.copy()
canvas = np.hstack((src_img_match, tar_img_match))
overlay = np.zeros_like(canvas)
for pt in kp0:
    cv2.drawMarker(canvas, (int(pt[0]), int(pt[1])), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=7)
# for pt in kp1_original:
#     cv2.drawMarker(canvas, (int(pt[0]+640), int(pt[1])), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=7)
for pt in kp1_original:
    cv2.drawMarker(canvas, (int(pt[0]+640), int(pt[1])), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=7)

for m in matches:
    pt0 = kp0[int(m[0])]
    pt1 = kp1[int(m[1])]
    if m[2] < 0.8 : # only draw matches with score < 0.5
        cv2.line(overlay, (int(pt0[0]), int(pt0[1])), (int(pt1[0]+640), int(pt1[1])), (0, 255, 0), 1)

# alphetically blend the overlay with the canvas
alpha = 0.5
cv2.addWeighted(overlay, alpha, canvas, 1, 0, canvas)
# show the matches
cv2.imshow('matches', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
