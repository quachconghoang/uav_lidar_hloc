#%% Load the reconstruction model
import os
import ros_numpy, rosbag
from utils import read_rosbag, read_rosbag_topics
import numpy as np
import math
from utils import rotate_point, create_xy_grid
from hloc.utils.read_write_model import qvec2rotmat, rotmat2qvec

from hloc.utils.visualize_model import Model
import open3d as o3d

model = Model()
model.read_model(path='./outputs/sfm/sfm_scaled', ext='.bin')

print("num_cameras:", len(model.cameras))
print("num_images:", len(model.images))
print("num_points3D:", len(model.points3D))

# load images poses xy

imgs = model.images
# get tvec from images
tvecs = []
for img in imgs.values():
    R = qvec2rotmat(img.qvec)
    t = img.tvec
    t = -R.T @ t
    tvecs.append(t)
tvecs = np.asarray(tvecs)

ref0 = tvecs[18].copy()
ref1 = tvecs[0].copy()
angle = math.atan2(ref1[1] - ref0[1], ref1[0] - ref0[0])
# transform axis with Oz and angle
# display using Open3D visualization tools
# draw x, y, z axes
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
grid = create_xy_grid(cell_size=10, num_cells=10, origin=(0, 0, 0))
transform = np.array([[math.cos(angle), -math.sin(angle), 0, 0],
                        [math.sin(angle), math.cos(angle), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
transform[0, 3] = - 70
transform[1, 3] = - 30

axis.transform(transform)
grid.transform(transform)


vis = model.create_window()
vis.add_geometry(axis)
vis.add_geometry(grid)

model.add_points()
model.add_cameras(scale=1)
model.show()

# tvecs = tvecs - tvecs[18]  # Normalize to the 36th image
# # draw tvecs as points in Open3D
# pc = o3d.geometry.PointCloud()
# pc.points = o3d.utility.Vector3dVector(tvecs)
# o3d.visualization.draw_geometries([pc, axis], window_name='Camera Poses', width=800, height=600)

# translate pc to pc.po