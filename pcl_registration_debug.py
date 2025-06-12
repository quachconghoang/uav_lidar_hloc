#%% read ro
import cv2
import numpy as np
import open3d as o3d
import ros_numpy
import copy
from utils import create_xy_grid

source = o3d.io.read_point_cloud('./Data/pointclouds/0410.pcd')
target = o3d.io.read_point_cloud('./Data/pointclouds/0415.pcd')


threshold = 15

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_init = copy.deepcopy(source)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_init.paint_uniform_color([1, 0, 0])
    source_temp.transform(transformation)
    # add text in axe (Ox red, Oy green, Oz blue)

    axe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, -17])
    ground = create_xy_grid(cell_size=10, num_cells=10, origin=(-50, -50, -17))

    # customize o3d visualizer with camera fx and fy
    o3d.visualization.draw_geometries([source_init, source_temp, target_temp,
                                       axe,ground])

# ICP registration

reg_icp = o3d.pipelines.registration.registration_icp(
    source, target, threshold,
    np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

draw_registration_result(source, target, reg_icp.transformation)