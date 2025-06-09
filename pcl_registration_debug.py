#%% read ro
import cv2
import numpy as np
import open3d as o3d
import ros_numpy
import copy

source = o3d.io.read_point_cloud('./Data/pointclouds/0410.pcd')
target = o3d.io.read_point_cloud('./Data/pointclouds/0415.pcd')

def create_xy_grid(cell_size=10, num_cells=10, origin=(0, 0,0)):
    points = []
    lines = []
    x0, y0, z0 = origin
    # Create grid points
    for i in range(num_cells + 1):
        for j in range(num_cells + 1):
            points.append([x0 + i * cell_size, y0 + j * cell_size, z0])
    # Connect horizontal lines
    for j in range(num_cells + 1):
        for i in range(num_cells):
            idx = j * (num_cells + 1) + i
            lines.append([idx, idx + 1])
    # Connect vertical lines
    for i in range(num_cells + 1):
        for j in range(num_cells):
            idx = j * (num_cells + 1) + i
            lines.append([idx, idx + (num_cells + 1)])
    # Optional: color all lines gray
    colors = [[0.5, 0.5, 0.5] for _ in lines]
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector(colors)
    return grid


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