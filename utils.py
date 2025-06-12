import rosbag
import math
import open3d as o3d

def read_rosbag(bag_file, topic_filter=None):
    messages = []
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=topic_filter):
            messages.append((topic, msg, t))
    return messages

def read_rosbag_topics(bag_file):

    topics = []
    with rosbag.Bag(bag_file, 'r') as bag:
        topics = bag.get_type_and_topic_info().topics.keys()
    return list(topics)

def rotate_point(point, angle, origin=(0, 0)):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy]


def get_nearest_timestamps(traj_time, lidar_time, image_time):
    nearest_lidar = []
    nearest_image = []

    for t in traj_time:
        nearest_lidar.append(min(lidar_time, key=lambda x: abs(x - t)))
        nearest_image.append(min(image_time, key=lambda x: abs(x - t)))

    return nearest_lidar, nearest_image


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
