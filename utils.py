import rosbag
import math

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