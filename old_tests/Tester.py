import rosbag
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np


def missing_numbers(complete_list,missing_list):
    original_list = [x for x in range(complete_list[0], complete_list[-1] + 1)]
    num_list = set(missing_list)
    return list(num_list ^ set(original_list))


bag = rosbag.Bag('drone.bag')
topics = bag.get_type_and_topic_info()[1].keys()

head_seq_list = []
head_pose_list = []
for topic, head_hat, t in bag.read_messages(topics=['/optitrack/head']):
    head_seq_list.append(head_hat.header.seq)
    head_pose_list.append(head_hat.pose.position)

bebop_seq_list = []
bebop_pose_list = []
for topic, bebop, t in bag.read_messages(topics=['/optitrack/bebop']):
    bebop_seq_list.append(bebop.header.seq)
    bebop_pose_list.append(bebop.pose.position)

bag.close()

# print(len(head_pose_list))
dist_list = []
missing_id = missing_numbers(bebop_seq_list,head_seq_list)
print(missing_id)
old_pose = [0, 0]
s = (len(bebop_pose_list), 3)
head_points = np.zeros(s)
bebop_points = np.zeros(s)
distances = np.zeros((len(bebop_pose_list),2))
for i in range(0, len(bebop_seq_list)):
    if (bebop_seq_list[i]) in missing_id:
        head_pose = old_pose
    else:
        head_pose = head_pose_list[head_seq_list.index(bebop_seq_list[i])]  # lista delle pose id
        old_pose = head_pose
    head_points[i][0] = head_pose.x
    head_points[i][1] = head_pose.y
    head_points[i][2] = head_pose.z

    bebop_pose = bebop_pose_list[i]

    bebop_points[i][0] = bebop_pose.x
    bebop_points[i][1] = bebop_pose.y
    bebop_points[i][2] = bebop_pose.z

    distances[i][0] = bebop_seq_list[i]
    distances[i][1] = distance.pdist([head_points[i], bebop_points[i]], 'euclidean')

# plt.plot(range(0, len(head_list)), head_list)
plt.plot(range(0, len(bebop_pose_list)), distances[:,1])
plt.show()
