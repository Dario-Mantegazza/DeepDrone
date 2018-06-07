import os

import pandas as pd

import rosbag

# method to convert time

def time_conversion_to_nano(sec, nano):
    return (sec * 1000 * 1000 * 1000) + nano


def get_bag_data_pandas(bag_file):
    bag = rosbag.Bag(bag_file)
    col_names = ['position', 'orientations', 'timestamp']
    hat_df = pd.DataFrame(columns=col_names)
    i = 0
    for topic, hat, t in bag.read_messages(topics=['/optitrack/head']):
        secs = t.secs
        nsecs = t.nsecs
        hat_df.loc[i] = [hat.pose.position, hat.pose.orientation, time_conversion_to_nano(secs, nsecs)]
        i += 1
    bebop_df = pd.DataFrame(columns=col_names)
    i = 0
    for topic, bebop, t in bag.read_messages(topics=['/optitrack/bebop']):
        secs = t.secs
        nsecs = t.nsecs
        # bebop_times.append(time_conversion_to_nano(secs, nsecs))
        # bebop_positions.append(bebop.pose.position)
        # bebop_orientaions.append(bebop.pose.orientation)
        hat_df.loc[i] = [bebop.pose.position, bebop.pose.orientation, time_conversion_to_nano(secs, nsecs)]
        i += 1

    col_names = ['frame', 'timestamp']
    camera_df = pd.DataFrame(columns=col_names)
    i = 0
    for topic, image_frame, t in bag.read_messages(topics=['/bebop/image_raw/compressed']):
        secs = t.secs
        nsecs = t.nsecs
        camera_df.loc[i] = [image_frame.data, time_conversion_to_nano(secs, nsecs)]
        # frames.append(image_frame.data)
        # camera_times.append(time_conversion_to_nano(secs, nsecs))
        i += 1

    bag.close()


path = "./bagfiles/train/"
files = [f for f in os.listdir(path) if f[-4:] == '.bag']

for f in files:
    get_bag_data_pandas(path + f)
