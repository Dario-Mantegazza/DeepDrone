# import io
# import matplotlib as mpl
import rosbag
import numpy as np
from matplotlib import pyplot as plt
# from PIL import Image
#
# raw image info
# height: 480
# width: 856
# encoding: "rgb8"
# is_bigendian: 0


# mpl.use('TkAgg')
# import imageio


def missing_numbers(complete_list, missing_list):  # complete list must be ordered and complete. maybe not what I need
    original_list = [x for x in range(complete_list[0], complete_list[-1] + 1)]
    num_list = set(missing_list)
    return list(num_list ^ set(original_list))


def nanosec_sec_to_milli(sec, nano):
    return (sec * 1000) + ((nano / 1000) / 1000)


bag = rosbag.Bag('drone.bag')
# topics = bag.get_type_and_topic_info()[1].keys()

# saveGifPIL = lambda filename, images, **mimsaveParams: imageio.mimsave(filename, [
#     [(img.save(buf, format='png'), buf.seek(0), imageio.imread(buf))[2] for buf in [io.BytesIO()]][0] for img in
#     images], **mimsaveParams)

# head_seq_list = []
# head_pose_list = []
# for topic, head_hat, t in bag.read_messages(topics=['/optitrack/head']):
#     head_seq_list.append(head_hat.header.seq)
#     head_pose_list.append(head_hat.pose.position)
#
bebop_time_list = []
# bebop_pose_list = []
for topic, bebop, t in bag.read_messages(topics=['/optitrack/bebop']):
    # print(bebop.header.stamp.secs)
    bebop_time_list.append(nanosec_sec_to_milli(bebop.header.stamp.secs, bebop.header.stamp.nsecs))
    # bebop_pose_list.append(bebop.pose.position)

# bebop_pose_list = []
print('extraction started')
frames_list = []
camera_time_list = []
count = 0
for topic, raw_frame, t in bag.read_messages(topics=['/bebop/image_raw']):
    count += 1
    print('extracting image ', count)
    if count > 500:
        continue
    frame = []
    for b in raw_frame.data:
        frame.append(b)
    plt.clf()

    plt.imshow(np.reshape(frame, (480, 856, 3)))
    plt.show(block=False)
    plt.pause(0.001)
    camera_time_list.append(nanosec_sec_to_milli(raw_frame.header.stamp.secs, raw_frame.header.stamp.nsecs))

bag.close()
plt.show()
