import io

import imageio
import rosbag
from PIL import Image
from PIL import ImageDraw


def missing_numbers(complete_list, missing_list):  # complete list must be ordered and complete. maybe not what I need
    original_list = [x for x in range(complete_list[0], complete_list[-1] + 1)]
    num_list = set(missing_list)
    return list(num_list ^ set(original_list))


bag = rosbag.Bag('drone.bag')
topics = bag.get_type_and_topic_info()[1].keys()


def nanosec_sec_to_milli(sec, nano):
    return (sec * 1000) + ((nano / 1000) / 1000)


saveGifPIL = lambda filename, images, **mimsaveParams: imageio.mimsave(filename, [
    [(img.save(buf, format='png'), buf.seek(0), imageio.imread(buf))[2] for buf in [io.BytesIO()]][0] for img in
    images], **mimsaveParams)

bebop_time_list = []
# bebop_pose_list = []
for topic, bebop, t in bag.read_messages(topics=['/optitrack/bebop']):
    # print(bebop.header.stamp.secs)
    bebop_time_list.append(nanosec_sec_to_milli(bebop.header.stamp.secs, bebop.header.stamp.nsecs))
    # bebop_pose_list.append(bebop.pose.position)

# bebop_pose_list = []

frames_list = []
camera_time_list = []
for topic, image_frame, t in bag.read_messages(topics=['/bebop/image_raw/compressed']):
    frames_list.append(image_frame.data)
    camera_time_list.append(nanosec_sec_to_milli(image_frame.header.stamp.secs, image_frame.header.stamp.nsecs))

    # Message filter time sycnhronizer or approximate time  wiki.ros.org/mesage_filters

bag.close()

# print(len(frames_list))
images_list = []
for frame in frames_list:
    img = Image.open(io.BytesIO(frame))
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    # font = ImageFont.truetype("sans-serif.ttf", 16)
    # # draw.text((x, y),"Sample Text",(r,g,b))
    # draw.text((0, 0), "Sample Text", (255, 255, 255), font=font)
    draw.text((0, 0), "Sample Text", (255, 255, 255))
    images_list.append(Image.open(io.BytesIO(frame)))
    # img.show()

# saveGifPIL("video.gif", images_list, fps=30)

# from PIL import Image
# from PIL import ImageFont
# from PIL import ImageDraw
#
# img = Image.open("sample_in.jpg")
# draw = ImageDraw.Draw(img)
# # font = ImageFont.truetype(<font-file>, <font-size>)
# font = ImageFont.truetype("sans-serif.ttf", 16)
# # draw.text((x, y),"Sample Text",(r,g,b))
# draw.text((0, 0),"Sample Text",(255,255,255),font=font)
# img.save('sample-out.jpg')
