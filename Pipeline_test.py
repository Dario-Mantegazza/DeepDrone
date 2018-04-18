# --------------------IMPORT-------------------

import io
import math
from subprocess import call

import imageio
import tf
import numpy as np
import rosbag
from PIL import Image
from PIL import ImageDraw
from gtts import gTTS
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.spatial import distance


# ----------FUNCTIONS DEFINITIONS---------------
# region Def

# Video Creation
def video_data_creator(h_position_list, b_position_list, dists, fr_list, h_id_list, b_id_list):
    file_list = []
    for i in range(0, len(fr_list)):
        img = Image.open(io.BytesIO(fr_list[i]))
        h_position = h_position_list[h_id_list[i]]
        b_position = b_position_list[b_id_list[i]]
        draw = ImageDraw.Draw(img)
        textprint = "Dist: " + str(dists[i][1]) + "\n" + "drone:\n" + "   x:" + str(
            b_position.x) + "\n" + "   y:" + str(b_position.y) + "\n" + "   z:" + str(
            b_position.z) + "\n" + "head:\n" + "   x:" + str(h_position.x) + "\n" + "   y:" + str(
            h_position.y) + "\n" + "   z:" + str(h_position.z)
        draw.text((5, 5), textprint, (255, 255, 255))
        png_path = "images/frame" + str(i) + ".png"
        img.save(png_path, "PNG")
        file_list.append(png_path)

    writer = imageio.get_writer('test.mp4', fps=30)
    for im in file_list:
        writer.append_data(imageio.imread(im))
    writer.close()


def video_creator(dists, fr_list, title='selected'):
    file_list = []
    for i in range(0, len(fr_list)):
        # for i in range(0, 500):
        img = Image.open(io.BytesIO(fr_list[i]))
        draw = ImageDraw.Draw(img)
        textprint = "Dist: " + str(dists[i][1])
        draw.text((5, 5), textprint, (255, 255, 255))
        png_path = "images/" + title + "_" + str(i) + ".png"
        img.save(png_path, "PNG")
        file_list.append(png_path)

    writer = imageio.get_writer(title + '.mp4', fps=2)
    for im in file_list:
        writer.append_data(imageio.imread(im))
    writer.close()


def video_plot_creator(h_position_list, b_position_list, fr_list, h_id_list, b_id_list, h_or_list, b_or_list, title):
    fig = plt.figure()
    file_list = []
    for i in range(0, len(fr_list)):
        # for i in range(300, 600):
        print("img: " + str(i))
        plt.clf()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        h_position = h_position_list[h_id_list[i]]
        b_position = b_position_list[b_id_list[i]]
        h_orientation = h_or_list[h_id_list[i]]
        b_orientation = b_or_list[b_id_list[i]]

        img = Image.open(io.BytesIO(fr_list[i]))
        raw_frame = list(img.getdata())
        frame = []
        for b in raw_frame:
            frame.append(b)
        reshaped_fr = np.reshape(np.array(frame, dtype=np.int64), (480, 856, 3))
        reshaped_fr = reshaped_fr.astype(np.uint8)

        ax1.imshow(reshaped_fr)
        plt.title("Frame: " + str(i))
        ax2.axis([-2.4, 2.4, -2.4, 2.4])

        # theta = h_orientation['yaw']
        h_theta = h_orientation[2]
        b_theta = b_orientation[2]
        arrow_length = 0.3
        spacing = 1.2
        minor_locator = MultipleLocator(spacing)

        # Set minor tick locations.
        ax2.yaxis.set_minor_locator(minor_locator)
        ax2.xaxis.set_minor_locator(minor_locator)
        # Set grid to use minor tick locations.
        ax2.grid(which='minor')
        #
        # plt.grid(True)
        ax2.plot(b_position.x, b_position.y, "ro", h_position.x, h_position.y, "go")
        ax2.arrow(h_position.x, h_position.y, arrow_length * np.cos(h_theta), arrow_length * np.sin(h_theta), head_width=0.05, head_length=0.1, fc='g', ec='g')
        ax2.arrow(b_position.x, b_position.y, arrow_length * np.cos(b_theta), arrow_length * np.sin(b_theta), head_width=0.05, head_length=0.1, fc='r', ec='r')

        png_path = "images/" + title + "_" + str(i) + ".png"
        plt.savefig(png_path)
        file_list.append(png_path)

    print("video creation")
    writer = imageio.get_writer(title + '.mp4', fps=30)
    for im in file_list:
        writer.append_data(imageio.imread(im))
    writer.close()


# Conversions
def time_conversion_to_nano(sec, nano):
    return (sec * 1000 * 1000 * 1000) + nano


def pil_to_pyplot(fr_list):
    reshaped_list = []
    for i in range(0, len(fr_list)):
        print("im: ", i)
        img = Image.open(io.BytesIO(fr_list[i]))
        raw_frame = list(img.getdata())
        frame = []
        for b in raw_frame:
            frame.append(b)

        reshaped_list.append(np.reshape(np.array(frame, dtype=np.int64), (480, 856, 3)))

    return reshaped_list


def quat_to_eul(orientation):
    quaternion = (
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)  # roll 0, pitch 1, yaw 2
    return euler


# tools
def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


def get_bag_data(bag_file):
    hat_positions = []
    hat_orientaions = []
    hat_times = []
    # if canTransform('/bebop/odom', source_frame, time):
    #     print "yay"
    for topic, hat, t in bag_file.read_messages(topics=['/optitrack/head']):
        secs = t.secs
        nsecs = t.nsecs
        hat_times.append(time_conversion_to_nano(secs, nsecs))

        hat_positions.append(hat.pose.position)
        hat_orientaions.append(quat_to_eul(hat.pose.orientation))

    bebop_positions = []
    bebop_orientaions = []
    bebop_times = []
    for topic, bebop, t in bag_file.read_messages(topics=['/optitrack/bebop']):
        secs = t.secs
        nsecs = t.nsecs
        bebop_times.append(time_conversion_to_nano(secs, nsecs))
        bebop_positions.append(bebop.pose.position)
        bebop_orientaions.append(quat_to_eul(bebop.pose.orientation))

    frames = []
    camera_times = []
    for topic, image_frame, t in bag_file.read_messages(topics=['/bebop/image_raw/compressed']):
        secs = t.secs
        nsecs = t.nsecs
        frames.append(image_frame.data)
        camera_times.append(time_conversion_to_nano(secs, nsecs))

    bag_file.close()
    # bag_topics = bag_file.get_type_and_topic_info()[1].keys()
    return camera_times, frames, bebop_times, bebop_positions, hat_times, hat_positions, hat_orientaions, bebop_orientaions


def get_distant_frame(dists, camera_times, frames, num=5):
    sorted_distances = dists[dists[:, 1].argsort()]
    frames_selected = []
    for i in range(len(dists) - num, len(dists)):
        frames_selected.append(frames[np.where(camera_times == sorted_distances[i][0])[0][0]])
    return frames_selected, sorted_distances[-num:]


def get_near_frame(dists, camera_times, frames, num=5):
    sorted_distances = dists[dists[:, 1].argsort()]
    frames_selected = []
    for i in range(0, num):
        frames_selected.append(frames[np.where(camera_times == sorted_distances[i][0])[0][0]])
    return frames_selected, sorted_distances[0:num]


def plotter(h_position_list, b_position_list, fr_list, h_id_list, b_id_list, h_or_list, b_or_list):
    fig = plt.figure()
    for i in range(0, len(fr_list)):
        plt.clf()
        plt.title("Frame: " + str(i))
        axl = fig.add_subplot(1, 3, 1)
        axc = fig.add_subplot(1, 3, 2)
        axr = fig.add_subplot(1, 3, 3)

        # Central IMAGE
        h_position = h_position_list[h_id_list[i]]
        b_position = b_position_list[b_id_list[i]]
        h_orientation = h_or_list[h_id_list[i]]
        b_orientation = b_or_list[b_id_list[i]]

        img = Image.open(io.BytesIO(fr_list[i]))
        raw_frame = list(img.getdata())
        frame = []
        for b in raw_frame:
            frame.append(b)
        reshaped_fr = np.reshape(np.array(frame, dtype=np.int64), (480, 856, 3))
        reshaped_fr = reshaped_fr.astype(np.uint8)
        axc.imshow(reshaped_fr)

        # RIGHT PLOT
        axr.axis([-2.4, 2.4, -2.4, 2.4])

        h_theta = h_orientation[2]
        b_theta = b_orientation[2]
        arrow_length = 0.3
        spacing = 1.2
        minor_locator = MultipleLocator(spacing)

        # Set minor tick locations.
        axr.yaxis.set_minor_locator(minor_locator)
        axr.xaxis.set_minor_locator(minor_locator)
        # Set grid to use minor tick locations.
        axr.grid(which='minor')

        # plt.grid(True)
        axr.plot(b_position.x, b_position.y, "ro", h_position.x, h_position.y, "go")
        axr.arrow(h_position.x, h_position.y, arrow_length * np.cos(h_theta), arrow_length * np.sin(h_theta), head_width=0.05, head_length=0.1, fc='g', ec='g')
        axr.arrow(b_position.x, b_position.y, arrow_length * np.cos(b_theta), arrow_length * np.sin(b_theta), head_width=0.05, head_length=0.1, fc='r', ec='r')

        # LEFT PLOT
        # transform from head to world to drone then compute atan2

        horizontal_angle = math.atan2(h_position.y, h_position.x)
        vertical_angle = math.atan2(h_position.z, h_position.x)
        axl.plot(horizontal_angle, vertical_angle, "go")

        # general plot stuff
        plt.show(block=False)
        plt.pause(0.01)


def plot_times(b_times, h_times, c_times):
    x1 = np.full((len(b_times)), 1)
    x2 = np.full((len(h_times)), 2)
    x3 = np.full((len(c_times)), 3)
    plt.plot(b_times, x1, "r.", h_times, x2, "g.", c_times, x3, "b.", markersize=1)
    plt.show()


def py_voice(text_to_speak="Computing Completed", l='en'):
    tts = gTTS(text=text_to_speak, lang=l)
    tts.save('voice.mp3')
    # os.system('/voice.mp3')
    call(["cvlc", "voice.mp3", '--play-and-exit'])


# endregion
# -------------------Main area----------------------
def main():
    # open the bag file
    bag = rosbag.Bag('drone.bag')

    # info_dict = yaml.load(Bag('drone.bag', 'r')._get_yaml_info())

    # extract data from bag file
    camera_time_list, frame_list, bebop_time_list, bebop_position_list, hat_time_list, hat_position_list, hat_orientation_list, bebop_orientation_list = get_bag_data(bag)

    # reformat some data as np array for future use
    camera_np_array = np.asarray(camera_time_list)
    bebop_np_array = np.asarray(bebop_time_list)
    hat_np_array = np.asarray(hat_time_list)

    # identify the nearest time frames of the bebop with respect of the camera data
    bebop_idx_nearest = []
    for v in camera_np_array:
        bebop_idx_nearest.append(find_nearest(bebop_np_array, v))

    # identify the nearest time frames of the hat with respect of the camera data
    hat_idx_nearest = []
    for v in camera_np_array:
        hat_idx_nearest.append(find_nearest(hat_np_array, v))

    # some variable inits
    distances = np.zeros((len(camera_np_array), 2))
    vect_structure = (len(camera_np_array), 3)
    hat_points = np.zeros(vect_structure)
    bebop_points = np.zeros(vect_structure)

    # computing the distances array/matrix
    for i in range(0, len(camera_np_array)):
        head_position = hat_position_list[hat_idx_nearest[i]]

        hat_points[i][0] = head_position.x
        hat_points[i][1] = head_position.y
        hat_points[i][2] = head_position.z

        bebop_position = bebop_position_list[bebop_idx_nearest[i]]

        bebop_points[i][0] = bebop_position.x
        bebop_points[i][1] = bebop_position.y
        bebop_points[i][2] = bebop_position.z

        distances[i][0] = camera_np_array[i]
        distances[i][1] = distance.pdist([hat_points[i], bebop_points[i]], 'euclidean')

    plotter(hat_position_list, bebop_position_list, frame_list, hat_idx_nearest, bebop_idx_nearest,hat_orientation_list, bebop_orientation_list)

    # video_plot_creator(hat_position_list, bebop_position_list, frame_list, hat_idx_nearest, bebop_idx_nearest, hat_orientation_list, bebop_orientation_list,"main_plot")

    # plot_times(bebop_time_list,hat_time_list,camera_time_list)

    # far_frames_sel, far_dist_sel = get_distant_frame(distances, camera_np_array, frame_list, num=30)
    # video_creator(far_dist_sel, far_frames_sel, title='far')
    #
    # near_frames_sel, near_dist_sel = get_near_frame(distances, camera_np_array, frame_list, num=30)
    # video_creator(near_dist_sel, near_frames_sel, title='near')

    py_voice("lavoro completato", l='it')


if __name__ == "__main__":
    main()
