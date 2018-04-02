# --------------------IMPORT-------------------

import io
from subprocess import call

import imageio
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
def video_data_creator(h_pose_list, b_pose_list, dists, fr_list, h_id_list, b_id_list):
    file_list = []
    for i in range(0, len(fr_list)):
        img = Image.open(io.BytesIO(fr_list[i]))
        h_pose = h_pose_list[h_id_list[i]]
        b_pose = b_pose_list[b_id_list[i]]
        draw = ImageDraw.Draw(img)
        textprint = "Dist: " + str(dists[i][1]) + "\n" + "drone:\n" + "   x:" + str(
            b_pose.x) + "\n" + "   y:" + str(b_pose.y) + "\n" + "   z:" + str(
            b_pose.z) + "\n" + "head:\n" + "   x:" + str(h_pose.x) + "\n" + "   y:" + str(
            h_pose.y) + "\n" + "   z:" + str(h_pose.z)
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


def video_plot_creator(h_pose_list, b_pose_list, fr_list, h_id_list, b_id_list, title):
    fig = plt.figure()
    file_list = []
    for i in range(0, len(fr_list)):
        print("img: ", i)
        plt.clf()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        h_pose = h_pose_list[h_id_list[i]]
        b_pose = b_pose_list[b_id_list[i]]

        img = Image.open(io.BytesIO(fr_list[i]))
        raw_frame = list(img.getdata())
        frame = []
        for b in raw_frame:
            frame.append(b)
        reshaped_fr = np.reshape(np.array(frame, dtype=np.int64), (480, 856, 3))
        ax1.imshow(reshaped_fr)
        plt.title("Frame: " + str(i))
        ax2.axis([-2.4, 2.4, -2.4, 2.4])

        spacing = 1.2
        minor_locator = MultipleLocator(spacing)

        # Set minor tick locations.
        ax2.yaxis.set_minor_locator(minor_locator)
        ax2.xaxis.set_minor_locator(minor_locator)
        # Set grid to use minor tick locations.
        ax2.grid(which='minor')
        #
        # plt.grid(True)
        ax2.plot(b_pose.x, b_pose.y, "ro", h_pose.x, h_pose.y, "go")
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


# tools
def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


def get_bag_data(bag_file):
    hat_poses = []
    hat_times = []
    for topic, hat, t in bag_file.read_messages(topics=['/optitrack/head']):
        secs = t.secs
        nsecs = t.nsecs
        hat_times.append(time_conversion_to_nano(secs, nsecs))
        # hat_times.append(time_conversion_to_nano(hat.header.stamp.secs, hat.header.stamp.nsecs))
        hat_poses.append(hat.pose.position)

    bebop_poses = []
    bebop_times = []
    for topic, bebop, t in bag_file.read_messages(topics=['/optitrack/bebop']):
        secs = t.secs
        nsecs = t.nsecs
        bebop_times.append(time_conversion_to_nano(secs, nsecs))
        # bebop_times.append(time_conversion_to_nano(bebop.header.stamp.secs, bebop.header.stamp.nsecs))
        bebop_poses.append(bebop.pose.position)

    frames = []
    camera_times = []
    for topic, image_frame, t in bag_file.read_messages(topics=['/bebop/image_raw/compressed']):
        secs = t.secs
        nsecs = t.nsecs
        frames.append(image_frame.data)
        camera_times.append(time_conversion_to_nano(secs, nsecs))
        # camera_times.append(time_conversion_to_nano(image_frame.header.stamp.secs, image_frame.header.stamp.nsecs))
    bag_file.close()
    bag_topics = bag_file.get_type_and_topic_info()[1].keys()

    # for idx, (topic, msg, mt) in enumerate(bag_file.read_messages(topics=bag_topics)):
    #    if(ts_from_header):
    #        msg.header.stamp.to_nsec()
    #    else:
    #        mt.to_nsec()

    return camera_times, frames, bebop_times, bebop_poses, hat_times, hat_poses


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


def plotter(h_pose_list, b_pose_list, fr_list, h_id_list, b_id_list):
    fig = plt.figure()
    for i in range(0, len(fr_list)):
        plt.clf()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        h_pose = h_pose_list[h_id_list[i]]
        b_pose = b_pose_list[b_id_list[i]]

        img = Image.open(io.BytesIO(fr_list[i]))
        raw_frame = list(img.getdata())
        frame = []
        for b in raw_frame:
            frame.append(b)
        reshaped_fr = np.reshape(np.array(frame, dtype=np.int64), (480, 856, 3))
        ax1.imshow(reshaped_fr)
        plt.title("Frame: " + str(i))
        ax2.axis([-2.4, 2.4, -2.4, 2.4])

        spacing = 1.2
        minor_locator = MultipleLocator(spacing)

        # Set minor tick locations.
        ax2.yaxis.set_minor_locator(minor_locator)
        ax2.xaxis.set_minor_locator(minor_locator)
        # Set grid to use minor tick locations.
        ax2.grid(which='minor')
        #
        # plt.grid(True)
        ax2.plot(b_pose.x, b_pose.y, "ro", h_pose.x, h_pose.y, "go")
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
    call(["vlc", "voice.mp3"])


# endregion
# -------------------Main area----------------------
def main():
    # open the bag file
    bag = rosbag.Bag('drone.bag')

    # info_dict = yaml.load(Bag('drone.bag', 'r')._get_yaml_info())

    # extract data from bag file
    camera_time_list, frame_list, bebop_time_list, bebop_pose_list, hat_time_list, hat_pose_list = get_bag_data(bag)

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
        head_pose = hat_pose_list[hat_idx_nearest[i]]

        hat_points[i][0] = head_pose.x
        hat_points[i][1] = head_pose.y
        hat_points[i][2] = head_pose.z

        bebop_pose = bebop_pose_list[bebop_idx_nearest[i]]

        bebop_points[i][0] = bebop_pose.x
        bebop_points[i][1] = bebop_pose.y
        bebop_points[i][2] = bebop_pose.z

        distances[i][0] = camera_np_array[i]
        distances[i][1] = distance.pdist([hat_points[i], bebop_points[i]], 'euclidean')

    # plotter(hat_pose_list, bebop_pose_list, frame_list, hat_idx_nearest, bebop_idx_nearest)
    
    video_plot_creator(hat_pose_list, bebop_pose_list, frame_list, hat_idx_nearest, bebop_idx_nearest,"main_plot")

    # plot_times(bebop_time_list,hat_time_list,camera_time_list)

    # far_frames_sel, far_dist_sel = get_distant_frame(distances, camera_np_array, frame_list, num=30)
    # video_creator(far_dist_sel, far_frames_sel, title='far')
    #
    # near_frames_sel, near_dist_sel = get_near_frame(distances, camera_np_array, frame_list, num=30)
    # video_creator(near_dist_sel, near_frames_sel, title='near')

    # py_voice("Lavoro completato", l='it')


if __name__ == "__main__":
    main()
