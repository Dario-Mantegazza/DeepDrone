# --------------------IMPORT-------------------
import io
import math
import numpy as np
import os
import random
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import MultipleLocator
from subprocess import call
from multiprocessing import Pool
import cv2
import pandas as pd
import rosbag
import tf
import tqdm as tqdm
from gtts import gTTS
from scipy.spatial import distance
from tf.transformations import (quaternion_conjugate, quaternion_multiply)
from transforms3d.derivations.quaternions import quat2mat

# ----------FUNCTIONS DEFINITIONS---------------
# region class video

bag_end_cut = {
    "1": 3150,
    "2": 7000,
    "3": 390,
    "4": 1850,
    "5": 3840,
    "6": 1650,
    "7": 2145,
    "8": 595,
    "9": 1065,
    "10": 2089,
    "11": 1370
}
bag_start_cut = {
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 58,
    "8": 63,
    "9": 75,
    "10": 50,
    "11": 0
}
bag_file_path = {
    "1": "./bagfiles/train/",
    "2": "./bagfiles/train/",
    "3": "./bagfiles/validation/",
    "4": "./bagfiles/validation/",
    "5": "./bagfiles/train/",
    "6": "./bagfiles/validation/",
    "7": "./bagfiles/train/",
    "8": "./bagfiles/train/",
    "9": "./bagfiles/train/",
    "10": "./bagfiles/validation/",
    "11": "./bagfiles/validation/"
}


class DatasetCreator:
    def __init__(self):
        self.dataset = []
        pass

    def generate_data(self, b_orientation, b_position, frame_list, h_orientation, h_position, f):
        self.b_orientation = b_orientation
        self.b_position = b_position
        self.frame_list = frame_list
        self.h_orientation = h_orientation
        self.h_position = h_position
        max_ = bag_end_cut[f[:-4]]
        min_ = bag_start_cut[f[:-4]]
        for i in tqdm.tqdm(range(min_, max_)):
            # for i in tqdm.tqdm(range(0, 10)):
            self.data_aggregator(i)

    def data_aggregator(self, i):
        img = Image.open(io.BytesIO(self.frame_list[i]))
        raw_frame = list(img.getdata())
        frame = []
        for b in raw_frame:
            frame.append(b)
        reshaped_fr = np.reshape(np.array(frame, dtype=np.int64), (480, 856, 3))
        reshaped_fr = reshaped_fr.astype(np.uint8)

        scaled_fr = cv2.resize(reshaped_fr, (107, 60))

        r_t_h = matrix_method(self.b_position[i], self.b_orientation[i], self.h_position[i], self.h_orientation[i])
        horizontal_angle = -math.degrees(math.atan2(r_t_h[1, 3], r_t_h[0, 3]))
        # vertical_angle = math.degrees(math.atan2(r_t_h[2, 3], r_t_h[0, 3]))
        label = int(horizontal_angle >= 0)
        self.dataset.append((scaled_fr, label))

    def save_dataset(self, flag_train):
        random.seed(42)
        # save
        if flag_train:
            shuffled_dataset = list(self.dataset)
            np.random.shuffle(shuffled_dataset)
            train = pd.DataFrame(shuffled_dataset)
            train.to_pickle("./dataset/train.pickle")
        else:
            shuffled_dataset = list(self.dataset)
            # no shuffling for validation
            # np.random.shuffle(shuffled_dataset)
            val = pd.DataFrame(shuffled_dataset)
            val.to_pickle("./dataset/validation.pickle")


class VideoCreator:
    def __init__(self, b_orientation, b_position, frame_list, h_orientation, h_position, f, title="test.avi"):
        self.fps = 30
        self.f = f
        self.video_writer = cv2.VideoWriter(title, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (640, 480))
        self.b_orientation = b_orientation
        self.b_position = b_position
        self.frame_list = frame_list
        self.h_orientation = h_orientation
        self.h_position = h_position

    def plotting_function(self, i):
        # TODO introduce pyqtgraph
        fig = plt.figure()
        axl = fig.add_subplot(1, 3, 1)
        axc = fig.add_subplot(1, 3, 2)
        axr = fig.add_subplot(1, 3, 3)
        canvas = FigureCanvas(fig)
        plt.title("Frame: " + str(i))

        # Central IMAGE
        img = Image.open(io.BytesIO(self.frame_list[i]))
        raw_frame = list(img.getdata())
        frame = []
        for b in raw_frame:
            frame.append(b)
        reshaped_fr = np.reshape(np.array(frame, dtype=np.int64), (480, 856, 3))
        reshaped_fr = reshaped_fr.astype(np.uint8)
        axc.imshow(reshaped_fr)
        axc.set_axis_off()

        # RIGHT PLOT

        axr.axis([-2.4, 2.4, -2.4, 2.4], 'equals')
        h_theta = quat_to_eul(self.h_orientation[i])[2]
        b_theta = quat_to_eul(self.b_orientation[i])[2]
        arrow_length = 0.3
        spacing = 1.2
        minor_locator = MultipleLocator(spacing)
        # Set minor tick locations.
        axr.yaxis.set_minor_locator(minor_locator)
        axr.xaxis.set_minor_locator(minor_locator)
        # Set grid to use minor tick locations.
        axr.grid(which='minor')
        # plt.grid(True)
        axr.plot(self.b_position[i].x, self.b_position[i].y, "ro", self.h_position[i].x, self.h_position[i].y, "go")
        axr.arrow(self.h_position[i].x, self.h_position[i].y, arrow_length * np.cos(h_theta), arrow_length * np.sin(h_theta), head_width=0.05, head_length=0.1, fc='g', ec='g')
        axr.arrow(self.b_position[i].x, self.b_position[i].y, arrow_length * np.cos(b_theta), arrow_length * np.sin(b_theta), head_width=0.05, head_length=0.1, fc='r', ec='r')

        # LEFT PLOT
        # transform from head to world to drone then compute atan2

        # p, q = jerome_method(self.b_position[i], self.b_orientation[i], self.h_position[i], self.h_orientation[i])
        # horizontal_angle = math.degrees(math.atan2(p[1], p[0]))
        # vertical_angle = math.degrees(math.atan2(p[2], p[0]))

        r_t_h = matrix_method(self.b_position[i], self.b_orientation[i], self.h_position[i], self.h_orientation[i])
        horizontal_angle = -math.degrees(math.atan2(r_t_h[1, 3], r_t_h[0, 3]))
        vertical_angle = math.degrees(math.atan2(r_t_h[2, 3], r_t_h[0, 3]))
        value_angle_axis = 45
        axl.set_xbound(-value_angle_axis, value_angle_axis)
        axl.set_ybound(-value_angle_axis, value_angle_axis)
        axl.axis([-value_angle_axis, value_angle_axis, -value_angle_axis, value_angle_axis], 'equal')
        axl.plot(horizontal_angle, vertical_angle, "go")

        # Drawing the plot
        canvas.draw()

        width, height = (fig.get_size_inches() * fig.get_dpi()).astype(dtype='int32')
        img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pt1 = (275, 25)
        pt2 = (375, 25)
        if horizontal_angle >= 0:
            cv2.arrowedLine(img, pt1, pt2, (0, 0, 255), 3)
        else:
            cv2.arrowedLine(img, pt2, pt1, (0, 255, 0), 3)

        self.video_writer.write(img)
        plt.close(fig)

    def video_plot_creator(self):
        # max_ = bag_file_cut[self.f[:-4]]
        max_ = len(self.frame_list)

        # for i in tqdm.tqdm(range(0, 100)):
        # for i in tqdm.tqdm(range(300, 700)):
        for i in tqdm.tqdm(range(0, max_)):
            self.plotting_function(i)
        self.video_writer.release()
        cv2.destroyAllWindows()


# def jerome_method(p_b, q_b, p_h, q_h):  # relative pose of h wrt b
#     np_q_b = quat_to_array(q_b)
#     np_p_b = np.array([p_b.x, p_b.y, p_b.z])
#
#     np_q_h = quat_to_array(q_h)
#     np_p_h = np.array([p_h.x, p_h.y, p_h.z])
#
#     cq_b = quaternion_conjugate(np_q_b)
#     p = np.concatenate([np_p_h - np_p_b, [0]])
#     p = quaternion_multiply(np_q_b, quaternion_multiply(p, cq_b))[:3]
#     q = quaternion_multiply(cq_b, np_q_h)
#     return p, q


def rospose2homogmat(p, q):
    w_r_o = np.array(quat2mat(quat_to_array(q))).astype(np.float64)  # rotation matrix of object wrt world frame
    np_pose = np.array([[p.x], [p.y], [p.z]])
    tempmat_b = np.hstack((w_r_o, np_pose))
    w_r_o = np.vstack((tempmat_b, [0, 0, 0, 1]))
    return w_r_o


def matrix_method(p_b, q_b, p_h, q_h):
    w_t_b = rospose2homogmat(p_b, q_b)
    w_t_h = rospose2homogmat(p_h, q_h)
    inv_wtb = np.linalg.inv(w_t_b)
    b_t_h = np.matmul(inv_wtb, w_t_h)
    return b_t_h


# endregion

# region altro
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


# export ROS_MASTER_URI=http://192.168.201.4:11311

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
        hat_orientaions.append(hat.pose.orientation)

    bebop_positions = []
    bebop_orientaions = []
    bebop_times = []
    for topic, bebop, t in bag_file.read_messages(topics=['/optitrack/bebop']):
        secs = t.secs
        nsecs = t.nsecs
        bebop_times.append(time_conversion_to_nano(secs, nsecs))
        bebop_positions.append(bebop.pose.position)
        bebop_orientaions.append(bebop.pose.orientation)

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


def quat_to_array(q):
    ret = np.zeros((4))
    ret[0] = q.w
    ret[1] = q.x
    ret[2] = q.y
    ret[3] = q.z
    return ret


def data_pre_processing(bag):
    camera_time_list, frames_list, bebop_time_list, bebop_position_list, hat_time_list, hat_position_list, hat_orientation_list, bebop_orientation_list = get_bag_data(bag)
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
    h_sel_positions = []
    b_sel_positions = []
    h_sel_orientations = []
    b_sel_orientations = []
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

        h_sel_positions.append(head_position)
        b_sel_positions.append(bebop_position)
        h_sel_orientations.append(hat_orientation_list[hat_idx_nearest[i]])
        b_sel_orientations.append(bebop_orientation_list[bebop_idx_nearest[i]])
    return b_sel_orientations, b_sel_positions, frames_list, h_sel_orientations, h_sel_positions


def bag_tovid(f):
    path = bag_file_path[f[:-4]]
    print("\nreading bag: " + str(f))
    bag = rosbag.Bag(path + f)
    b_sel_orientations, b_sel_positions, frames_list, h_sel_orientations, h_sel_positions = data_pre_processing(bag)

    vidcr = VideoCreator(b_orientation=b_sel_orientations, b_position=b_sel_positions, frame_list=frames_list, h_orientation=h_sel_orientations, h_position=h_sel_positions, f=f,
                         title="./video/" + f[:-4] + ".avi")
    vidcr.video_plot_creator()
    print("\nvideo : " + str(f[:-4] + " completed"))


# endregion
# -------------------Main area----------------------
def main():
    scelta = raw_input("Video o dataset:[v/d]")
    if scelta == "v":
        path1 = "./bagfiles/train/"
        path2 = "./bagfiles/validation/"

        files1 = [f for f in os.listdir(path1) if f[-4:] == '.bag']
        if not files1:
            print('No bag files found!')
            return None
        files2 = [f for f in os.listdir(path2) if f[-4:] == '.bag']
        if not files2:
            print('No bag files found!')
            return None
        files = []
        # comporre files
        for f_ in files1:
            files.append(f_)
        for f_ in files2:
            files.append(f_)

        pool = Pool(processes=6)
        pool.map(bag_tovid, files)
        pool.close()
        pool.join()

        py_voice("Video Creato!", l='it')
    else:
        # train
        datacr_train = DatasetCreator()
        path = "./bagfiles/train/"
        files = [f for f in os.listdir(path) if f[-4:] == '.bag']
        if not files:
            print('No bag files found!')
            return None

        for f in files:
            bag = rosbag.Bag(path + f)
            b_sel_orientations, b_sel_positions, frames_list, h_sel_orientations, h_sel_positions = data_pre_processing(bag)
            datacr_train.generate_data(b_orientation=b_sel_orientations, b_position=b_sel_positions, frame_list=frames_list, h_orientation=h_sel_orientations, h_position=h_sel_positions, f=f)
        datacr_train.save_dataset(flag_train=True)

        # validation
        datacr_val = DatasetCreator()
        path = "./bagfiles/validation/"
        files = [f for f in os.listdir(path) if f[-4:] == '.bag']
        if not files:
            print('No bag files found!')
            return None

        for f in files:
            bag = rosbag.Bag(path + f)
            b_sel_orientations, b_sel_positions, frames_list, h_sel_orientations, h_sel_positions = data_pre_processing(bag)
            datacr_val.generate_data(b_orientation=b_sel_orientations, b_position=b_sel_positions, frame_list=frames_list, h_orientation=h_sel_orientations, h_position=h_sel_positions, f=f)
        datacr_val.save_dataset(flag_train=False)

        py_voice("Dataset creato!", l='it')


if __name__ == "__main__":
    main()
