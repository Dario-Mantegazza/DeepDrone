# ------ Import ------
import io
import math
import os
import random
from PIL import Image
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import rosbag
import tf
import tqdm as tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import MultipleLocator
from scipy.spatial import distance
from transforms3d.derivations.quaternions import quat2mat

# ------ Global Dictionaries ------

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
    "11": 1370,
    "12": 5600,
    "13": 8490,
    "14": 4450,
    "15": 7145,
    "16": 3500,
    "17": 1400,
    "18": 1300,
    "19": 1728,
    "20": 5070,
    "21": 11960,
    "22": 5200
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
    "11": 0,
    "12": 470,
    "13": 40,
    "14": 50,
    "15": 0,
    "16": 0,
    "17": 0,
    "18": 0,
    "19": 0,
    "20": 220,
    "21": 0,
    "22": 222
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
    "10": "./bagfiles/train/",
    "11": "./bagfiles/train/",
    "12": "./bagfiles/train/",
    "13": "./bagfiles/train/",
    "14": "./bagfiles/train/",
    "15": "./bagfiles/validation/",
    "16": "./bagfiles/train/",
    "17": "./bagfiles/train/",
    "18": "./bagfiles/train/",
    "19": "./bagfiles/train/",
    "20": "./bagfiles/train/",
    "21": "./bagfiles/train/",
    "22": "./bagfiles/validation/"
}


# ------ Classes ------

# This class handles the dataset creation.
class DatasetCreator:
    def __init__(self):
        self.dataset = []

    # Main method of the class, cycles through different nparrays and call other methods to compose the dataset.
    def generate_data(self, distances, b_orientation, b_position, frame_list, h_orientation, h_position, delta_z, f):
        self.b_orientation = b_orientation
        self.b_position = b_position
        self.frame_list = frame_list
        self.h_orientation = h_orientation
        self.h_position = h_position
        self.distances = distances
        self.delta_z = delta_z
        max_ = bag_end_cut[f[:-4]]
        min_ = bag_start_cut[f[:-4]]
        for i in tqdm.tqdm(range(min_, max_)):
            # for i in tqdm.tqdm(range(100, 400)):
            self.data_aggregator(i)

    # append a frame with labels into the dataset file. Using a flag it is possible to select the labels to associate with the camera frame
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
        label = (self.distances[i], horizontal_angle, self.delta_z[i])

        self.dataset.append((scaled_fr, label))

    # saves the dataset pickle file.
    def save_dataset(self, flag_train, title="wrong.pickle"):
        random.seed(42)
        # save
        if flag_train == "train":
            shuffled_dataset = list(self.dataset)
            # np.random.shuffle(shuffled_dataset)
            train = pd.DataFrame(shuffled_dataset)
            train.to_pickle("./dataset/train.pickle")
        elif flag_train == "validation":
            shuffled_dataset = list(self.dataset)
            # no shuffling for validation
            val = pd.DataFrame(shuffled_dataset)
            val.to_pickle("./dataset/validation.pickle")
        elif flag_train == "cross":
            val = pd.DataFrame(list(self.dataset))
            val.to_pickle("./dataset/crossvalidation/" + title)
        else:
            print("ERROR in FLAG TRAIN")
            return None


# Class used for creating video to analyze new data from bag files.
class VideoCreator:
    def __init__(self, b_orientation, distances, b_position, frame_list, h_orientation, h_position, delta_z, f, title="test.avi"):
        self.fps = 30
        self.f = f
        self.video_writer = cv2.VideoWriter(title, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (640, 480))
        self.b_orientation = b_orientation
        self.b_position = b_position
        self.frame_list = frame_list
        self.h_orientation = h_orientation
        self.h_position = h_position
        self.distances = distances
        self.delta_z = delta_z

    # given an index compose the frame for the video. Each frame has two graphs and the camera frame associated.
    def plotting_function(self, i):
        fig = plt.figure()
        fig.suptitle("Frame: " + str(i), fontsize=12)
        axll = fig.add_subplot(2, 2, 1)
        axl = fig.add_subplot(2, 2, 2)
        axc = fig.add_subplot(2, 2, 3)
        axr = fig.add_subplot(2, 2, 4)
        canvas = FigureCanvas(fig)

        # Central image: here we add the camera feed to the video
        img = Image.open(io.BytesIO(self.frame_list[i]))
        raw_frame = list(img.getdata())
        frame = []
        for b in raw_frame:
            frame.append(b)
        reshaped_fr = np.reshape(np.array(frame, dtype=np.int64), (480, 856, 3))
        reshaped_fr = reshaped_fr.astype(np.uint8)
        axc.imshow(reshaped_fr)
        axc.set_axis_off()

        # RIGHT PLOT: here we create the right plot that represent the position and heading of the bebop and head
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

        # LEFT PLOT: here we represent the distance on the y axis and the heading correction for the drone in degrees on the x-axis
        r_t_h = matrix_method(self.b_position[i], self.b_orientation[i], self.h_position[i], self.h_orientation[i])
        horizontal_angle = -math.degrees(math.atan2(r_t_h[1, 3], r_t_h[0, 3]))
        # vertical_angle = math.degrees(math.atan2(r_t_h[2, 3], r_t_h[0, 3]))

        value_angle_axis = 45
        axl.set_xlim(-value_angle_axis, value_angle_axis)
        axl.set_ylim(0.1, 3)
        axl.set_xlabel('Angle y')
        axl.set_ylabel('Distance')
        # axl.axis([-value_angle_axis, value_angle_axis, 0.1, 3], 'equal')
        axl.plot(horizontal_angle, self.distances[i], 'go')

        axll.set_xlim(-value_angle_axis, value_angle_axis)
        axll.set_ylim(-1, 1)
        axll.set_xlabel('Angle y')
        axll.set_ylabel('Delta z')
        axll.plot(horizontal_angle, self.delta_z[i], 'go')
        # Drawing the plot
        canvas.draw()

        # some additional informations as arrows
        width, height = (fig.get_size_inches() * fig.get_dpi()).astype(dtype='int32')
        img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pt1 = (275, 40)
        pt2 = (375, 40)
        if horizontal_angle >= 0:
            cv2.arrowedLine(img, pt1, pt2, (0, 0, 255), 3)
        else:
            cv2.arrowedLine(img, pt2, pt1, (0, 0, 255), 3)

        pt3 = (25, 175)
        pt4 = (25, 225)
        if self.distances[i] < 1.437:
            cv2.arrowedLine(img, pt3, pt4, (0, 255, 0), 3)
        else:
            cv2.arrowedLine(img, pt4, pt3, (0, 255, 0), 3)

        self.video_writer.write(img)
        plt.close(fig)

    # composing the video
    def video_plot_creator(self):
        max_ = len(self.frame_list)
        for i in tqdm.tqdm(range(0, max_)):
        # for i in tqdm.tqdm(range(100, 400)):
            self.plotting_function(i)
        self.video_writer.release()
        cv2.destroyAllWindows()


# ----------FUNCTIONS DEFINITIONS---------------

    # function that convert rospose to homogeneus matrix
def rospose2homogmat(p, q):
    w_r_o = np.array(quat2mat(quat_to_array(q))).astype(np.float64)  # rotation matrix of object wrt world frame
    np_pose = np.array([[p.x], [p.y], [p.z]])
    tempmat = np.hstack((w_r_o, np_pose))
    w_t_o = np.vstack((tempmat, [0, 0, 0, 1]))
    return w_t_o


# method that compute the change of frame of reference of the head wrt world to head wrt bebop
def matrix_method(p_b, q_b, p_h, q_h):
    w_t_b = rospose2homogmat(p_b, q_b)
    w_t_h = rospose2homogmat(p_h, q_h)
    inv_wtb = np.linalg.inv(w_t_b)
    b_t_h = np.matmul(inv_wtb, w_t_h)
    return b_t_h


# method to convert time
def time_conversion_to_nano(sec, nano):
    return (sec * 1000 * 1000 * 1000) + nano


# method to convert quaternion orientation to euler orientation
def quat_to_eul(orientation):
    quaternion = (
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)  # roll 0, pitch 1, yaw 2
    return euler


# find nearest value in array
def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


# given a bagfile, this function extracts the info from three topics
def get_bag_data(bag_file):
    hat_positions = []
    hat_orientaions = []
    hat_times = []
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
    return camera_times, frames, bebop_times, bebop_positions, hat_times, hat_positions, hat_orientaions, bebop_orientaions


# method used in the past
def get_distant_frame(dists, camera_times, frames, num=5):
    sorted_distances = dists[dists[:, 1].argsort()]
    frames_selected = []
    for i in range(len(dists) - num, len(dists)):
        frames_selected.append(frames[np.where(camera_times == sorted_distances[i][0])[0][0]])
    return frames_selected, sorted_distances[-num:]


# method used in the past
def get_near_frame(dists, camera_times, frames, num=5):
    sorted_distances = dists[dists[:, 1].argsort()]
    frames_selected = []
    for i in range(0, num):
        frames_selected.append(frames[np.where(camera_times == sorted_distances[i][0])[0][0]])
    return frames_selected, sorted_distances[0:num]


# transform a quaternion into array form
def quat_to_array(q):
    ret = np.zeros((4))
    ret[0] = q.w
    ret[1] = q.x
    ret[2] = q.y
    ret[3] = q.z
    return ret


# method where most of the data is pre processed.
def data_pre_processing(bag):
    # get data from bag file
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
    distances = np.zeros((len(camera_np_array)))
    delta_z = np.zeros((len(camera_np_array)))
    vect_structure = (len(camera_np_array), 3)
    hat_points = np.zeros(vect_structure)
    bebop_points = np.zeros(vect_structure)
    h_sel_positions = []
    b_sel_positions = []
    h_sel_orientations = []
    b_sel_orientations = []

    # syncing data and computing the distances array
    for i in range(0, len(camera_np_array)):
        head_position = hat_position_list[hat_idx_nearest[i]]

        hat_points[i][0] = head_position.x
        hat_points[i][1] = head_position.y
        hat_points[i][2] = head_position.z

        bebop_position = bebop_position_list[bebop_idx_nearest[i]]

        bebop_points[i][0] = bebop_position.x
        bebop_points[i][1] = bebop_position.y
        bebop_points[i][2] = bebop_position.z

        distances[i] = distance.pdist([hat_points[i], bebop_points[i]], 'euclidean')
        delta_z[i] = head_position.z - bebop_position.z

        h_sel_positions.append(head_position)
        b_sel_positions.append(bebop_position)
        h_sel_orientations.append(hat_orientation_list[hat_idx_nearest[i]])
        b_sel_orientations.append(bebop_orientation_list[bebop_idx_nearest[i]])
    return b_sel_orientations, b_sel_positions, frames_list, h_sel_orientations, h_sel_positions, distances, delta_z


# method used for creating video for each bag file
def bag_to_vid(f):
    path = bag_file_path[f[:-4]]
    print("\nreading bag: " + str(f))
    with rosbag.Bag(path + f) as bag:
        b_sel_orientations, b_sel_positions, frames_list, h_sel_orientations, h_sel_positions, distance_list, delta_z_list = data_pre_processing(bag)
    vidcr = VideoCreator(b_orientation=b_sel_orientations,
                         distances=distance_list,
                         b_position=b_sel_positions,
                         frame_list=frames_list,
                         h_orientation=h_sel_orientations,
                         h_position=h_sel_positions,
                         delta_z=delta_z_list,
                         f=f,
                         title="./video/" + f[:-4] + ".avi")
    vidcr.video_plot_creator()
    print("\nvideo : " + str(f[:-4] + " completed"))


def bag_to_pickle(f):
    path = bag_file_path[f[:-4]]
    print("\nreading bag: " + str(f))
    datacr = DatasetCreator()
    with rosbag.Bag(path + f) as bag:
        b_sel_orientations, b_sel_positions, frames_list, h_sel_orientations, h_sel_positions, distance_list, delta_z_list = data_pre_processing(bag)
    datacr.generate_data(distances=distance_list,
                         b_orientation=b_sel_orientations,
                         b_position=b_sel_positions,
                         frame_list=frames_list,
                         h_orientation=h_sel_orientations,
                         h_position=h_sel_positions,
                         delta_z=delta_z_list,
                         f=f)
    datacr.save_dataset(flag_train="cross", title=f[:-4] + ".pickle")
    print("\nCompleted pickle #" + str(f))



# ------ Main ------
def main():
    # Main selection
    scelta = raw_input("Video or dataset:[v/d]")

    # if video is selected, then bag_to_vid is mapped to a pool of processes to parallelize the computation. (Fast as the slowest ~ 1h)
    if scelta == "v":
        path1 = "./bagfiles/train/"
        path2 = "./bagfiles/validation/"
        path3 = "./bagfiles/new/"

        scelta_2 = raw_input("new data?:[y/n]")
        if scelta_2 == 'n':
            files1 = [f for f in os.listdir(path1) if f[-4:] == '.bag']
            if not files1:
                print('No bag files found!')
                return None
            files2 = [f for f in os.listdir(path2) if f[-4:] == '.bag']
            if not files2:
                print('No bag files found!')
                return None
            files = []
            for f_ in files1:
                files.append(f_)
            for f_ in files2:
                files.append(f_)
        else:
            files = [f for f in os.listdir(path3) if f[-4:] == '.bag']
            if not files:
                print('No bag files found!')
                return None

        scelta_3 = raw_input("Single or multi:[s/m]")
        if scelta_3 == 's':
            for f in files:
                bag_to_vid(f)
        else:
            pool = Pool(processes=4)
            pool.map(bag_to_vid, files[:])
            pool.close()
            pool.join()

    else:
        # # compute dataset mean distance
        scelta_2 = raw_input("Train/val, Distance or cross:[t/d/c]")

        if scelta_2 == "d":
            path = "./bagfiles/train/"
            files = [f for f in os.listdir(path) if f[-4:] == '.bag']
            if not files:
                print('No bag files found!')
                return None
            sum_dist = []
            for f in files:
                bag = rosbag.Bag(path + f)
                _, _, _, _, _, distance_list, _ = data_pre_processing(bag)
                sum_dist.append(np.mean(distance_list))



            path = "./bagfiles/validation/"
            files = [f for f in os.listdir(path) if f[-4:] == '.bag']
            if not files:
                print('No bag files found!')
                return None

            for f in files:
                bag = rosbag.Bag(path + f)
                _, _, _, _, _, distance_list, _ = data_pre_processing(bag)
                sum_dist.append(np.mean(distance_list))

            dist_mean = np.mean(sum_dist)  # 1.5527058420265916
            print(dist_mean)
        elif scelta_2 == "t":
            # create dataset, not parallelized.
            # train
            datacr_train = DatasetCreator()
            path = "./bagfiles/train/"
            files = [f for f in os.listdir(path) if f[-4:] == '.bag']
            if not files:
                print('No bag files found!')
                return None

            for f in files:
                with rosbag.Bag(path + f) as bag:
                    b_sel_orientations, b_sel_positions, frames_list, h_sel_orientations, h_sel_positions, distance_list, delta_z_list = data_pre_processing(bag)

                datacr_train.generate_data(distances=distance_list,
                                           b_orientation=b_sel_orientations,
                                           b_position=b_sel_positions,
                                           frame_list=frames_list,
                                           h_orientation=h_sel_orientations,
                                           h_position=h_sel_positions,
                                           delta_z=delta_z_list,
                                           f=f)
            datacr_train.save_dataset(flag_train="train")

            # validation
            datacr_val = DatasetCreator()
            path = "./bagfiles/validation/"
            files = [f for f in os.listdir(path) if f[-4:] == '.bag']
            if not files:
                print('No bag files found!')
                return None

            for f in files:
                with rosbag.Bag(path + f) as bag:
                    b_sel_orientations, b_sel_positions, frames_list, h_sel_orientations, h_sel_positions, distance_list, delta_z_list = data_pre_processing(bag)
                datacr_val.generate_data(distances=distance_list,
                                         b_orientation=b_sel_orientations,
                                         b_position=b_sel_positions,
                                         frame_list=frames_list,
                                         h_orientation=h_sel_orientations,
                                         h_position=h_sel_positions,
                                         delta_z=delta_z_list,
                                         f=f)
            datacr_val.save_dataset(flag_train="validation")
        elif scelta_2 == "c":
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
            for f_ in files1:
                files.append(f_)
            for f_ in files2:
                files.append(f_)

            scelta_3 = raw_input("Single or multi:[s/m]")
            if scelta_3 == 's':
                for f in files:
                    bag_to_pickle(f)
            else:
                pool = Pool(processes=4)
                pool.map(bag_to_pickle, files[:])
                pool.close()
                pool.join()
        else:
            print('Error in selection')
            return None


if __name__ == "__main__":
    main()
