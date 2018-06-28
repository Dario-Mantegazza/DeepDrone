# ------ Import ------
import io
import math
import os
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
from utils import find_nearest,time_conversion_to_nano
from global_parameters import *


# ------ Classes ------

# This class handles the dataset creation.
class DatasetCreator:
    def __init__(self):
        """
           initializer for the class. creates an empty self.dataset
        """
        self.dataset = []

    def generate_data(self, distances, b_orientation, b_position, frame_list, h_orientation, h_position, delta_z, f):
        """
            Cycles through different npArrays and call other methods to compose the dataset.
        Args:
            distances: list of distance user-drone
            b_orientation: bebop orientation array
            b_position: bebop position array
            frame_list: camera frame list
            h_orientation: head orientation array
            h_position: head orientation array
            delta_z: height difference list
            f: file name
        """
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
            self.data_aggregator(i)

    def data_aggregator(self, i):
        """
            append a frame with labels into the self.dataset variable.
        Args:
            i: frame number
        """
        img = Image.open(io.BytesIO(self.frame_list[i]))
        raw_frame = list(img.getdata())
        frame = []
        for b in raw_frame:
            frame.append(b)
        reshaped_fr = np.reshape(np.array(frame, dtype=np.int64), (480, 856, 3))
        reshaped_fr = reshaped_fr.astype(np.uint8)

        scaled_fr = cv2.resize(reshaped_fr, (image_width, image_height))

        r_t_h = matrix_method(self.b_position[i], self.b_orientation[i], self.h_position[i], self.h_orientation[i])
        horizontal_angle = -math.degrees(math.atan2(r_t_h[1, 3], r_t_h[0, 3]))
        label = (self.distances[i], horizontal_angle, self.delta_z[i])

        self.dataset.append((scaled_fr, label))

    def save_dataset(self, flag_train, title="wrong.pickle"):
        """
            Save the dataset in one of three forms
                - train set
                - validation set
                - single set part of crossvalidation
        Args:
            flag_train: flag indicating the type of dataset to be saved
            title: name of the dateset file

        Returns:
            None if error in flag_train
        """
        if flag_train == "train":
            train = pd.DataFrame(list(self.dataset))
            train.to_pickle("./dataset/old/train.pickle")
        elif flag_train == "validation":
            val = pd.DataFrame(list(self.dataset))
            val.to_pickle("./dataset/old/validation.pickle")
        elif flag_train == "cross":
            val = pd.DataFrame(list(self.dataset))
            val.to_pickle("./dataset/old/crossvalidation/" + title)
        else:
            print("ERROR in FLAG TRAIN")
            return None


# Class used for creating video to analyze new data from bag files.
class VideoCreator:
    def __init__(self, b_orientation, distances, b_position, frame_list, h_orientation, h_position, delta_z, f, title="test.avi"):
        """
            Initializer for the class
        Args:
            distances: list of distance user-drone
            b_orientation: bebop orientation array
            b_position: bebop position array
            frame_list: camera frame list
            h_orientation: head orientation array
            h_position: head orientation array
            delta_z: height difference list
            f: bag file name
            title: video file name
        """
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

    def plotting_function(self, i):
        """
            Given an index compose the frame for the video.
        Args:
            i: frame number
        """
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
        axr.plot(self.b_position[i].x, self.b_position[i].y, "ro", self.h_position[i].x, self.h_position[i].y, "go")
        axr.arrow(self.h_position[i].x, self.h_position[i].y, arrow_length * np.cos(h_theta), arrow_length * np.sin(h_theta), head_width=0.05, head_length=0.1, fc='g', ec='g')
        axr.arrow(self.b_position[i].x, self.b_position[i].y, arrow_length * np.cos(b_theta), arrow_length * np.sin(b_theta), head_width=0.05, head_length=0.1, fc='r', ec='r')

        # LEFT PLOT: here we represent the distance on the y axis and the heading correction for the drone in degrees on the x-axis
        r_t_h = matrix_method(self.b_position[i], self.b_orientation[i], self.h_position[i], self.h_orientation[i])
        horizontal_angle = -math.degrees(math.atan2(r_t_h[1, 3], r_t_h[0, 3]))

        value_angle_axis = 45
        axl.set_xlim(-value_angle_axis, value_angle_axis)
        axl.set_ylim(0.1, 3)
        axl.set_xlabel('Angle y')
        axl.set_ylabel('Distance')
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
        if self.distances[i] < 1.5:
            cv2.arrowedLine(img, pt3, pt4, (0, 255, 0), 3)
        else:
            cv2.arrowedLine(img, pt4, pt3, (0, 255, 0), 3)

        self.video_writer.write(img)
        plt.close(fig)

    def video_plot_creator(self):
        """
            calls frame composers for every frame
            complete video creation
        """
        max_ = len(self.frame_list)
        for i in tqdm.tqdm(range(0, max_)):
            self.plotting_function(i)
        self.video_writer.release()
        cv2.destroyAllWindows()


# ----------FUNCTIONS DEFINITIONS---------------

def rospose2homogmat(p, q):
    """
         Convert rospose Pose to homogeneus matrix
     Args:
         p: position array
         q: rotation quaternion array

     Returns:
         w_t_o: Homogeneous roto-translation matrix
             World
                 T
                   object
     """
    w_r_o = np.array(quat2mat(quat_to_array(q))).astype(np.float64)  # rotation matrix of object wrt world frame
    np_pose = np.array([[p.x], [p.y], [p.z]])
    tempmat = np.hstack((w_r_o, np_pose))
    w_t_o = np.vstack((tempmat, [0, 0, 0, 1]))
    return w_t_o


def matrix_method(p_b, q_b, p_h, q_h):
    """
         Change frame of reference of pose head from World to bebop.

         Args:
             q_h: head quaternion
             p_h: head position
             q_b: bebop quaternion
             p_b: bebop position

         Returns:
             the new pose for head:
                 bebop
                     T
                      head
     """
    w_t_b = rospose2homogmat(p_b, q_b)
    w_t_h = rospose2homogmat(p_h, q_h)
    inv_wtb = np.linalg.inv(w_t_b)
    b_t_h = np.matmul(inv_wtb, w_t_h)
    return b_t_h


def quat_to_eul(orientation):
    """
        Convert quaternion orientation to euler orientation
    Args:
        orientation: ros quaternion message

    Returns:
        euler: array of 3-D rotation   [roll, pitch, yaw]
    """
    quaternion = (
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)  # roll 0, pitch 1, yaw 2
    return euler


def get_bag_data(bag_file):
    """
        Read a bag object and save data from three topics into multiple lists
        topics:
            /optitrack/head:
                -timestamp of recording
                -poseStamped message
            /optitrack/bebop:
                -timestamp of recording
                -poseStamped message
            /bebop/image_raw/compressed:
                -timestamp of recording
                -camera feed data
    Args:
        bag_file: bagfile object

    Returns:
        camera_times: camera feed timestamp list
        frames: frame lists
        bebop_times : bebop pose timestamp list
        bebop_positions: bebop position list
        head_times: head timestamp list
        head_positions: head position list
        head_orientations: head orientation list
        bebop_orientations: bebop orientation list

    """
    head_positions = []
    head_orientations = []
    head_times = []
    for topic, hat, t in bag_file.read_messages(topics=['/optitrack/head']):
        secs = t.secs
        nsecs = t.nsecs
        head_times.append(time_conversion_to_nano(secs, nsecs))

        head_positions.append(hat.pose.position)
        head_orientations.append(hat.pose.orientation)

    bebop_positions = []
    bebop_orientations = []
    bebop_times = []
    for topic, bebop, t in bag_file.read_messages(topics=['/optitrack/bebop']):
        secs = t.secs
        nsecs = t.nsecs
        bebop_times.append(time_conversion_to_nano(secs, nsecs))
        bebop_positions.append(bebop.pose.position)
        bebop_orientations.append(bebop.pose.orientation)

    frames = []
    camera_times = []
    for topic, image_frame, t in bag_file.read_messages(topics=['/bebop/image_raw/compressed']):
        secs = t.secs
        nsecs = t.nsecs
        frames.append(image_frame.data)
        camera_times.append(time_conversion_to_nano(secs, nsecs))

    bag_file.close()
    return camera_times, frames, bebop_times, bebop_positions, head_times, head_positions, head_orientations, bebop_orientations


def quat_to_array(q):
    """
        transform a ros quaternion orientation message into array form
    Args:
        q:

    Returns:

    """
    ret = np.zeros((4))
    ret[0] = q.w
    ret[1] = q.x
    ret[2] = q.y
    ret[3] = q.z
    return ret


# method where most of the data is pre processed.
def data_pre_processing(bag):
    """
    Process data from dictionary bag_df_dict into a multiple varaibles
    Args:
        bag: bag file object

    Returns:
        b_sel_orientations = bebop selected orientations
        b_sel_positions = bebop selected positions
        frames_list = camera frame list
        h_sel_orientations = head selected orientations
        h_sel_positions = head selected positions
        distances = selected distancese user-drone
        delta_z = selected height differences list

    """
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


def bag_to_vid(f):
    """
        Creates a video for a bag file
    Args:
        f: bag file name e.g. "5.bag"
    """
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
    """
        Creates a pickle for a bag file
    Args:
        f: bag file name e.g. "5.bag"
    """
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
    """
        Using user input from console select which functionaly execute:
            - if video is selected create video using multi threaded script
                (can be run sinigle thread for debug)
            - if dataset is selected three more options are available:
                - compute mean distance
                - create dataset, not parallelized. (used for debugging)
                - create dataset, parallelized.

    Returns:
        None in case of errors
    """
    # Main selection
    scelta = raw_input("Video or dataset:[v/d]")

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
