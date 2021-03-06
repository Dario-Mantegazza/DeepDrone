# ------ Import ------
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import rosbag
import tf
import tqdm as tqdm
from transforms3d.derivations.quaternions import quat2mat

from global_parameters import *
from utils import jpeg2np, time_conversion_to_nano, find_nearest


class DatasetCreator:
    def __init__(self):
        """
            initializer for the class. creates an empty self.dataset
        """
        self.dataset = []

    def generate_data(self, data_vec):
        """
            append new data_vec to self.dataset
        Args:
            data_vec: vector of data
        """
        self.dataset += data_vec

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
            train.to_pickle("./dataset/train.pickle")
        elif flag_train == "validation":
            val = pd.DataFrame(list(self.dataset))
            val.to_pickle("./dataset/validation.pickle")
        elif flag_train == "cross":
            val = pd.DataFrame(list(self.dataset))
            val.to_pickle("./dataset/crossvalidation/" + title)
        else:
            print("ERROR in FLAG TRAIN")
            return None


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
    w_r_o = np.array(quat2mat(q)).astype(np.float64)
    tempmat = np.hstack((w_r_o, np.expand_dims(p, axis=1)))
    w_t_o = np.vstack((tempmat, [0, 0, 0, 1]))
    return w_t_o


def quat_to_eul(q):
    """
        Convert quaternion orientation to euler orientation
    Args:
        q: quaternion array

    Returns:
        euler: array of 3-D rotation   [roll, pitch, yaw]
    """
    euler = tf.transformations.euler_from_quaternion(q)  #
    return euler


def change_frame_reference(pose_bebop, pose_head):
    """
        Change frame of reference of pose head from World to bebop.

        Args:
            pose_bebop: pose of the bebop
            pose_head: pose of the head

        Returns:
            the new pose for head:
                bebop
                    T
                     head
    """
    position_bebop = pose_bebop[['b_pos_x', 'b_pos_y', 'b_pos_z']].values
    quaternion_bebop = pose_bebop[['b_rot_w', 'b_rot_x', 'b_rot_y', 'b_rot_z']].values
    position_head = pose_head[['h_pos_x', 'h_pos_y', 'h_pos_z']].values
    quaternion_head = pose_head[['h_rot_w', 'h_rot_x', 'h_rot_y', 'h_rot_z']].values
    w_t_b = rospose2homogmat(position_bebop, quaternion_bebop)
    w_t_h = rospose2homogmat(position_head, quaternion_head)
    b_t_w = np.linalg.inv(w_t_b)
    b_t_h = np.matmul(b_t_w, w_t_h)

    return b_t_h


def get_bag_data_pandas(bag):
    """
        Read a bag object and save data from three topics into Pandas dataframe
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
        bag: bagfile object

    Returns:
        dictionary:
        {'head_df': head_df,
         'bebop_df': bebop_df,
         'camera_df': camera_df}
        Composed of the three Pandas dataframe containing the three topics data
    """
    h_id = []
    h_v = []
    for topic, hat, t in bag.read_messages(topics=['/optitrack/head']):
        secs = t.secs
        nsecs = t.nsecs
        h_id.append(time_conversion_to_nano(secs, nsecs))
        pos_rot_dict = (lambda x, y: {'h_pos_x': x.x,
                                      'h_pos_y': x.y,
                                      'h_pos_z': x.z,
                                      'h_rot_w': y.w,
                                      'h_rot_x': y.x,
                                      'h_rot_y': y.y,
                                      'h_rot_z': y.z})(hat.pose.position, hat.pose.orientation)
        h_v.append(pos_rot_dict)
    head_df = pd.DataFrame(data=h_v, index=h_id, columns=h_v[0].keys())

    b_id = []
    b_v = []
    for topic, bebop, t in bag.read_messages(topics=['/optitrack/bebop']):
        secs = t.secs
        nsecs = t.nsecs
        b_id.append(time_conversion_to_nano(secs, nsecs))
        pos_rot_dict = (lambda x, y: {'b_pos_x': x.x,
                                      'b_pos_y': x.y,
                                      'b_pos_z': x.z,
                                      'b_rot_w': y.w,
                                      'b_rot_x': y.x,
                                      'b_rot_y': y.y,
                                      'b_rot_z': y.z})(bebop.pose.position, bebop.pose.orientation)
        b_v.append(pos_rot_dict)
    bebop_df = pd.DataFrame(data=b_v, index=b_id, columns=b_v[0].keys())

    c_id = []
    c_v = []
    for topic, image_frame, t in bag.read_messages(topics=['/bebop/image_raw/compressed']):
        secs = t.secs
        nsecs = t.nsecs
        c_id.append(time_conversion_to_nano(secs, nsecs))
        img = jpeg2np(image_frame.data, (image_width, image_height))
        camera_frame = (lambda x: {'vid': x})(img)
        c_v.append(camera_frame)
    camera_df = pd.DataFrame(data=c_v, index=c_id, columns=c_v[0].keys())
    bag.close()
    return {'head_df': head_df, 'bebop_df': bebop_df, 'camera_df': camera_df}


def processing(bag_df_dict, data_id, f):
    """
        Process data from dictionary bag_df_dict into a data vector
    Args:
        bag_df_dict: dictionary of Pandas dataframes
        data_id: id of the bag file processed
        f: bag file name, used as key for dictionary

    Returns:
        data vector: vector of tutples (image, (targe_x, target_y, target_z, target_relative_yaw)

    """
    camera_t = bag_df_dict["camera_df"].index.values
    bebop_t = bag_df_dict["bebop_df"].index.values
    head_t = bag_df_dict["head_df"].index.values
    data_vec = []
    max_ = bag_end_cut[f[:-4]]
    min_ = bag_start_cut[f[:-4]]
    for i in tqdm.tqdm(range(min_, max_), desc="processing data " + str(data_id)):
        b_id = find_nearest(bebop_t, camera_t[i])
        h_id = find_nearest(head_t, camera_t[i])

        head_pose = bag_df_dict["head_df"].iloc[h_id]
        bebop_pose = bag_df_dict["bebop_df"].iloc[b_id]

        img = bag_df_dict["camera_df"].iloc[i].values[0]

        b_t_h = change_frame_reference(bebop_pose, head_pose)

        quaternion_bebop = bebop_pose[['b_rot_x', 'b_rot_y', 'b_rot_z', 'b_rot_w']].values
        quaternion_head = head_pose[['h_rot_x', 'h_rot_y', 'h_rot_z', 'h_rot_w']].values
        _, _, head_yaw = quat_to_eul(quaternion_head)
        _, _, bebop_yaw = quat_to_eul(quaternion_bebop)
        relative_yaw = (head_yaw - bebop_yaw - np.pi)
        if relative_yaw < -np.pi:
            relative_yaw += 2 * np.pi
        target_position = b_t_h[:-1, -1:].T[0]
        target = (target_position[0], target_position[1], target_position[2], relative_yaw)
        data_vec.append((img, target))
    return data_vec


def bag_to_pickle(f):
    """
        Core method used to transforms and saves a bag file into a .pickle dataset file
    Args:
        f: file name e.g. "7.bag"
    """
    path = bag_file_path[f[:-4]]
    print("\nreading bag: " + str(f))
    datacr = DatasetCreator()
    with rosbag.Bag(path + f) as bag:
        bag_df_dict = get_bag_data_pandas(bag)
    data_vec = processing(bag_df_dict=bag_df_dict, data_id=f[:-4], f=f)
    datacr.generate_data(data_vec=data_vec)
    datacr.save_dataset(flag_train="cross", title=f[:-4] + ".pickle")

    print("\nCompleted pickle " + str(f[:-4]))


def main():
    """
        Using user input from console select which functionaly execute:
            - create single dataset (Single threaded script)
            - create crossvalidation dataset (Multi threaded script, high CPU usage.
                                                Can run single thread for debugging)
    Returns:
        None in case of errors
    """
    scelta = raw_input("Train/val or cross:[t/c]")
    if scelta == "t":
        path = "./bagfiles/train/"
        files = [f for f in os.listdir(path) if f[-4:] == '.bag']
        if not files:
            print('No bag files found!')
            return None
        datacr_train = DatasetCreator()
        for f in files:
            path = bag_file_path[f[:-4]]
            print("\nreading bag: " + str(f))
            with rosbag.Bag(path + f) as bag:
                bag_df_dict = get_bag_data_pandas(bag)
            data_vec = processing(bag_df_dict=bag_df_dict, data_id=f[:-4], f=f)
            datacr_train.generate_data(data_vec=data_vec)
        datacr_train.save_dataset(flag_train="train")

        path = "./bagfiles/validation/"
        files = [f for f in os.listdir(path) if f[-4:] == '.bag']
        if not files:
            print('No bag files found!')
            return None
        datacr_val = DatasetCreator()
        for f in files:
            path = bag_file_path[f[:-4]]
            print("\nreading bag: " + str(f))
            with rosbag.Bag(path + f) as bag:
                bag_df_dict = get_bag_data_pandas(bag)
            data_vec = processing(bag_df_dict=bag_df_dict, data_id=f[:-4], f=f)
            datacr_val.generate_data(data_vec=data_vec)
        datacr_val.save_dataset(flag_train="validation")

    elif scelta == "c":
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
