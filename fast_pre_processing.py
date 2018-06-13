# ------ Import ------
import math
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import rosbag
import tqdm as tqdm
import tf

from transforms3d.derivations.quaternions import quat2mat

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


# This class handles the dataset creation.
class DatasetCreator:
    def __init__(self):
        self.dataset = []

    def generate_data(self, data_vec):
        self.dataset += data_vec

    def save_dataset(self, flag_train, title="wrong.pickle"):
        if flag_train == "train":
            shuffled_dataset = list(self.dataset)
            train = pd.DataFrame(shuffled_dataset)
            train.to_pickle("./pose_dataset/train.pickle")
        elif flag_train == "validation":
            shuffled_dataset = list(self.dataset)
            val = pd.DataFrame(shuffled_dataset)
            val.to_pickle("./pose_dataset/validation.pickle")
        elif flag_train == "cross":
            val = pd.DataFrame(list(self.dataset))
            val.to_pickle("./pose_dataset/crossvalidation/" + title)
        else:
            print("ERROR in FLAG TRAIN")
            return None


# method to convert time

def time_conversion_to_nano(sec, nano):
    return (sec * 1000 * 1000 * 1000) + nano


def get_bag_data_pandas(bag):
    # bag = rosbag.Bag(bag_file)
    h_id = []
    h_v = []
    for topic, hat, t in bag.read_messages(topics=['/optitrack/head']):
        # print("head")
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
        # print("bebop")
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
        # print("camera")
        secs = t.secs
        nsecs = t.nsecs
        c_id.append(time_conversion_to_nano(secs, nsecs))
        img = jpeg2np(image_frame.data, (107, 60))
        camera_frame = (lambda x: {'vid': x})(img)
        c_v.append(camera_frame)
    camera_df = pd.DataFrame(data=c_v, index=c_id, columns=c_v[0].keys())
    bag.close()
    return {'head_df': head_df, 'bebop_df': bebop_df, 'camera_df': camera_df}


# function that convert rospose to homogeneus matrix
def rospose2homogmat(p, q):
    w_r_o = np.array(quat2mat(q)).astype(np.float64)  # rotation matrix of object wrt world frame
    tempmat = np.hstack((w_r_o, np.expand_dims(p, axis=1)))
    w_t_o = np.vstack((tempmat, [0, 0, 0, 1]))
    return w_t_o


def change_frame_reference(pose_bebop, pose_head):
    """Change frame of reference of pose head from World to bebop.
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


# find nearest value in array
def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


def jpeg2np(image, size=None):
    """Converts a jpeg image in a 2d numpy array of RGB pixels and resizes it to the given size (if provided).
      Args:
        image: a compressed BGR jpeg image.
        size: a tuple containing width and height, or None for no resizing.

      Returns:
        the raw, resized image as a 2d numpy array of RGB pixels.
    """
    compressed = np.fromstring(image, np.uint8)
    raw = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    # TODO eliminate conversion everywhere
    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    if size:
        img = cv2.resize(img, size)

    return img


# method to convert quaternion orientation to euler orientation
def quat_to_eul(q):
    euler = tf.transformations.euler_from_quaternion(q)  # roll 0, pitch 1, yaw 2
    return euler


def pre_proc(bag_df_dict, data_id, f):
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
        quaternion_bebop = bebop_pose[['b_rot_w', 'b_rot_x', 'b_rot_y', 'b_rot_z']].values
        quaternion_head = head_pose[['h_rot_w', 'h_rot_x', 'h_rot_y', 'h_rot_z']].values
        _, _, head_yaw = quat_to_eul(quaternion_head)
        head_yaw_deg = math.degrees(head_yaw)
        _, _, bebop_yaw = quat_to_eul(quaternion_bebop)
        bebop_yaw_deg = math.degrees(bebop_yaw)

        relative_yaw_deg = head_yaw_deg - bebop_yaw_deg
        label_position = b_t_h[:-1, -1:].T[0]
        label = (label_position[0], label_position[1], label_position[2], relative_yaw_deg)
        data_vec.append((img, label))
    return data_vec


def bag_to_pickle(f):
    path = bag_file_path[f[:-4]]
    print("\nreading bag: " + str(f))
    datacr = DatasetCreator()
    with rosbag.Bag(path + f) as bag:
        bag_df_dict = get_bag_data_pandas(bag)
    data_vec = pre_proc(bag_df_dict=bag_df_dict, data_id=f[:-4], f=f)
    datacr.generate_data(data_vec=data_vec)
    datacr.save_dataset(flag_train="cross", title=f[:-4] + ".pickle")

    print("\nCompleted pickle " + str(f[:-4]))


def main():
    scelta = raw_input("Train/val or cross:[t/c]")

    if scelta == "t":
        # create dataset, not parallelized.
        # train
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
            data_vec = pre_proc(bag_df_dict=bag_df_dict, data_id=f[:-4], f=f)
            datacr_train.generate_data(data_vec=data_vec)
        datacr_train.save_dataset(flag_train="train")

        # validation
        path = "./bagfiles/validation/"
        files = [f for f in os.listdir(path) if f[-4:] == '.bag']
        if not files:
            print('No bag files found!')
            return None
        datacr_val=DatasetCreator()
        for f in files:
            path = bag_file_path[f[:-4]]
            print("\nreading bag: " + str(f))
            with rosbag.Bag(path + f) as bag:
                bag_df_dict = get_bag_data_pandas(bag)
            data_vec = pre_proc(bag_df_dict=bag_df_dict, data_id=f[:-4], f=f)
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
