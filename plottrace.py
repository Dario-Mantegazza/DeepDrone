# ------ Import ------
import math
import os

import cv2
import numpy as np
import pandas as pd
import rosbag
import tf
import tqdm as tqdm
from transforms3d.derivations.quaternions import quat2mat

from global_parameters import *
from utils import jpeg2np, time_conversion_to_nano, find_nearest


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

# def plotline()
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
            /bebop/head/pred
                -timestamp of recording
                -poseStamped message
            /bebop/target
                -timestamp of recording
                -poseStamped message
    Args:
        bag: bagfile object

    Returns:
        dictionary:
         {'head_df': head_df,
         'bebop_df': bebop_df,
         'camera_df': camera_df,
         'prediction_df': prediction_df,
         'target_df': target_df}
        Composed of the three Pandas dataframe containing the five topics data
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

    p_id = []
    p_v = []
    for topic, data, t in bag.read_messages(topics=['/bebop/head/pred']):
        secs = t.secs
        nsecs = t.nsecs
        p_id.append(time_conversion_to_nano(secs, nsecs))
        pos_rot_dict = (lambda x, y: {'p_pos_x': x.x,
                                      'p_pos_y': x.y,
                                      'p_pos_z': x.z,
                                      'p_rot_w': y.w,
                                      'p_rot_x': y.x,
                                      'p_rot_y': y.y,
                                      'p_rot_z': y.z})(data.pose.position, data.pose.orientation)
        p_v.append(pos_rot_dict)
    prediction_df = pd.DataFrame(data=p_v, index=p_id, columns=p_v[0].keys())

    t_id = []
    t_v = []
    for topic, data, t in bag.read_messages(topics=['/bebop/target']):
        secs = t.secs
        nsecs = t.nsecs
        t_id.append(time_conversion_to_nano(secs, nsecs))
        pos_rot_dict = (lambda x, y: {'t_pos_x': x.x,
                                      't_pos_y': x.y,
                                      't_pos_z': x.z,
                                      't_rot_w': y.w,
                                      't_rot_x': y.x,
                                      't_rot_y': y.y,
                                      't_rot_z': y.z})(data.pose.position, data.pose.orientation)
        t_v.append(pos_rot_dict)
    target_df = pd.DataFrame(data=t_v, index=t_id, columns=t_v[0].keys())
    bag.close()
    return {'head_df': head_df, 'bebop_df': bebop_df, 'camera_df': camera_df, 'prediction_df': prediction_df, 'target_df': target_df}


def processing(bag_df_dict, idx):
    """
        Process data from dictionary bag_df_dict into a target_df dataframe
    Args:
        bag_df_dict: dictionary of Pandas dataframes
        idx: bagfile index


    Returns:
        target_df: Pandas dataframe with the followind columns
            opt_head_x
            opt_head_y
            opt_head_z
            opt_head_yaw
            pred_x
            pred_y
            pred_z
            pred_yaw
            target_x
            target_y
            target_z
            target_yaw
            frames

    """
    camera_t = bag_df_dict["camera_df"].index.values
    bebop_t = bag_df_dict["bebop_df"].index.values
    head_t = bag_df_dict["head_df"].index.values
    prediction_t = bag_df_dict["prediction_df"].index.values
    target_t = bag_df_dict["target_df"].index.values
    data_vec = []
    data_id = []
    for i in tqdm.tqdm(range(0, camera_t.size), desc="processing data " + str(idx)):
        data_id.append(camera_t[i])
        b_id = find_nearest(bebop_t, camera_t[i])
        h_id = find_nearest(head_t, camera_t[i])
        p_id = find_nearest(prediction_t, camera_t[i])
        t_id = find_nearest(target_t, camera_t[i])

        head_pose = bag_df_dict["head_df"].iloc[h_id]
        bebop_pose = bag_df_dict["bebop_df"].iloc[b_id]
        prediction_pose = bag_df_dict["prediction_df"].iloc[p_id]
        target_pose = bag_df_dict["target_df"].iloc[t_id]

        img = bag_df_dict["camera_df"].iloc[i].values[0]

        b_t_h = change_frame_reference(bebop_pose, head_pose)

        quaternion_bebop = bebop_pose[['b_rot_x', 'b_rot_y', 'b_rot_z', 'b_rot_w']].values
        quaternion_head = head_pose[['h_rot_x', 'h_rot_y', 'h_rot_z', 'h_rot_w']].values
        quaternion_prediction = prediction_pose[['p_rot_x',
                                                 'p_rot_y',
                                                 'p_rot_z',
                                                 'p_rot_w']].values
        quaternion_target = target_pose[['t_rot_x',
                                         't_rot_y',
                                         't_rot_z',
                                         't_rot_w']].values
        position_prediction = prediction_pose[['p_pos_x',
                                               'p_pos_y',
                                               'p_pos_z']].values
        position_target = target_pose[['t_pos_x',
                                       't_pos_y',
                                       't_pos_z']].values
        _, _, head_yaw = quat_to_eul(quaternion_head)
        _, _, bebop_yaw = quat_to_eul(quaternion_bebop)
        _, _, prediction_yaw = quat_to_eul(quaternion_prediction)
        _, _, target_yaw = quat_to_eul(quaternion_target)
        relative_yaw = (head_yaw - bebop_yaw - np.pi)
        if relative_yaw < -np.pi:
            relative_yaw += 2 * np.pi
        label_position = b_t_h[:-1, -1:].T[0]
        data_dict = (lambda o_pose, o_yaw, p_pose, p_yaw, t_pose, t_yaw, frame:
                     {'opt_head_x': o_pose[0],
                      'opt_head_y': o_pose[1],
                      'opt_head_z': o_pose[2],
                      'opt_head_yaw': o_yaw,
                      'pred_x': p_pose[0],
                      'pred_y': p_pose[1],
                      'pred_z': p_pose[2],
                      'pred_yaw': p_yaw,
                      'target_x': t_pose[0],
                      'target_y': t_pose[1],
                      'target_z': t_pose[2],
                      'target_yaw': t_yaw,
                      'frames': frame})(label_position, relative_yaw, position_prediction, prediction_yaw, position_target, target_yaw, img)
        data_vec.append(data_dict)
    target_df = pd.DataFrame(data=data_vec, index=data_id, columns=data_vec[0].keys())
    return target_df


def main():
    """
        reads bag files and calls data processing and video creation
    Returns:
        None in case of errors
    """
    path = "./Bag_flight/"
    files = [f for f in os.listdir(path) if f[-4:] == '.bag']
    if not files:
        print('No bag files found!')
        return None
    for f in files:
        print("\nreading bag: " + str(f))
        with rosbag.Bag(path + f) as bag:
            bag_df_dict = get_bag_data_pandas(bag)
        data_df = processing(bag_df_dict=bag_df_dict, id=f[:-4])



if __name__ == "__main__":
    main()
