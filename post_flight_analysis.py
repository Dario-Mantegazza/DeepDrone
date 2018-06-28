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


class KerasVideoCreator:
    def __init__(self, df, title="Validation.avi"):
        """
            Initializer for the class
        Args:
            df: dataframe containing data for the video
            title: videofile name
        """
        self.fps = 30
        self.df = df
        self.width = 1280
        self.height = 480
        self.video_writer = cv2.VideoWriter(title, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (self.width, self.height))
        self.PADCOLOR = [255, 255, 255]
        self.drone_im = cv2.resize(cv2.imread("drone.png"), (0, 0), fx=0.08, fy=0.08)
        self.mean_dist = 1.5

    def frame_composer(self, i):
        """
            using self.df, compose the frame
        Args:
            i: frame number
        """
        # Adjusting the image
        img_f = (self.df["frames"].iloc[i]).astype(np.uint8)
        scaled = cv2.resize(img_f, (0, 0), fx=4, fy=4)
        vert_p = int((480 - scaled.shape[0]) / 2)

        hor_p = int((640 - scaled.shape[1]) / 2)
        im_pad = cv2.copyMakeBorder(scaled,
                                    vert_p,
                                    vert_p if vert_p * 2 + scaled.shape[0] == 480 else vert_p + (480 - (vert_p * 2 + scaled.shape[0])),
                                    hor_p,
                                    hor_p if hor_p * 2 + scaled.shape[1] == 640 else hor_p + (640 - (hor_p * 2 + scaled.shape[1])),
                                    cv2.BORDER_CONSTANT, value=self.PADCOLOR)
        im_partial = cv2.cvtColor(im_pad, cv2.COLOR_RGB2BGR)
        data_area = (np.ones((480, 640, 3)) * 255).astype(np.uint8)
        im_final = np.hstack((data_area, im_partial))

        # Setting some variables
        font = cv2.FONT_HERSHEY_DUPLEX
        text_color = (0, 0, 0)
        y_d = [self.df['pred_x'].iloc[i],
               self.df['pred_y'].iloc[i],
               self.df['pred_z'].iloc[i],
               self.df['pred_yaw'].iloc[i] - np.pi]

        l_d = [self.df["opt_head_x"].iloc[i],
               self.df["opt_head_y"].iloc[i],
               self.df["opt_head_z"].iloc[i],
               self.df["opt_head_yaw"].iloc[i]]

        t_d = [self.df['target_x'].iloc[i],
               self.df['target_y'].iloc[i],
               self.df['target_z'].iloc[i],
               self.df['target_yaw'].iloc[i]]

        cv2.putText(im_final, "Frame: %s" % i, (900, 50), font, 0.5, text_color, 1, cv2.LINE_AA)

        # Top view
        triangle_color = (255, 229, 204)

        # Text Information
        cv2.putText(im_final, "X True: %.2f" % (l_d[0]), (10, 10), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "X P: %.2f" % (y_d[0]), (10, 25), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "X target: %.2f" % (t_d[0]), (10, 40), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Y True: %.2f" % (l_d[1]), (110, 10), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Y P: %.2f" % (y_d[1]), (110, 25), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Y target: %.2f" % (t_d[1]), (110, 40), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Z True: %.2f" % (l_d[2]), (210, 10), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Z P: %.2f" % (y_d[2]), (210, 25), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Z target: %.2f" % (t_d[2]), (210, 40), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Yaw True: %.2f" % (l_d[3]), (310, 10), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Yaw P: %.2f" % (y_d[3]), (310, 25), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Yaw target: %.2f" % (t_d[3]), (310, 40), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Relative pose (X, Y)", (300, 55), font, 0.5, text_color, 1, cv2.LINE_AA)

        # draw legend
        pr_color = (255, 0, 0)

        gt_color = (0, 255, 0)

        targ_color = (0, 0, 255)

        cv2.putText(im_final, "Truth", (1025, 400), font, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.circle(im_final, center=(1000, 395), radius=5, color=gt_color, thickness=2)

        cv2.putText(im_final, "Prediction", (1025, 420), font, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.circle(im_final, center=(1000, 415), radius=5, color=pr_color, thickness=5)

        cv2.putText(im_final, "Target pose for drone", (1025, 440), font, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.circle(im_final, center=(1000, 435), radius=5, color=targ_color, thickness=5)

        # Draw FOV and drone
        t_x = 330
        t_y = 400
        camera_fov = 90
        triangle_side_len = 400
        x_offset = t_x - self.drone_im.shape[0] / 2 - 4
        y_offset = t_y - 7
        im_final[y_offset:y_offset + self.drone_im.shape[0], x_offset:x_offset + self.drone_im.shape[1]] = self.drone_im
        triangle = np.array([[int(t_x - (math.sin(math.radians(camera_fov / 2)) * triangle_side_len)), int(t_y - (math.cos(math.radians(camera_fov / 2)) * triangle_side_len))],
                             [t_x, t_y],
                             [int(t_x + (math.sin(math.radians(camera_fov / 2)) * triangle_side_len)), int(t_y - (math.cos(math.radians(camera_fov / 2)) * triangle_side_len))]], np.int32)
        cv2.fillConvexPoly(im_final, triangle, color=triangle_color, lineType=1)
        scale_factor = (math.cos(math.radians(camera_fov / 2)) * triangle_side_len) / (2 * self.mean_dist)

        # vertical axis
        cv2.line(im_final,
                 (30, int(t_y - (math.cos(math.radians(camera_fov / 2)) * triangle_side_len))),
                 (30, t_y),
                 color=(0, 0, 0),
                 thickness=1)

        cv2.line(im_final,
                 (15, int((t_y - (math.cos(math.radians(camera_fov / 2)) * triangle_side_len)))),
                 (30, int((t_y - (math.cos(math.radians(camera_fov / 2)) * triangle_side_len)))),
                 color=(0, 0, 0),
                 thickness=1)
        cv2.putText(im_final, "%.1f m" % (self.mean_dist * 2), (31, int((t_y - (math.cos(math.radians(camera_fov / 2)) * triangle_side_len)))), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.line(im_final,
                 (15, int((t_y - ((math.cos(math.radians(camera_fov / 2)) * triangle_side_len) / 2)))),
                 (30, int((t_y - ((math.cos(math.radians(camera_fov / 2)) * triangle_side_len) / 2)))),
                 color=(0, 0, 0),
                 thickness=1)
        cv2.putText(im_final, "%.1f m" % self.mean_dist, (31, int((t_y + 3 - ((math.cos(math.radians(camera_fov / 2)) * triangle_side_len) / 2)))), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.line(im_final,
                 (15, t_y),
                 (30, t_y),
                 color=(0, 0, 0),
                 thickness=1)
        cv2.putText(im_final, "0 m", (31, t_y + 5), font, 0.4, text_color, 1, cv2.LINE_AA)

        # horizontal axis
        cv2.line(im_final,
                 (int(t_x - (math.sin(math.radians(camera_fov / 2)) * triangle_side_len)), 90),
                 (int(t_x + (math.sin(math.radians(camera_fov / 2)) * triangle_side_len)), 90),
                 color=(0, 0, 0),
                 thickness=1)

        cv2.line(im_final,
                 (int(t_x - (math.sin(math.radians(camera_fov / 2)) * triangle_side_len)), 75),
                 (int(t_x - (math.sin(math.radians(camera_fov / 2)) * triangle_side_len)), 90),
                 color=(0, 0, 0),
                 thickness=1)
        cv2.line(im_final,
                 (int(t_x - (math.sin(math.radians(camera_fov / 2)) * triangle_side_len) / 2), 75),
                 (int(t_x - (math.sin(math.radians(camera_fov / 2)) * triangle_side_len) / 2), 90),
                 color=(0, 0, 0),
                 thickness=1)
        cv2.line(im_final,
                 (t_x, 75),
                 (t_x, 90),
                 color=(0, 0, 0),
                 thickness=1)
        cv2.line(im_final,
                 (int(t_x + (math.sin(math.radians(camera_fov / 2)) * triangle_side_len) / 2), 75),
                 (int(t_x + (math.sin(math.radians(camera_fov / 2)) * triangle_side_len) / 2), 90),
                 color=(0, 0, 0),
                 thickness=1)
        cv2.line(im_final,
                 (int(t_x + (math.sin(math.radians(camera_fov / 2)) * triangle_side_len)), 75),
                 (int(t_x + (math.sin(math.radians(camera_fov / 2)) * triangle_side_len)), 90),
                 color=(0, 0, 0),
                 thickness=1)
        cv2.putText(im_final, "+%.1f m" % (self.mean_dist * 2), (int(t_x - 10 - scale_factor * (self.mean_dist * 2)), 70), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "+%.1f m" % self.mean_dist, (int(t_x - 10 - scale_factor * self.mean_dist), 70), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "0 m", (t_x - 4, 70), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "-%.1f m" % self.mean_dist, (int(t_x + scale_factor * self.mean_dist), 70), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "-%.1f m" % (self.mean_dist * 2), (int(t_x - 5 + scale_factor * (self.mean_dist * 2)), 70), font, 0.4, text_color, 1, cv2.LINE_AA)

        # draw GT point
        gt_x = int((t_x - scale_factor * l_d[1]))
        gt_y = int((t_y - scale_factor * l_d[0]))
        gt_center = (gt_x,
                     gt_y)
        cv2.circle(im_final, center=gt_center, radius=5, color=gt_color, thickness=2)

        # draw gt arrow
        arrow_len = 40
        # GT
        l_angle_for_cv2 = -l_d[3] + np.pi / 2
        y_angle_for_cv2 = -y_d[3] + np.pi / 2
        t_angle_for_cv2 = -t_d[3] + np.pi / 2 + np.pi

        cv2.arrowedLine(im_final,
                        gt_center,
                        (int(gt_x + (arrow_len * math.cos(l_angle_for_cv2))),
                         int(gt_y + (arrow_len * math.sin(l_angle_for_cv2)))
                         ),
                        color=gt_color,
                        thickness=2)

        # draw Pred point
        pr_x = int((t_x - scale_factor * y_d[1]))
        pr_y = int((t_y - scale_factor * y_d[0]))
        pr_center = (pr_x,
                     pr_y)
        cv2.circle(im_final, center=pr_center, radius=5, color=pr_color, thickness=5)

        # prediction arrow
        cv2.arrowedLine(im_final,
                        pr_center,
                        (int(pr_x + (arrow_len * math.cos(y_angle_for_cv2))),
                         int(pr_y + (arrow_len * math.sin(y_angle_for_cv2)))
                         ),
                        color=pr_color,
                        thickness=2)

        # draw Target point
        targ_x = int((t_x - scale_factor * t_d[1]))
        targ_y = int((t_y - scale_factor * t_d[0]))
        targ_center = (targ_x,
                       targ_y)
        cv2.circle(im_final, center=targ_center, radius=5, color=targ_color, thickness=5)

        # Target arrow
        cv2.arrowedLine(im_final,
                        targ_center,
                        (int(targ_x + (arrow_len / 2.0 * math.cos(t_angle_for_cv2))),
                         int(targ_y + (arrow_len / 2.0 * math.sin(t_angle_for_cv2)))
                         ),
                        color=targ_color,
                        thickness=2)
        # draw height

        h_x = 640
        h_y = 90
        cv2.putText(im_final, "Relative Z", (h_x, h_y), font, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "+1 m", (h_x + 65, h_y + 15), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "0 m", (h_x + 65, h_y + 164), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "-1 m", (h_x + 65, h_y + 312), font, 0.4, text_color, 1, cv2.LINE_AA)

        cv2.rectangle(im_final, (h_x + 30, h_y + 10), (h_x + 60, h_y + 310), color=(0, 0, 0), thickness=2)
        cv2.line(im_final, (h_x + 35, h_y + 160), (h_x + 55, h_y + 160), color=(0, 0, 0), thickness=1)
        h_c_x = h_x + 45
        h_c_y = h_y + 160

        h_scale_factor = 300 / 2

        gt_h_center = (h_c_x,
                       int((h_c_y - h_scale_factor * l_d[2])))
        pr_h_center = (h_c_x,
                       int((h_c_y - h_scale_factor * y_d[2])))
        cv2.circle(im_final, center=gt_h_center, radius=5, color=gt_color, thickness=2)
        cv2.circle(im_final, center=pr_h_center, radius=5, color=pr_color, thickness=5)
        self.video_writer.write(im_final)

    def video_plot_creator(self):
        """
            calls frame composers for every frame
            complete video creation
        """
        df_id = self.df.index.values
        max_ = df_id.size
        for i in tqdm.tqdm(range(0, max_)):
            self.frame_composer(i)
        self.video_writer.release()
        cv2.destroyAllWindows()


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
        vid_cr = KerasVideoCreator(data_df, title="./video/post_flight-" + str(f[:-4]) + ".avi")
        vid_cr.video_plot_creator()


if __name__ == "__main__":
    main()
