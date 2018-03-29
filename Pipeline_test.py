# --------------------IMPORT-------------------

import io
import imageio

import numpy as np
import rosbag
from PIL import Image
from PIL import ImageDraw
from scipy.spatial import distance


# ----------FUNCTIONS DEFINITIONS---------------
# region Def
def time_conversion_to_nano(sec, nano):
    return (sec * 1000 * 1000 * 1000) + nano


def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


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
        png_path = "images/"+title+"_" + str(i) + ".png"
        img.save(png_path, "PNG")
        file_list.append(png_path)

    writer = imageio.get_writer(title+'.mp4', fps=2)

    for im in file_list:
        writer.append_data(imageio.imread(im))
    writer.close()


def get_bag_data(bag_file):
    hat_poses = []
    hat_times = []
    for topic, hat, t in bag_file.read_messages(topics=['/optitrack/head']):
        hat_times.append(time_conversion_to_nano(hat.header.stamp.secs, hat.header.stamp.nsecs))
        hat_poses.append(hat.pose.position)

    bebop_poses = []
    bebop_times = []
    for topic, bebop, t in bag_file.read_messages(topics=['/optitrack/bebop']):
        bebop_times.append(time_conversion_to_nano(bebop.header.stamp.secs, bebop.header.stamp.nsecs))
        bebop_poses.append(bebop.pose.position)

    frames = []
    camera_times = []
    for topic, image_frame, t in bag_file.read_messages(topics=['/bebop/image_raw/compressed']):
        frames.append(image_frame.data)
        camera_times.append(time_conversion_to_nano(image_frame.header.stamp.secs, image_frame.header.stamp.nsecs))
    bag_file.close()
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


# endregion
# -------------------Main area----------------------
def main():
    #open the bag file
    bag = rosbag.Bag('drone.bag')

    #extract data from bag file
    camera_time_list, frame_list, bebop_time_list, bebop_pose_list, hat_time_list, hat_pose_list = get_bag_data(bag)

    # reformat some data as np array for future use
    camera_np_array = np.asarray(camera_time_list)
    bebop_np_array = np.asarray(bebop_time_list)
    hat_np_array = np.asarray(hat_time_list)

    #identify the nearest time frames of the bebop with respect of the camera data
    bebop_idx_nearest = []
    for v in camera_np_array:
        bebop_idx_nearest.append(find_nearest(bebop_np_array, v))

    #identify the nearest time frames of the hat with respect of the camera data
    hat_idx_nearest = []
    for v in camera_np_array:
        hat_idx_nearest.append(find_nearest(hat_np_array, v))

    #some variable inits
    distances = np.zeros((len(camera_np_array), 2))
    vect_structure = (len(camera_np_array), 3)
    hat_points = np.zeros(vect_structure)
    bebop_points = np.zeros(vect_structure)

    #computing the distances array/matrix
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




if __name__ == "__main__":
    main()
