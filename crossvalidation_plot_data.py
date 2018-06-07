import math

import cv2
import numpy as np
import tqdm
from matplotlib import pyplot as plt


# class that is used to create video
class KerasVideoCreator:
    def __init__(self, x_test, labels, preds, title="Validation.avi"):
        self.fps = 30
        self.width = 1280
        self.height = 480
        self.video_writer = cv2.VideoWriter(title, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (self.width, self.height))
        self.frame_list = x_test
        self.labels = labels
        self.preds = preds
        self.PADCOLOR = [255, 255, 255]
        self.drone_im = cv2.resize(cv2.imread("drone.png"), (0, 0), fx=0.08, fy=0.08)

    # function used to compose the frame
    def frame_composer(self, i):
        # Adjusting the image
        img_f = 1 - (self.frame_list[i]).astype(np.uint8)
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
        y_d = [self.preds[0][i], self.preds[1][i], self.preds[2][i]]
        l_d = self.labels[i]
        cv2.putText(im_final, "Frame: %s" % i, (900, 50), font, 0.5, text_color, 1, cv2.LINE_AA)

        # Top view
        triangle_color = (255, 229, 204)
        angle_deg = y_d[1]

        # Text Information
        cv2.putText(im_final, "Distance T: %.3f" % (l_d[0]), (20, 20), font, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Distance P: %.3f" % (y_d[0]), (20, 40), font, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Angle T: %.3f" % (l_d[1]), (220, 20), font, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Angle P: %.3f" % angle_deg, (220, 40), font, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Delta z T: %.3f" % (l_d[2]), (400, 20), font, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Delta z P: %.3f" % (y_d[2]), (400, 40), font, 0.5, text_color, 1, cv2.LINE_AA)

        # draw legend
        pr_color = (255, 0, 0)

        gt_color = (0, 255, 0)

        cv2.putText(im_final, "Truth", (420, 425), font, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.circle(im_final, center=(405, 420), radius=5, color=gt_color, thickness=2)

        cv2.putText(im_final, "Prediction", (420, 455), font, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.circle(im_final, center=(405, 450), radius=5, color=pr_color, thickness=5)

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
        scale_factor = (math.cos(math.radians(camera_fov / 2)) * triangle_side_len) / (2 * 1.437)

        cv2.putText(im_final, "Relative pose", (300, 90), font, 0.5, text_color, 1, cv2.LINE_AA)
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
        cv2.putText(im_final, "2.8 m", (31, int((t_y - (math.cos(math.radians(camera_fov / 2)) * triangle_side_len)))), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.line(im_final,
                 (15, int((t_y - ((math.cos(math.radians(camera_fov / 2)) * triangle_side_len) / 2)))),
                 (30, int((t_y - ((math.cos(math.radians(camera_fov / 2)) * triangle_side_len) / 2)))),
                 color=(0, 0, 0),
                 thickness=1)
        cv2.putText(im_final, "1.4 m", (31, int((t_y + 3 - ((math.cos(math.radians(camera_fov / 2)) * triangle_side_len) / 2)))), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.line(im_final,
                 (15, t_y),
                 (30, t_y),
                 color=(0, 0, 0),
                 thickness=1)
        cv2.putText(im_final, "0 m", (31, t_y + 5), font, 0.4, text_color, 1, cv2.LINE_AA)

        # draw GT
        gt_center = (int((t_x + scale_factor * (math.sin(math.radians(self.labels[i, 1])) * self.labels[i, 0]))),
                     int((t_y - scale_factor * (math.cos(math.radians(self.labels[i, 1])) * self.labels[i, 0]))))
        cv2.circle(im_final, center=gt_center, radius=5, color=gt_color, thickness=2)

        # draw Pred
        pr_center = (int((t_x + scale_factor * (math.sin(math.radians(self.preds[1][i])) * self.preds[0][i]))),
                     int((t_y - scale_factor * (math.cos(math.radians(self.preds[1][i])) * self.preds[0][i]))))
        cv2.circle(im_final, center=pr_center, radius=5, color=pr_color, thickness=5)

        # draw height

        h_x = 640
        h_y = 90
        cv2.putText(im_final, "Delta Height", (h_x, h_y), font, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "+1 m", (h_x + 65, h_y + 15), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "0 m", (h_x + 65, h_y + 164), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "-1 m", (h_x + 65, h_y + 312), font, 0.4, text_color, 1, cv2.LINE_AA)

        cv2.rectangle(im_final, (h_x + 30, h_y + 10), (h_x + 60, h_y + 310), color=(0, 0, 0), thickness=2)
        cv2.line(im_final, (h_x + 35, h_y + 160), (h_x + 55, h_y + 160), color=(0, 0, 0), thickness=1)
        h_c_x = h_x + 45
        h_c_y = h_y + 160

        h_scale_factor = 300 / 2

        gt_h_center = (h_c_x,
                       int((h_c_y - h_scale_factor * self.labels[i, 2])))
        pr_h_center = (h_c_x,
                       int((h_c_y - h_scale_factor * self.preds[2][i])))
        cv2.circle(im_final, center=gt_h_center, radius=5, color=gt_color, thickness=2)
        cv2.circle(im_final, center=pr_h_center, radius=5, color=pr_color, thickness=5)
        self.video_writer.write(im_final)

    def video_plot_creator(self):
        max_ = len(self.frame_list)
        for i in tqdm.tqdm(range(0, max_)):
            self.frame_composer(i)
        self.video_writer.release()
        cv2.destroyAllWindows()


def plot_results(history, y_pred, y_test, save_dir, i):
    my_dpi = 96
    f_angle = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    tp_angle = f_angle.add_subplot(2, 2, 1)
    mse_angle = f_angle.add_subplot(2, 2, 2)
    mae_angle = f_angle.add_subplot(2, 2, 3)
    scatter_angle = f_angle.add_subplot(2, 2, 4)

    tp_angle.plot(y_test[:, 1])
    tp_angle.plot(y_pred[1])
    tp_angle.set_title('test-prediction angle')
    tp_angle.set_xlabel('frame')
    tp_angle.set_ylabel('value')
    tp_angle.legend(['test', 'pred'], loc='upper right')

    mse_angle.plot(history.history['angle_pred_mean_squared_error'])
    mse_angle.plot(history.history['val_angle_pred_mean_squared_error'])
    mse_angle.set_title('angle MSE')
    mse_angle.set_xlabel('epoch')
    mse_angle.set_ylabel('error')
    mse_angle.legend(['train', 'validation'], loc='upper right')

    mae_angle.plot(history.history['angle_pred_loss'])
    mae_angle.plot(history.history['val_angle_pred_loss'])
    mae_angle.set_title('angle loss(MAE)')
    mae_angle.set_xlabel('epoch')
    mae_angle.set_ylabel('MAE')
    mae_angle.legend(['train', 'test'], loc='upper right')

    scatter_angle.scatter(y_test[:, 1], y_pred[1])
    scatter_angle.set_title('scatter-plot angle')
    scatter_angle.set_xlabel('thruth')
    scatter_angle.set_ylabel('pred')
    scatter_angle.set_xlim(-50, +50)
    scatter_angle.set_ylim(-50, +50)

    f_angle.savefig(save_dir + "/result_model_" + str(i) + "/angle.png")

    f_distance = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    tp_distance = f_distance.add_subplot(2, 2, 1)
    mse_distance = f_distance.add_subplot(2, 2, 2)
    mae_distance = f_distance.add_subplot(2, 2, 3)
    scatter_distance = f_distance.add_subplot(2, 2, 4)

    tp_distance.plot(y_test[:, 0])
    tp_distance.plot(y_pred[0])
    tp_distance.set_title('test-prediction distance')
    tp_distance.set_xlabel('frame')
    tp_distance.set_ylabel('value')
    tp_distance.legend(['test', 'pred'], loc='upper right')

    mse_distance.plot(history.history['distance_pred_mean_squared_error'])
    mse_distance.plot(history.history['val_distance_pred_mean_squared_error'])
    mse_distance.set_title('distance MSE')
    mse_distance.set_xlabel('epoch')
    mse_distance.set_ylabel('error')
    mse_distance.legend(['train', 'validation'], loc='upper right')

    mae_distance.plot(history.history['distance_pred_loss'])
    mae_distance.plot(history.history['val_distance_pred_loss'])
    mae_distance.set_title('distance loss (MAE)')
    mae_distance.set_xlabel('epoch')
    mae_distance.set_ylabel('MAE')
    mae_distance.legend(['train', 'test'], loc='upper right')

    scatter_distance.scatter(y_test[:, 0], y_pred[0])
    scatter_distance.set_title('scatter-plot distance')
    scatter_distance.set_ylabel('pred')
    scatter_distance.set_xlabel('thruth')
    scatter_distance.set_xlim(0, +3)
    scatter_distance.set_ylim(0, +3)
    f_distance.savefig(save_dir + "/result_model_" + str(i) + "/distance.png")

    f_height = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    tp_height = f_height.add_subplot(2, 2, 1)
    mse_height = f_height.add_subplot(2, 2, 2)
    mae_height = f_height.add_subplot(2, 2, 3)
    scatter_height = f_height.add_subplot(2, 2, 4)

    tp_height.plot(y_test[:, 2])
    tp_height.plot(y_pred[2])
    tp_height.set_title('test-prediction height')
    tp_height.set_xlabel('frame')
    tp_height.set_ylabel('value')
    tp_height.legend(['test', 'pred'], loc='upper right')

    mse_height.plot(history.history['height_pred_mean_squared_error'])
    mse_height.plot(history.history['val_height_pred_mean_squared_error'])
    mse_height.set_title('height MSE')
    mse_height.set_xlabel('epoch')
    mse_height.set_ylabel('error')
    mse_height.legend(['train', 'validation'], loc='upper right')

    mae_height.plot(history.history['height_pred_loss'])
    mae_height.plot(history.history['val_height_pred_loss'])
    mae_height.set_title('height loss (MAE)')
    mae_height.set_xlabel('epoch')
    mae_height.set_ylabel('MAE')
    mae_height.legend(['train', 'test'], loc='upper right')

    scatter_height.scatter(y_test[:, 2], y_pred[2])
    scatter_height.set_title('scatter-plot height')
    scatter_height.set_ylabel('pred')
    scatter_height.set_xlabel('thruth')
    scatter_height.set_xlim(-1, +1)
    scatter_height.set_ylim(-1, +1)
    f_height.savefig(save_dir + "/result_model_" + str(i) + "/height.png")


def history_data_plot_crossvalidation(history_list, save_dir):
    angle_loss_list = []
    angle_mse_list = []
    val_angle_loss_list = []
    val_angle_mse_list = []
    distance_loss_list = []
    distance_mse_list = []
    val_distance_loss_list = []
    val_distance_mse_list = []
    height_loss_list = []
    height_mse_list = []
    val_height_loss_list = []
    val_height_mse_list = []
    for hist in history_list:
        angle_loss_list.append(hist['angle_pred_loss'])
        angle_mse_list.append(hist['angle_pred_mean_squared_error'])
        val_angle_loss_list.append(hist['val_angle_pred_loss'])
        val_angle_mse_list.append(hist['val_angle_pred_mean_squared_error'])
        distance_loss_list.append(hist['distance_pred_loss'])
        distance_mse_list.append(hist['distance_pred_mean_squared_error'])
        val_distance_loss_list.append(hist['val_distance_pred_loss'])
        val_distance_mse_list.append(hist['val_distance_pred_mean_squared_error'])
        height_loss_list.append(hist['height_pred_loss'])
        height_mse_list.append(hist['height_pred_mean_squared_error'])
        val_height_loss_list.append(hist['val_height_pred_loss'])
        val_height_mse_list.append(hist['val_height_pred_mean_squared_error'])
    with open(save_dir + "/mean_results.txt", "w+") as outfile:
        outfile.write("Mean Results across 5-fold crossvalidation\n")
        outfile.write("== == == == == == == == == == == ==\n")
        outfile.write("angle_pred_loss:         %.3f" % (np.mean(angle_loss_list)) + '\n')
        outfile.write("angle_mse_loss:          %.3f" % (np.mean(angle_mse_list)) + '\n')
        outfile.write("val_angle_pred_loss:     %.3f" % (np.mean(val_angle_loss_list)) + '\n')
        outfile.write("val_angle_mse_loss:      %.3f" % (np.mean(val_angle_mse_list)) + '\n')
        outfile.write("distance_pred_loss:      %.3f" % (np.mean(distance_loss_list)) + '\n')
        outfile.write("distance_mse_loss:       %.3f" % (np.mean(distance_mse_list)) + '\n')
        outfile.write("val_distance_pred_loss:  %.3f" % (np.mean(val_distance_loss_list)) + '\n')
        outfile.write("val_distance_mse_loss:   %.3f" % (np.mean(val_distance_mse_list)) + '\n')
        outfile.write("height_pred_loss:        %.3f" % (np.mean(height_loss_list)) + '\n')
        outfile.write("height_mse_loss:         %.3f" % (np.mean(height_mse_list)) + '\n')
        outfile.write("val_height_pred_loss:    %.3f" % (np.mean(val_height_loss_list)) + '\n')
        outfile.write("val_height_mse_loss:     %.3f" % (np.mean(val_height_mse_list)) + '\n')
        outfile.write("== == == == == == == == == == == ==\n")
        outfile.close()
    my_dpi = 96
    f_angle = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    mse_angle = f_angle.add_subplot(121)
    mae_angle = f_angle.add_subplot(122)
    mse_angle.plot(np.mean(np.array(angle_mse_list), axis=0))
    mse_angle.plot(np.mean(np.array(val_angle_mse_list), axis=0))
    mse_angle.set_title('mean angle MSE')
    mse_angle.set_xlabel('epoch')
    mse_angle.set_ylabel('error')
    mse_angle.legend(['train', 'validation'], loc='upper right')
    mae_angle.plot(np.mean(np.array(angle_loss_list), axis=0))
    mae_angle.plot(np.mean(np.array(val_angle_loss_list), axis=0))
    mae_angle.set_title('mean angle loss(MAE)')
    mae_angle.set_xlabel('epoch')
    mae_angle.set_ylabel('MAE')
    mae_angle.legend(['train', 'test'], loc='upper right')
    f_angle.savefig(save_dir + "mean_angle_result.png")
    f_distance = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    mse_distance = f_distance.add_subplot(121)
    mae_distance = f_distance.add_subplot(122)
    mse_distance.plot(np.mean(np.array(distance_mse_list), axis=0))
    mse_distance.plot(np.mean(np.array(val_distance_mse_list), axis=0))
    mse_distance.set_title('mean distance MSE')
    mse_distance.set_xlabel('epoch')
    mse_distance.set_ylabel('error')
    mse_distance.legend(['train', 'validation'], loc='upper right')
    mae_distance.plot(np.mean(np.array(distance_loss_list), axis=0))
    mae_distance.plot(np.mean(np.array(val_distance_loss_list), axis=0))
    mae_distance.set_title('mean distance loss(MAE)')
    mae_distance.set_xlabel('epoch')
    mae_distance.set_ylabel('MAE')
    mae_distance.legend(['train', 'test'], loc='upper right')
    f_distance.savefig(save_dir + "mean_distance_result.png")
    f_height = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    mse_height = f_height.add_subplot(121)
    mae_height = f_height.add_subplot(122)
    mse_height.plot(np.mean(np.array(height_mse_list), axis=0))
    mse_height.plot(np.mean(np.array(val_height_mse_list), axis=0))
    mse_height.set_title('mean height MSE')
    mse_height.set_xlabel('epoch')
    mse_height.set_ylabel('error')
    mse_height.legend(['train', 'validation'], loc='upper right')
    mae_height.plot(np.mean(np.array(height_loss_list), axis=0))
    mae_height.plot(np.mean(np.array(val_height_loss_list), axis=0))
    mae_height.set_title('mean height loss(MAE)')
    mae_height.set_xlabel('epoch')
    mae_height.set_ylabel('MAE')
    mae_height.legend(['train', 'test'], loc='upper right')
    f_height.savefig(save_dir + "mean_height_result.png")
