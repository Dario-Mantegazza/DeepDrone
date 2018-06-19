import math

import cv2
import numpy as np
import tqdm
from matplotlib import pyplot as plt

output_names = ["x_pred", "y_pred", "z_pred", "yaw_pred"]


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
        self.mean_dist = 1.5

    # function used to compose the frame
    def frame_composer(self, i):
        # Adjusting the image
        img_f = 255 - (self.frame_list[i]).astype(np.uint8)
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
        y_d = [self.preds[0][i], self.preds[1][i], self.preds[2][i], self.preds[3][i]]
        l_d = self.labels[i]
        # print(i)
        cv2.putText(im_final, "Frame: %s" % i, (900, 50), font, 0.5, text_color, 1, cv2.LINE_AA)

        # Top view
        triangle_color = (255, 229, 204)

        # Text Information
        cv2.putText(im_final, "X T: %.3f" % (l_d[0]), (10, 10), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "X P: %.3f" % (y_d[0]), (10, 25), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Y T: %.3f" % (l_d[1]), (110, 10), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Y P: %.3f" % (y_d[1]), (110, 25), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Z T: %.3f" % (l_d[2]), (210, 10), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Z P: %.3f" % (y_d[2]), (210, 25), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Yaw T: %.3f" % (l_d[3]), (310, 10), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Yaw P: %.3f" % (y_d[3]), (310, 25), font, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(im_final, "Relative pose (X, Y)", (300, 50), font, 0.5, text_color, 1, cv2.LINE_AA)

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
        gt_x = int((t_x - scale_factor * self.labels[i, 1]))
        gt_y = int((t_y - scale_factor * self.labels[i, 0]))
        gt_center = (gt_x,
                     gt_y)
        cv2.circle(im_final, center=gt_center, radius=5, color=gt_color, thickness=2)

        # draw gt arrow
        arrow_len = 40
        # GT
        l_angle_for_cv2 = -l_d[3] + np.pi / 2
        y_angle_for_cv2 = -y_d[3] + np.pi / 2

        cv2.arrowedLine(im_final,
                        gt_center,
                        (int(gt_x + (arrow_len * math.cos(l_angle_for_cv2))),
                         int(gt_y + (arrow_len * math.sin(l_angle_for_cv2)))
                         ),
                        color=gt_color,
                        thickness=2)

        # draw Pred point
        pr_x = int((t_x - scale_factor * self.preds[1][i]))
        pr_y = int((t_y - scale_factor * self.preds[0][i]))
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
                       int((h_c_y - h_scale_factor * self.labels[i, 2])))
        pr_h_center = (h_c_x,
                       int((h_c_y - h_scale_factor * self.preds[2][i])))
        cv2.circle(im_final, center=gt_h_center, radius=5, color=gt_color, thickness=2)
        cv2.circle(im_final, center=pr_h_center, radius=5, color=pr_color, thickness=5)
        self.video_writer.write(im_final)

    def video_plot_creator(self):
        max_ = len(self.frame_list)
        for i in tqdm.tqdm(range(0, max_)):
            # for i in tqdm.tqdm(range(5000, 6000)):
            self.frame_composer(i)
        self.video_writer.release()
        cv2.destroyAllWindows()


def plot_results(history, y_pred, y_test, dumb_pred):
    figures = []
    for i in range(4):
        fig = plt.figure()
        fig.tight_layout()
        tp = fig.add_subplot(2, 2, 1)
        mse = fig.add_subplot(2, 2, 2)
        mae = fig.add_subplot(2, 2, 3)
        sct = fig.add_subplot(2, 2, 4)

        tp.plot(y_test[:, i])
        tp.plot(y_pred[i], alpha=0.9)
        tp.set_title('test-prediction ' + str(output_names[i]))
        tp.set_xlabel('frame')
        tp.set_ylabel('value')
        tp.legend(['test', 'pred'], loc='upper right')

        mse.plot(history.history[str(output_names[i]) + '_mean_squared_error'])
        mse.plot(history.history['val_' + str(output_names[i]) + '_mean_squared_error'])
        mse.axhline(dumb_pred[i][0], c='r', ls='--')
        mse.set_title(str(output_names[i]) + ' MSE')
        mse.set_xlabel('epoch')
        mse.set_ylabel('mse')
        mse.set_ylim(0, (2 * dumb_pred[i][0]))
        mse.legend(['train', 'validation', 'mse dumb'], loc='upper right')

        mae.plot(history.history[str(output_names[i]) + '_loss'])
        mae.plot(history.history['val_' + str(output_names[i]) + '_loss'])
        mae.axhline(dumb_pred[i][1], c='r', ls='--')
        mae.set_title(str(output_names[i]) + ' loss(MAE)')
        mae.set_xlabel('epoch')
        mae.set_ylabel('mae')
        mae.set_ylim(0, (2 * dumb_pred[i][1]))

        mae.legend(['train', 'test', 'mae dumb'], loc='upper right')

        # sct.set(adjustable='box-forced', aspect='equal')
        sct.set(adjustable='box', aspect='equal')
        sct.scatter(y_test[:, i], y_pred[i], alpha=0.30, s=10, c='g')
        sct.plot(sct.get_xlim(), sct.get_ylim(), ls="--", c="b")
        sct.set_title('scatter-plot ' + str(output_names[i]))
        sct.set_xlabel('thruth')
        sct.set_ylabel('pred')
        sct.axis('equal')
        sct.legend(['diagonal', 'datapoints'], loc='upper left')
        figures.append(fig)
    plt.show()


def plot_results_cross(history, y_pred, y_test, dumb_pred, save_dir, j):
    my_dpi = 96
    for i in range(4):
        fig = plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
        fig.tight_layout()
        tp = fig.add_subplot(2, 2, 1)
        mse = fig.add_subplot(2, 2, 2)
        mae = fig.add_subplot(2, 2, 3)
        sct = fig.add_subplot(2, 2, 4)

        tp.plot(y_test[:, i])
        tp.plot(y_pred[i], alpha=0.9)
        tp.set_title('test-prediction ' + str(output_names[i]))
        tp.set_xlabel('frame')
        tp.set_ylabel('value')
        tp.legend(['test', 'pred'], loc='upper right')

        mse.plot(history.history[str(output_names[i]) + '_mean_squared_error'])
        mse.plot(history.history['val_' + str(output_names[i]) + '_mean_squared_error'])
        mse.axhline(dumb_pred[i][0], c='r', ls='--')
        mse.set_title(str(output_names[i]) + ' MSE')
        mse.set_xlabel('epoch')
        mse.set_ylabel('mse')
        mse.set_ylim(0, (2 * dumb_pred[i][0]))
        mse.legend(['train', 'validation', 'mse dumb'], loc='upper right')

        mae.plot(history.history[str(output_names[i]) + '_loss'])
        mae.plot(history.history['val_' + str(output_names[i]) + '_loss'])
        mae.axhline(dumb_pred[i][1], c='r', ls='--')
        mae.set_title(str(output_names[i]) + ' loss(MAE)')
        mae.set_xlabel('epoch')
        mae.set_ylabel('mae')
        mae.set_ylim(0, (2 * dumb_pred[i][1]))

        mae.legend(['train', 'test', 'mae dumb'], loc='upper right')

        sct.set(adjustable='box', aspect='equal')
        sct.scatter(y_test[:, i], y_pred[i], alpha=0.30, s=10, c='g')
        sct.plot(sct.get_xlim(), sct.get_ylim(), ls="--", c="b")
        sct.set_title('scatter-plot ' + str(output_names[i]))
        sct.set_xlabel('thruth')
        sct.set_ylabel('pred')
        sct.axis('equal')
        sct.legend(['diagonal', 'datapoints'], loc='upper left')

        fig.savefig(save_dir + "/result_model_" + str(j) + "/" + str(output_names[i]) + ".png")


def history_data_plot_crossvalidation(history_list, dumb_list, save_dir):
    my_dpi = 96
    mean_dumb = np.mean(np.array(dumb_list), axis=0)
    #TODO dop mean of dumb result socan be plotted
    with open(save_dir + "/mean_results.txt", "w+") as outfile:
        outfile.write("Mean Results across 5-fold crossvalidation\n")
        outfile.write("== == == == == == == == == == == ==\n")
        for i in range(4):
            mae_list = []
            mse_list = []
            val_mae_list = []
            val_mse_list = []
            for hist in history_list:
                mae_list.append(hist[str(output_names[i]) + '_loss'])
                mse_list.append(hist[str(output_names[i]) + '_mean_squared_error'])
                val_mae_list.append(hist['val_' + str(output_names[i]) + '_loss'])
                val_mse_list.append(hist['val_' + str(output_names[i]) + '_mean_squared_error'])
            outfile.write(str(output_names[i]) + "_mae:          %.3f" % (np.mean(mae_list)) + '\n')
            outfile.write(str(output_names[i]) + "_mse:          %.3f" % (np.mean(mse_list)) + '\n')
            outfile.write("val_" + str(output_names[i]) + "_mae:      %.3f" % (np.mean(val_mae_list)) + '\n')
            outfile.write("val_" + str(output_names[i]) + "_mse:      %.3f" % (np.mean(val_mse_list)) + '\n')
            fig=plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
            mse = fig.add_subplot(121)
            mae = fig.add_subplot(122)

            mse.plot(np.mean(np.array(mse_list), axis=0))
            mse.plot(np.mean(np.array(val_mse_list), axis=0))
            mse.axhline(mean_dumb[i][0], c='r', ls='--')
            mse.set_title('mean '+str(output_names[i])+' MSE')
            mse.set_xlabel('epoch')
            mse.set_ylabel('error')
            mse.legend(['train', 'validation', 'mse dumb'], loc='upper right')

            mae.plot(np.mean(np.array(mae_list), axis=0))
            mae.plot(np.mean(np.array(val_mae_list), axis=0))
            mae.axhline(mean_dumb[i][1], c='r', ls='--')
            mae.set_title('mean '+str(output_names[i])+' loss(MAE)')
            mae.set_xlabel('epoch')
            mae.set_ylabel('MAE')
            mae.legend(['train', 'test', 'mae dumb'], loc='upper right')

            fig.savefig(save_dir + "/mean_"+str(output_names[i])+"_result.png")
        outfile.write("== == == == == == == == == == == ==\n")
        outfile.close()
