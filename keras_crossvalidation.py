import math
import os

import cv2
import keras
import numpy as np
import pandas as pd
import tqdm
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn import metrics


# Cnn method contains the definition, training, testing and plotting of the CNN model and dataset
def CNNMethod(batch_size, epochs, model_name, num_classes, save_dir, x_test, x_train, y_test, y_train):
    model = Sequential()
    model.add(Conv2D(10, (6, 6), padding='same', input_shape=(60, 107, 3), name="1_conv"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), name="1_pool"))
    model.add(Conv2D(15, (6, 6), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(20, (6, 6), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('linear'))

    opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

    model.compile(loss='mean_absolute_error',
                  optimizer=opt,
                  metrics=['mse'])
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True)

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)

    w_save_dir = os.path.join(os.getcwd(), 'saved_weights')
    w_name = 'keras_bebop_trained_weights.h5'
    if not os.path.isdir(w_save_dir):
        os.makedirs(w_save_dir)
    w_path = os.path.join(w_save_dir, w_name)
    model.save_weights(w_path)

    print('Saved trained model at %s ' % model_path)
    print('Saved trained weights at %s ' % w_path)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    y_pred = model.predict(x_test)
    print('Test loss:', scores[0])
    print('Test mse:', scores[1])

    r2 = metrics.r2_score(y_test, y_pred)
    print('Test r2:', r2)

    mean_y = np.mean(y_test)
    mean_array = np.full(y_test.shape, mean_y)
    mae = metrics.mean_absolute_error(y_test, mean_array)
    print("----- mean value regressor metric -----")
    print('Mean mae:', mae)

    # here the video are composed
    # y_train_pred = model.predict(x_train)
    #
    # vidcr_train = KerasVideoCreator(x_test=x_train, labels=y_train, preds=y_train_pred, title="./video/train_result.avi")
    # vidcr_train.video_plot_creator()

    vidcr_test = KerasVideoCreator(x_test=x_test, labels=y_test, preds=y_pred, title="./video/test_result.avi")
    vidcr_test.video_plot_creator()

    # show some plots
    plt.figure()
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model MSE')
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend(['train', 'validation'], loc='upper right')

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc='upper right')

    plt.figure()
    plt.plot(y_test[:, 1])
    plt.plot(y_pred[:, 1])
    plt.title('test-prediction angle')
    plt.xlabel('frame')
    plt.ylabel('value')
    plt.legend(['test', 'pred'], loc='upper right')

    plt.figure()
    plt.plot(y_test[:, 0])
    plt.plot(y_pred[:, 0])
    plt.title('test-prediction distance')
    plt.xlabel('frame')
    plt.ylabel('value')
    plt.legend(['test', 'pred'], loc='upper right')

    plt.figure()
    plt.plot(y_test[:, 2])
    plt.plot(y_pred[:, 2])
    plt.title('test-prediction delta z')
    plt.xlabel('frame')
    plt.ylabel('value')
    plt.legend(['test', 'pred'], loc='upper right')

    plt.figure()
    plt.scatter(y_test[:, 1], y_pred[:, 1])
    plt.title('scatter-plot angle')
    plt.xlabel('thruth')
    plt.ylabel('pred')

    plt.figure()
    plt.scatter(y_test[:, 0], y_pred[:, 0])
    plt.title('scatter-plot distance')
    plt.ylabel('pred')
    plt.xlabel('thruth')


    plt.figure()
    plt.scatter(y_test[:, 2], y_pred[:, 2])
    plt.title('scatter-plot delta z')
    plt.ylabel('pred')
    plt.xlabel('thruth')
    plt.show()


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
        y_d = self.preds[i]
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
        pr_center = (int((t_x + scale_factor * (math.sin(math.radians(self.preds[i, 1])) * self.preds[i, 0]))),
                     int((t_y - scale_factor * (math.cos(math.radians(self.preds[i, 1])) * self.preds[i, 0]))))
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
                       int((h_c_y - h_scale_factor * self.preds[i, 2])))
        cv2.circle(im_final, center=gt_h_center, radius=5, color=gt_color, thickness=2)
        cv2.circle(im_final, center=pr_h_center, radius=5, color=pr_color, thickness=5)
        self.video_writer.write(im_final)

    def video_plot_creator(self):
        max_ = len(self.frame_list)
        for i in tqdm.tqdm(range(0, max_)):
            self.frame_composer(i)
        self.video_writer.release()
        cv2.destroyAllWindows()


# ------------------- Main ----------------------
def main():
    train = pd.read_pickle("./dataset/train.pickle").values
    validation = pd.read_pickle("./dataset/validation.pickle").values

    batch_size = 64
    num_classes = 3
    epochs = 5

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_bebop_trained_model.h5'

    # The data, split between train and test sets:
    x_train = 1 - train[:, 0]  # otherwise is inverted
    x_train = np.vstack(x_train[:]).astype(np.float)
    x_train = np.reshape(x_train, (-1, 60, 107, 3))
    y_train = train[:, 1]
    y_train = np.asarray([np.asarray(sublist) for sublist in y_train])

    x_test = 1 - validation[:, 0]
    x_test = np.vstack(x_test[:]).astype(np.float)
    x_test = np.reshape(x_test, (-1, 60, 107, 3))
    y_test = validation[:, 1]
    y_test = np.asarray([np.asarray(sublist) for sublist in y_test])

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    CNNMethod(batch_size, epochs, model_name, num_classes, save_dir, x_test, x_train, y_test, y_train)


if __name__ == "__main__":
    main()
