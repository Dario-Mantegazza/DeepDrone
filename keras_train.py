import math
import os
from matplotlib import pyplot as plt

import cv2
import keras
import numpy as np
import pandas as pd
import tqdm as tqdm
from keras.backend import clear_session
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
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
    #
    #
    # clear_session()
    # del model  # deletes the existing model
    # model = keras.models.load_model("./saved_models/keras_bebop_trained_model.h5")

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
        self.width = 640
        self.height = 480
        self.video_writer = cv2.VideoWriter(title, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (self.width, self.height))
        self.frame_list = x_test
        self.labels = labels
        self.preds = preds
        self.PADCOLOR = [200, 200, 200]

    # function used to compose the frame
    def frame_composer(self, i):
        # Adjusting the image
        img = 1 - (self.frame_list[i]).astype(np.uint8)
        scaled = cv2.resize(img, (0, 0), fx=2, fy=2)
        vert_p = 180
        hor_p = 213
        im_pad = cv2.copyMakeBorder(scaled, vert_p, vert_p, hor_p, hor_p, cv2.BORDER_CONSTANT, value=self.PADCOLOR)
        im_final = cv2.cvtColor(im_pad, cv2.COLOR_RGB2BGR)

        # Setting some variables
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_d = self.preds[i]
        l_d = self.labels[i]
        arrow_len = 40
        scale_arrow = 50

        # Top view
        cv2.putText(im_final, "Top View", (35, 90), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        c_t_x = 55  # center top view x
        c_t_y = 170  # center top view y
        drone_top_view = (c_t_x, c_t_y)
        angle_deg = y_d[1]
        angle_rad = math.radians(angle_deg - 90.0)
        cv2.circle(im_final, center=drone_top_view, radius=2, color=(0, 0, 0), thickness=3)
        cv2.arrowedLine(im_final, drone_top_view, (int(c_t_x + arrow_len * np.cos(angle_rad)), int(c_t_y + arrow_len * np.sin(angle_rad))), (255, 0, 0), 1)  # heading arrow
        cv2.arrowedLine(im_final, drone_top_view, (c_t_x, int(c_t_y + scale_arrow * (1.437 - y_d[0]))), (0, 255, 0), 1)  # distance arrow

        # Right side View
        arrow_r_len = 20
        vertical_scale = 20
        cv2.putText(im_final, "Right side View", (35, 300), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        c_r_x = 55
        c_r_y = 280
        drone_right_center = (c_r_x, c_r_y)
        cv2.circle(im_final, center=drone_right_center, radius=2, color=(0, 0, 0), thickness=3)
        cv2.arrowedLine(im_final, (c_r_x, c_r_y + int(vertical_scale * y_d[2])), (c_r_x + arrow_r_len, c_r_y + int(vertical_scale * y_d[2])), (255, 0, 0), 1)  # delta x arrow
        cv2.arrowedLine(im_final, drone_right_center, (int(c_r_x + scale_arrow * (y_d[0] - 1.437)), c_r_y), (0, 255, 0), 1)  # distance arrow

        # Text Information
        cv2.putText(im_final, "Distance T: %.3f" % (l_d[0]), (15, 15), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(im_final, "Distance P: %.3f" % (y_d[0]), (15, 35), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(im_final, "Angle T: %.3f" % (l_d[1]), (170, 15), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(im_final, "Angle P: %.3f" % angle_deg, (170, 35), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(im_final, "Delta z T: %.3f" % (l_d[2]), (330, 15), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(im_final, "Delta z P: %.3f" % (y_d[2]), (330, 35), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        self.video_writer.write(im_final)

    def video_plot_creator(self):
        max_ = len(self.frame_list)
        for i in tqdm.tqdm(range(0, max_)):
            self.frame_composer(i)
        self.video_writer.release()
        cv2.destroyAllWindows()


# method not used, useful to have a peek at the images anywhere in the code
def showResult(frame):
    img = (255 * frame).astype(np.uint8)
    scaled = cv2.resize(img, (0, 0), fx=2, fy=2)
    im_final = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # im_final = cv2.resize(im_final, (640, 480))
    cv2.imshow("Display window", im_final)
    cv2.waitKey(1)


# ------------------- Main ----------------------
def main():
    train = pd.read_pickle("./dataset/train.pickle").values
    validation = pd.read_pickle("./dataset/validation.pickle").values

    batch_size = 64
    num_classes = 3
    epochs = 1

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
