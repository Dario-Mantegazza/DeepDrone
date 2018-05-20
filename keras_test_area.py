import io
import numpy as np
import os
import sys
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from subprocess import call

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from scipy import ndimage
import cv2
import keras
import pandas as pd
import tqdm as tqdm
from gtts import gTTS
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from sklearn.metrics import roc_auc_score




def py_voice(text_to_speak="Computing Completed", l='en'):
    tts = gTTS(text=text_to_speak, lang=l)
    tts.save('voice.mp3')
    call(["cvlc", "voice.mp3", '--play-and-exit'])


def CNNMethod(batch_size, epochs, model_name, num_classes, save_dir, x_test, x_train, y_test, y_train):
    model = Sequential()
    model.add(Conv2D(2, (6, 6), padding='same', input_shape=(60, 107, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(5, (6, 6), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(10, (6, 6), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(16))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))
    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # Let's train the model using RMSprop
    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['accuracy'])
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    y_pred = model.predict(x_test)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    print('Test AUC:', roc_auc_score(y_test.tolist(), y_pred.tolist()))
    py_voice("Rete treinata. Creazione video", l='it')
    vidcr = KerasVideoCreator(x_test=x_test, labels=y_test, preds=y_pred, title="./video/CNNresults.avi")
    vidcr.video_plot_creator()
    py_voice("Video validescion creato", l='it')


class KerasVideoCreator:
    def __init__(self, x_test, labels, preds, title="Validation.avi"):
        self.fps = 2
        self.width = 640
        self.height = 480
        self.video_writer = cv2.VideoWriter(title, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (self.width, self.height))
        self.frame_list = x_test
        self.labels = labels
        self.preds = preds
        self.PADCOLOR = [200, 200, 200]

    def plotting_function(self, i):
        img = (255 * self.frame_list[i]).astype(np.uint8)
        scaled = cv2.resize(img, (0, 0), fx=2, fy=2)
        vert_p = 180
        hor_p = 213
        im_pad = cv2.copyMakeBorder(scaled, vert_p, vert_p, hor_p, hor_p, cv2.BORDER_CONSTANT, value=self.PADCOLOR)
        im_final = cv2.cvtColor(im_pad, cv2.COLOR_RGB2BGR)

        pt1 = (275, 50)
        pt2 = (375, 50)
        if self.labels[i] > 0:
            cv2.arrowedLine(im_final, pt1, pt2, (0, 255, 0), 3)
        else:
            cv2.arrowedLine(im_final, pt2, pt1, (0, 255, 0), 3)

        pt1 = (275, 25)
        pt2 = (375, 25)
        if self.preds[i] >= 0.5:
            cv2.arrowedLine(im_final, pt1, pt2, (255, 0, 0), 3)
        else:
            cv2.arrowedLine(im_final, pt2, pt1, (255, 0, 0), 3)

        x_p = 213 + int(214 * self.preds[i])
        pt1_p = (x_p, 5)
        pt2_p = (x_p, 20)
        cv2.arrowedLine(im_final, pt1_p, pt2_p, (255, 0, 0), 3)

        self.video_writer.write(im_final)

    def video_plot_creator(self):
        max_ = len(self.frame_list)
        for i in tqdm.tqdm(range(0, max_)):
            self.plotting_function(i)
        self.video_writer.release()
        cv2.destroyAllWindows()


# -------------------Main area----------------------
def main():
    # sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

    train = pd.read_pickle("./dataset/train.pickle").values
    validation = pd.read_pickle("./dataset/validation.pickle").values

    batch_size = 32
    num_classes = 1
    epochs = 15
    # data_augmentation = True
    num_predictions = 20
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_bebop_trained_model.h5'

    # The data, split between train and test sets:
    x_train = train[:, 0]
    x_train = np.vstack(x_train[:]).astype(np.float)
    x_train = np.reshape(x_train, (-1, 60, 107, 3))
    y_train = train[:, 1]

    x_test = validation[:, 0]
    x_test = np.vstack(x_test[:]).astype(np.float)
    x_test = np.reshape(x_test, (-1, 60, 107, 3))
    y_test = validation[:, 1]
    # (x_test, y_test) = validation
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # print("asd: ", sum(y_test)/len(y_test))

    CNNMethod(batch_size, epochs, model_name, num_classes, save_dir, x_test, x_train, y_test, y_train)


if __name__ == "__main__":
    main()
