import os
from subprocess import call

import cv2
import keras
import numpy as np
import pandas as pd
import tqdm as tqdm
from gtts import gTTS
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.backend import clear_session
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt




def CNNMethod(batch_size, epochs, model_name, num_classes, save_dir, x_test, x_train, y_test, y_train):
    model = Sequential()
    model.add(Conv2D(5, (6, 6), padding='same', input_shape=(60, 107, 3), name="1_conv"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), name="1_pool"))
    model.add(Conv2D(10, (6, 6), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (6, 6), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(32))
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

    clear_session()
    del model  # deletes the existing model
    model = keras.models.load_model("./saved_models/keras_bebop_trained_model.h5")
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    y_pred = model.predict(x_test)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    print('Test AUC:', roc_auc_score(y_test.tolist(), y_pred.tolist()))

    vidcr = KerasVideoCreator(x_test=x_test, labels=y_test, preds=y_pred, title="./video/CNNresults.avi")
    vidcr.video_plot_creator()

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(y_test)
    plt.plot(y_pred)
    plt.title('test-prediction')
    plt.ylabel('value')
    plt.xlabel('frame')
    plt.legend(['test', 'pred'], loc='upper right')
    plt.show()

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

    def plotting_function(self, i):
        img = 1-(255 * self.frame_list[i]).astype(np.uint8)
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

def showResult(frame):
    img = (255 * (frame)).astype(np.uint8)
    # scaled = cv2.resize(img, (0, 0), fx=2, fy=2)
    im_final = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # im_final = cv2.resize(im_final, (640, 480))
    cv2.imshow("Display window", im_final)
    cv2.waitKey(1)

# -------------------Main area----------------------
def main():
    train = pd.read_pickle("./dataset/train.pickle").values
    validation = pd.read_pickle("./dataset/validation.pickle").values

    batch_size = 16
    num_classes = 1
    epochs = 10

    # data_augmentation = True
    num_predictions = 20
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_bebop_trained_model.h5'

    # The data, split between train and test sets:
    x_train = 1-train[:, 0] #otherwise is inverted
    x_train = np.vstack(x_train[:]).astype(np.float)
    x_train = np.reshape(x_train, (-1, 60, 107, 3))
    y_train = train[:, 1]

    x_test = 1-validation[:, 0]
    x_test = np.vstack(x_test[:]).astype(np.float)
    x_test = np.reshape(x_test, (-1, 60, 107, 3))
    # showResult(x_train[0])
    y_test = validation[:, 1]
    # (x_test, y_test) = validation
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # print("asd: ", sum(y_test)/len(y_test))

    CNNMethod(batch_size, epochs, model_name, num_classes, save_dir, x_test, x_train, y_test, y_train)


if __name__ == "__main__":
    main()
