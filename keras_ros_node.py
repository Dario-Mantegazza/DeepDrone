# --------------------IMPORT-------------------
import io
import sys
# !/usr/bin/env python
import time
from PIL import Image
from subprocess import call

import cv2
import keras
import numpy as np
import rospy
import tensorflow as tf
from gtts import gTTS
from keras.backend import clear_session
from keras.models import Sequential
from numpy import array
from sensor_msgs.msg import CompressedImage

import threading

distance_tolerance = 0.5
tolerance = 0.001
angle_tolerance = 0.01
max_linear_speed = 0.2
vel_P = 1
vel_I = 0
vel_D = 0

ang_P = 2
ang_I = 0
ang_D = 0

point_P = 10
point_I = 0
point_D = 0


def py_voice(text_to_speak="Computing Completed", l='en'):
    tts = gTTS(text=text_to_speak, lang=l)
    tts.save('voice.mp3')
    call(["cvlc", "voice.mp3", '--play-and-exit'])


class PID:

    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.last_e = None
        self.sum_e = 0

    def step(self, e, dt):
        """ dt should be the time interval from the last method call """
        if (self.last_e is not None):
            derivative = (e - self.last_e) / dt
        else:
            derivative = 0
        self.last_e = e
        self.sum_e += e * dt
        return self.Kp * e + self.Kd * derivative + self.Ki * self.sum_e


class TrainedModel:
    def __init__(self):
        self.model = Sequential()
        self.num_classes = 1
        self.pub_name = 'bebop'
        self.hz = 30.0
        rospy.init_node('magic_node', anonymous=True)
        self.rate = rospy.Rate(self.hz)
        self.PADCOLOR = [200, 200, 200]

    def setup(self):
        clear_session()
        del self.model  # deletes the existing model
        self.model = keras.models.load_model("./saved_models/keras_bebop_trained_model.h5")
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        self.model.compile(loss='mean_squared_error',
                                              optimizer=opt,
                                              metrics=['accuracy'])
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        self.camera_feed = rospy.Subscriber(self.pub_name + '/image_raw/compressed', CompressedImage, self.predict_)


    def predict_(self, msg_data):
        data = 1-array(Image.open(io.BytesIO(msg_data.data)))
        scaled_fr = cv2.resize(data, (107, 60))
        x_data = np.vstack(scaled_fr[:]).astype(np.float)
        x_data = np.reshape(x_data, (-1, 60, 107, 3))
        with self.graph.as_default():
            y_pred = self.model.predict(x_data)
        self.showResult(x_data[0], y_pred)
        # self.showResult(data, y_pred)

    def showResult(self, frame, y):
        img = (255 * (frame)).astype(np.uint8)

        scaled = cv2.resize(img, (0, 0), fx=2, fy=2)
        vert_p = 180
        hor_p = 213
        im_pad = cv2.copyMakeBorder(scaled, vert_p, vert_p, hor_p, hor_p, cv2.BORDER_CONSTANT, value=self.PADCOLOR)
        im_final = cv2.cvtColor(im_pad, cv2.COLOR_RGB2BGR)
        im_final = cv2.resize(im_final, (640, 480))

        pt1 = (275, 25)
        pt2 = (375, 25)
        if y >= 0.5:
            cv2.arrowedLine(im_final, pt1, pt2, (255, 0, 0), 3)
        else:
            cv2.arrowedLine(im_final, pt2, pt1, (255, 0, 0), 3)
        x_p = 213 + int(214 * y)
        pt1_p = (x_p, 5)
        pt2_p = (x_p, 20)
        cv2.arrowedLine(im_final, pt1_p, pt2_p, (255, 0, 0), 3)
        cv2.imshow("Display window", im_final)
        cv2.waitKey(1)


    def main_cycle(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

# -------------------Main area----------------------
def main():
    cnn = TrainedModel()
    cnn.setup()
    while not rospy.is_shutdown():
        cnn.main_cycle()
        sys.exit(0)


if __name__ == "__main__":
    main()
