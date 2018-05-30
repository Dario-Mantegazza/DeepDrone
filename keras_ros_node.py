from subprocess import call

import cv2
import keras
import numpy as np
from gtts import gTTS
from keras.models import Sequential
# !/usr/bin/env python
import rospy
import sys

import numpy as np
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from math import cos, sin, asin, tan, atan2, radians, atan, pow, atan2, sqrt
# msgs and srv for working with the set_model_service
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
# a handy tool to convert orientations
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf
# from rbx1_nav.transform_utils import quat_to_angle
import PyKDL

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
    def __init__(self, publisher_name):
        self.model = Sequential()
        self.num_classes = 1
        self.pub_name = publisher_name

    def setup(self):
        self.model = keras.models.load_model("./saved_models/keras_bebop_trained_model.h5")
        # self.model = Sequential()
        # self.model.add(Conv2D(2, (6, 6), padding='same', input_shape=(60, 107, 3)))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(3, 3)))
        # self.model.add(Conv2D(5, (6, 6), padding='same'))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Conv2D(10, (6, 6), padding='same'))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Flatten())
        # self.model.add(Dense(64))
        # self.model.add(Activation('relu'))
        # # model.add(Dropout(0.5))
        # self.model.add(Dense(16))
        # self.model.add(Activation('relu'))
        # # model.add(Dropout(0.5))
        # self.model.add(Dense(self.num_classes))
        # self.model.add(Activation('sigmoid'))
        # # initiate RMSprop optimizer
        # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        # # Let's train the model using RMSprop
        # self.model.compile(loss='mean_squared_error',
        #                    optimizer=opt,
        #                    metrics=['accuracy'])
        self.camera_feed = rospy.Subscriber('/bebop/image_raw/compressed', Range, self.check_transition) #cambiare tipo, cambiare funzione callback, testare
        self.camera_feed = rospy.Subscriber(self.pub_name + '/image_raw/compressed', Range, self.check_transition)

    # def check_transition(self, data):
    #     self.prox_sensors_measure[data.header.frame_id] = data.range
    #     if data.range < 0.06 and self.state == WALKING:
    #         self.state = TURNING
    #
    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        self.showResult(x_test, y_pred)

    def showResult(self, frame, y):
        img = (255 * frame).astype(np.uint8)
        scaled = cv2.resize(img, (0, 0), fx=2, fy=2)
        vert_p = 180
        hor_p = 213
        im_pad = cv2.copyMakeBorder(scaled, vert_p, vert_p, hor_p, hor_p, cv2.BORDER_CONSTANT, value=self.PADCOLOR)
        im_final = cv2.cvtColor(im_pad, cv2.COLOR_RGB2BGR)

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
        cv2.destroyAllWindows()


# -------------------Main area----------------------
def main():
    cnn = TrainedModel()
    cnn.setup()


if __name__ == "__main__":
    main()
