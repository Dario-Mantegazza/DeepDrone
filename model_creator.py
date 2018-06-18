import keras

from keras.models import Sequential, Model
from keras.layers import *


def model_creator(num_classes, show_summary=False, old=False):
    if old:
        seq_model = create_sequential()
        model_input = Input((60, 107, 3))
        out_sequential = seq_model(model_input)
        y_1 = (Dense(1, activation='linear', name="distance_pred"))(out_sequential)
        y_2 = (Dense(1, activation='linear', name="angle_pred"))(out_sequential)
        y_3 = (Dense(1, activation='linear', name="height_pred"))(out_sequential)
        model = Model(inputs=model_input, outputs=[y_1, y_2, y_3])
        learn_rate = 0.001
        decay = 1e-6
        opt = keras.optimizers.rmsprop(lr=learn_rate, decay=decay)
        model.compile(loss='mean_absolute_error',
                      optimizer=opt,
                      metrics=['mse'])
        if show_summary:
            model.summary()
    else:  # NEW
        seq_model = create_sequential()
        model_input = Input((60, 107, 3))
        out_sequential = seq_model(model_input)
        y_1 = (Dense(1, activation='linear', name="x_pred"))(out_sequential)
        y_2 = (Dense(1, activation='linear', name="y_pred"))(out_sequential)
        y_3 = (Dense(1, activation='linear', name="z_pred"))(out_sequential)
        y_4 = (Dense(1, activation='linear', name="yaw_pred"))(out_sequential)
        model = Model(inputs=model_input, outputs=[y_1, y_2, y_3, y_4])
        learn_rate = 0.001
        decay = 1e-6
        opt = keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # opt = keras.optimizers.rmsprop(lr=learn_rate, decay=decay)
        model.compile(loss='mean_absolute_error',
                      optimizer=opt,
                      metrics=['mse'])
        if show_summary:
            model.summary()

    return model, learn_rate, decay


def create_sequential():
    model = Sequential()
    model.add(Conv2D(10, (6, 6), padding='same', input_shape=(60, 107, 3), name="1_conv"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), name="1_pool"))
    model.add(Conv2D(15, (6, 6), padding='same', name="2_conv"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name="2_pool"))
    model.add(Conv2D(20, (6, 6), padding='same', name="3_conv"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name="3_pool"))
    model.add(Flatten())
    model.add(Dense(256, name="1_dense"))
    model.add(Activation('relu'))
    model.add(Dense(64, name="2_dense"))
    model.add(Activation('relu'))
    return model


def data_augmentor(frame, label, noise=False):
    if np.random.choice([True, False]):
        frame = np.fliplr(frame)  # IMG
        label[1] = -label[1]  # Y
        label[3] = -label[3]  # Relative YAW
    return frame, label


def generator(features, labels, batch_size, old=False):
    if old:  # OLD
        while True:
            indexes = np.random.choice(np.arange(0, features.shape[0]), batch_size)
            batch_features = features[indexes]
            batch_labels = labels[indexes]
            for i in range(0, batch_features.shape[0]):
                batch_features[i], batch_labels[i] = data_augmentor(batch_features[i], batch_labels[i])
            yield batch_features, [batch_labels[:, 0], batch_labels[:, 1], batch_labels[:, 2]]
    else:  # NEW
        while True:
            indexes = np.random.choice(np.arange(0, features.shape[0]), batch_size)
            batch_features = features[indexes]
            batch_labels = labels[indexes]
            for i in range(0, batch_features.shape[0]):
                batch_features[i], batch_labels[i] = data_augmentor(batch_features[i], batch_labels[i])
            yield batch_features, [batch_labels[:, 0], batch_labels[:, 1], batch_labels[:, 2], batch_labels[:, 3]]

