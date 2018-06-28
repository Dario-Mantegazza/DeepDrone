import keras

from keras.models import Sequential, Model
from keras.layers import *
from global_parameters import *


def model_creator(show_summary=False, old=False):
    """
        Generate the model.
        Available two architcture
            -old
            -new
    Args:
        show_summary: if true, show keras model summary in console
        old: if true create old architecture

    Returns:

    """
    if old:
        seq_model = create_sequential()
        model_input = Input((image_height, image_width, 3))
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
        model_input = Input((image_height, image_width, 3))
        out_sequential = seq_model(model_input)
        y_1 = (Dense(1, activation='linear', name="x_pred"))(out_sequential)
        y_2 = (Dense(1, activation='linear', name="y_pred"))(out_sequential)
        y_3 = (Dense(1, activation='linear', name="z_pred"))(out_sequential)
        y_4 = (Dense(1, activation='linear', name="yaw_pred"))(out_sequential)
        model = Model(inputs=model_input, outputs=[y_1, y_2, y_3, y_4])
        learn_rate = 0.00005
        opt = keras.optimizers.Adam(lr=learn_rate)
        model.compile(loss='mean_absolute_error',
                      optimizer=opt,
                      metrics=['mse'])
        if show_summary:
            model.summary()

    return model, learn_rate, 0


def create_sequential():
    """
        Create sequential part of the model architecture
    Returns:
        model: keras Sequential object
    """
    model = Sequential()
    model.add(Conv2D(10, (6, 6), padding='same', input_shape=(image_height, image_width, 3), name="1_conv"))
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


def data_augmentor(frame, target):
    """
        recieves frame and targets with p=50% flip vertically
    Args:
        frame: image
        target:  target for CNN

    Returns:
        frame and targers eventually flipped
    """
    if np.random.choice([True, False]):
        frame = np.fliplr(frame)  # IMG
        target[1] = -target[1]  # Y
        target[3] = -target[3]  # Relative YAW
    return frame, target


def generator(samples, targets, batch_size, old=False):
    """
        Genereator of minibatches of size batch_size
    Args:
        samples: sample array
        targets: targets array
        batch_size: batch size
        old: if true genereate data for old architecture
    Yields:
        batch of samples and array of batch of targets
    """
    if old:  # OLD
        while True:
            indexes = np.random.choice(np.arange(0, samples.shape[0]), batch_size)
            batch_samples = samples[indexes]
            batch_targets = targets[indexes]
            for i in range(0, batch_samples.shape[0]):
                batch_samples[i], batch_targets[i] = data_augmentor(batch_samples[i], batch_targets[i])
            yield batch_samples, [batch_targets[:, 0], batch_targets[:, 1], batch_targets[:, 2]]
    else:  # NEW
        while True:
            indexes = np.random.choice(np.arange(0, samples.shape[0]), batch_size)
            batch_samples = samples[indexes]
            batch_targets = targets[indexes]
            for i in range(0, batch_samples.shape[0]):
                batch_samples[i], batch_targets[i] = data_augmentor(batch_samples[i], batch_targets[i])
            yield batch_samples, [batch_targets[:, 0], batch_targets[:, 1], batch_targets[:, 2], batch_targets[:, 3]]

