import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential


def model_creator(num_classes,show_summary=False):
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
    if show_summary:
        model.summary()

    return model
