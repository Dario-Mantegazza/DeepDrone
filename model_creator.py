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


# from keras.models import Model
# from keras.layers import *
#
# #inp is a "tensor", that can be passed when calling other layers to produce an output
# inp = Input((10,)) #supposing you have ten numeric values as input
#
#
# #here, SomeLayer() is defining a layer,
# #and calling it with (inp) produces the output tensor x
# x = SomeLayer(blablabla)(inp)
# x = SomeOtherLayer(blablabla)(x) #here, I just replace x, because this intermediate output is not interesting to keep
#
#
# #here, I want to keep the two different outputs for defining the model
# #notice that both left and right are called with the same input x, creating a fork
# out1 = LeftSideLastLayer(balbalba)(x)
# out2 = RightSideLastLayer(banblabala)(x)
#
#
# #here, you define which path you will follow in the graph you've drawn with layers
# #notice the two outputs passed in a list, telling the model I want it to have two outputs.
# model = Model(inp, [out1,out2])
# model.compile(optimizer = ...., loss = ....) #loss can be one for both sides or a list with different loss functions for out1 and out2
#
# model.fit(inputData,[outputYLeft, outputYRight], epochs=..., batch_size=...)