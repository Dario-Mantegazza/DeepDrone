import os
from subprocess import call

import numpy as np
import keras
import pandas as pd
from sklearn.metrics import roc_auc_score
from gtts import gTTS
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential


def py_voice(text_to_speak="Computing Completed", l='en'):
    tts = gTTS(text=text_to_speak, lang=l)
    tts.save('voice.mp3')
    call(["cvlc", "voice.mp3", '--play-and-exit'])


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


py_voice("Rete treinata", l='it')
