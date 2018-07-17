import math
import os

import numpy as np
import pandas as pd

from tool_to_plot_data import KerasVideoCreator, plot_results
from dumb_regressor import dumb_regressor_result
from model_creator import model_creator, generator
from global_parameters import *
from utils import isdebugging


# Cnn method contains the definition, training, testing and plotting of the CNN model and dataset
def CNNMethod(batch_size, epochs, model_name, save_dir, x_test, x_train, y_test, y_train):
    """
         Cnn method runs:
            -train
            -test
            -save model as a h5py file

    Args:
        batch_size: size of a batch
        epochs: number of epochs
        model_name: name of the model, used for naming saved models
        save_dir: directory of the running test folder
        x_test: validation samples
        x_train: training samples
        y_test: validation target
        y_train: training target

    Returns:
        history: metric history
        y_pred: prediction on test set
    """
    model, _, _ = model_creator(show_summary=True)
    batch_per_epoch = math.ceil(x_train.shape[0] / batch_size)
    gen = generator(x_train, y_train, batch_size)

    history = model.fit_generator(generator=gen,
                                  validation_data=(x_test, [y_test[:, 0], y_test[:, 1], y_test[:, 2], y_test[:, 3]]),
                                  epochs=epochs,
                                  steps_per_epoch=batch_per_epoch)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    scores = model.evaluate(x_test, [y_test[:, 0], y_test[:, 1], y_test[:, 2], y_test[:, 3]], verbose=1)

    y_pred = model.predict(x_test)
    print('Test loss:', scores[0])
    print('Test mse:', scores[1])
    return history, y_pred


# ------------------- Main ----------------------
def main():
    """
        -read pickle file for train and validation
        -calls method for train and predict
        -calls method to run dumb prediction
        -calls method to create a video for qualitative evaluation
        -calls method to plot data for quantitative evaluation
    """
    train = pd.read_pickle("./dataset/train.pickle").values
    validation = pd.read_pickle("./dataset/validation.pickle").values

    if isdebugging():
        print("debugging-settings")
        batch_size = 128
        epochs = 2
    else:
        batch_size = 64
        epochs = 100
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_bebop_trained_model.h5'
    # split between train and test sets:
    x_train = 255 - train[:, 0]  # otherwise is inverted
    x_train = np.vstack(x_train[:]).astype(np.float32)
    x_train = np.reshape(x_train, (-1, image_height, image_width, 3))
    y_train = train[:, 1]
    y_train = np.asarray([np.asarray(sublist) for sublist in y_train])

    x_test = 255 - validation[:, 0]
    x_test = np.vstack(x_test[:]).astype(np.float32)
    x_test = np.reshape(x_test, (-1, image_height, image_width, 3))
    y_test = validation[:, 1]
    y_test = np.asarray([np.asarray(sublist) for sublist in y_test])

    print('x_train shape: ' + str(x_train.shape))
    print('train samples: ' + str(x_train.shape[0]))
    print('test samples:  ' + str(x_test.shape[0]))

    history, y_pred = CNNMethod(batch_size, epochs, model_name, save_dir, x_test, x_train, y_test, y_train)
    dumb_metrics = dumb_regressor_result(x_test, x_train, y_test, y_train)
    if isdebugging():
        print("debugging-settings - no video")
    else:
        print("debugging-settings - no video")
        # vidcr_test = KerasVideoCreator(x_test=x_test, targets=y_test, preds=y_pred, title="./video/test_result.avi")
        # vidcr_test.video_plot_creator()
    plot_results(history, y_pred, y_test, dumb_metrics)


if __name__ == "__main__":
    main()
