import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
from keras.utils import plot_model

from dumb_regressor import dumb_regressor_result
from model_creator import model_creator, generator
from tool_to_plot_data import history_data_plot_crossvalidation, plot_results_cross, KerasVideoCreator

pickle_sections = {
    "1": 1,
    "2": 0,
    "3": 3,
    "4": 4,
    "5": 4,
    "6": 3,
    "7": 2,
    "8": 0,
    "9": 1,
    "10": 3,
    "11": 4,
    "12": 2,
    "13": 3,
    "14": 4,
    "15": 1,
    "16": 0,
    "17": 2,
    "18": 2,
    "19": 2,
    "20": 1,
    "21": 0,
    "22": 4
}


# Cnn method contains the definition, training, testing and plotting of the CNN model and dataset
def CNNMethod(batch_size, epochs, model_name, num_classes, save_dir, x_test, x_train, y_test, y_train, i):
    print("k-fold:" + str(i))
    model, lr, _ = model_creator(num_classes, show_summary=True)
    if i == 0:
        # plot_model(model.layers[1], to_file=save_dir + '/model_seq.png')
        # plot_model(model, to_file=save_dir + '/model_out.png')
        with open(save_dir + "/model_info.txt", "w+") as outfile:
            outfile.write("Hyperparameters\n")
            outfile.write("== == == == == == == == == == == ==\n")
            outfile.write("learning_rate:" + str(lr) + "\n")
            outfile.write("batch size:" + str(batch_size) + "\n")
            outfile.write("epochs:" + str(epochs) + "\n")
            outfile.write("== == == == == == == == == == == ==\n")
            model.layers[1].summary(print_fn=lambda x: outfile.write(x + '\n'))
            model.summary(print_fn=lambda x: outfile.write(x + '\n'))
            outfile.close()
    batch_per_epoch = math.ceil(x_train.shape[0] / batch_size)
    gen = generator(x_train, y_train, batch_size)
    history = model.fit_generator(generator=gen,
                                  validation_data=(x_test, [y_test[:, 0], y_test[:, 1], y_test[:, 2], y_test[:, 3]]),
                                  epochs=epochs,
                                  steps_per_epoch=batch_per_epoch)

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(x_test, [y_test[:, 0], y_test[:, 1], y_test[:, 2], y_test[:, 3]], verbose=1)
    y_pred = model.predict(x_test)
    print('Test loss:', scores[0])
    print('Test mse:', scores[1])

    vidcr_test = KerasVideoCreator(x_test=x_test, labels=y_test, preds=y_pred, title=save_dir + "/result_model_" + str(i) + "/test_result.avi")
    vidcr_test.video_plot_creator()
    dumb_metrics = dumb_regressor_result(x_test, x_train, y_test, y_train)
    plot_results_cross(history, y_pred, y_test, dumb_metrics, save_dir, i)
    return history.history, dumb_metrics


def crossValidation(k_fold, batch_size, num_classes, epochs):
    start_time = datetime.now()
    save_path = 'saves/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    try:
        os.makedirs(save_path)
    except OSError:
        if not os.path.isdir(save_path):
            raise
    for i in range(k_fold):
        try:
            os.makedirs(save_path + "/result_model_" + str(i))
        except OSError:
            if not os.path.isdir(save_path + "/result_model_" + str(i)):
                raise
    path = "./dataset/crossvalidation/"
    files = [f for f in os.listdir(path) if f[-7:] == '.pickle']
    dumb_list = []
    history_list = []
    save_dir = os.path.join(os.getcwd(), save_path)
    if not files:
        print('No bag files found!')
        return None
    for i in range(k_fold):  # test selection,
        e_start_time = datetime.now()
        x_test_list = []
        x_train_list = []

        # create test and train set
        for f in files:  # train selection
            section = pickle_sections[f[:-7]]

            if section == i:
                x_test_list.append(pd.read_pickle("./dataset/crossvalidation/" + f))

            else:
                x_train_list.append(pd.read_pickle("./dataset/crossvalidation/" + f))
        train = pd.concat(x_train_list).values
        validation = pd.concat(x_test_list).values

        model_name = 'keras_bebop_trained_model_' + str(i) + '.h5'
        x_train = 255 - train[:, 0]  # otherwise is inverted
        x_train = np.vstack(x_train[:]).astype(np.float32)
        x_train = np.reshape(x_train, (-1, 60, 107, 3))
        y_train = train[:, 1]
        y_train = np.asarray([np.asarray(sublist) for sublist in y_train])
        x_test = 255 - validation[:, 0]
        x_test = np.vstack(x_test[:]).astype(np.float32)
        x_test = np.reshape(x_test, (-1, 60, 107, 3))
        y_test = validation[:, 1]
        y_test = np.asarray([np.asarray(sublist) for sublist in y_test])
        print('x_train shape: ' + str(x_train.shape))
        print('train samples: ' + str(x_train.shape[0]))
        print('test samples:  ' + str(x_test.shape[0]))
        history, dumb_results = CNNMethod(batch_size, epochs, model_name, num_classes, save_dir, x_test, x_train, y_test, y_train, i)
        history_list.append(history)
        dumb_list.append(dumb_results)
        e_end_time = datetime.now()
        print("")
        print("")
        print("----k-Fold " + str(i) + " info:")
        print("started: " + e_start_time.strftime("%Y-%m-%d-%H-%M-%S"))
        print("ended:   " + e_end_time.strftime("%Y-%m-%d-%H-%M-%S"))
        h = divmod((e_end_time - e_start_time).total_seconds(), 3600)  # hours
        m = divmod(h[1], 60)  # minutes
        s = m[1]  # seconds
        print("lasted:  %d hours, %d minutes, %d seconds" % (h[0], m[0], s))
        if i < 4:
            eta_h = divmod((k_fold - i - 1) * (e_end_time - e_start_time).total_seconds(), 3600)  # hours
            eta_m = divmod(eta_h[1], 60)  # minutes
            eta_s = eta_m[1]  # seconds
            print("ETA:     %d hours, %d minutes, %d seconds" % (eta_h[0], eta_m[0], eta_s))
        print("----")
        print("")
        print("")
        with open(save_dir + "/computation_time.txt", "w+") as outfile:
            outfile.write("----k-Fold " + str(i) + " info:")
            outfile.write("started: " + e_start_time.strftime("%Y-%m-%d-%H-%M-%S"))
            outfile.write("ended:   " + e_end_time.strftime("%Y-%m-%d-%H-%M-%S"))
            outfile.write("lasted:  %d hours, %d minutes, %d seconds" % (h[0], m[0], s))
            outfile.write("----")
            outfile.close()
    hist_df = pd.DataFrame(history_list)
    pickle_path = save_dir + "/metrics_history.pickle"
    hist_df.to_pickle(pickle_path)
    print("history saved: " + pickle_path)
    history_data_plot_crossvalidation(history_list, dumb_list, save_dir)
    end_time = datetime.now()
    print("")
    print("")
    print("---- total computation time:")
    print("started: " + start_time.strftime("%Y-%m-%d-%H-%M-%S"))
    print("ended:   " + end_time.strftime("%Y-%m-%d-%H-%M-%S"))
    h = divmod((end_time - start_time).total_seconds(), 3600)  # hours
    m = divmod(h[1], 60)  # minutes
    s = m[1]  # seconds
    print("lasted:  %d hours, %d minutes, %d seconds" % (h[0], m[0], s))
    print("----")
    with open(save_dir + "/computation_time.txt", "w+") as outfile:
        outfile.write("---- total computation time:")
        outfile.write("started: " + start_time.strftime("%Y-%m-%d-%H-%M-%S"))
        outfile.write("ended:   " + end_time.strftime("%Y-%m-%d-%H-%M-%S"))
        outfile.write("lasted:  %d hours, %d minutes, %d seconds" % (h[0], m[0], s))
        outfile.write("----")
        outfile.close()


# ------------------- Main ----------------------
def main():
    k_fold = 5
    batch_size = 64
    num_classes = 4
    epochs = 40
    # epochs = 2
    crossValidation(k_fold, batch_size, num_classes, epochs)


if __name__ == "__main__":
    main()
