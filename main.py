import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import shutil
import argparse

from model_run import Runner

np.random.seed(1234)
tf_data_type = tf.float64
tf.config.list_physical_devices("GPU")
tf.keras.backend.clear_session()


def main(save_index):
    # Create directories
    current_directory = os.getcwd()
    case = "Case_"
    folder_index = str(save_index)
    results_dir = "/" + case + folder_index + "/Results"
    variable_dir = "/" + case + folder_index + "/Variables"
    save_results_to = current_directory + results_dir
    save_variables_to = current_directory + variable_dir

    # Remove existing results
    if os.path.exists(save_results_to) or os.path.exists(save_variables_to):
        shutil.rmtree(save_results_to)
        shutil.rmtree(save_variables_to)

    os.makedirs(save_results_to)
    os.makedirs(save_variables_to)

    p = 80

    hyperparameters = {
        "n_channels": 2,
        "filter_size_1": 3,
        "filter_size_2": 3,
        "filter_size_3": 3,
        "filter_size_4": 3,
        "stride": 1,
        "num_filters_1": 40,
        "num_filters_2": 60,
        "num_filters_3": 100,
        "num_filters_4": 180,
        "B_net": [180, 80, p],
        "T_net": [2, 80, 80, p],
        "bs": 400,
        "tsbs": 100,
        "epochs": 1000,
    }

    io.savemat(save_variables_to + "/hyperparameters.mat", mdict=hyperparameters)

    # Initialise and run the model
    network = Runner(tf_data_type)
    network.run(hyperparameters, save_results_to, save_variables_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Case_", default=1, type=int, help="prior_sigma")
    args = parser.parse_args()
    Case = args.Case_
    main(Case)
