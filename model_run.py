import tensorflow as tf
import numpy as np
import time
import scipy.io as io

from dataset import DataSet
from DeepONet import DeepONet
from model_train import Train_Adam
from model_error import Error_Test
from model_plot import Plot


class Runner:
    def __init__(self, tf_data_type):
        self.tf_data_type = tf_data_type

    def run(self, hyperparameters, save_results_to, save_variables_to):
        """
        Runs the model.

        Args:
            hyperparameters (Dict): Dictionary of hyperparameters.
            save_results_to (str): Directory to save results to.
            save_variables_to (str): Directory to save variables to.
        """
        filter_size_1 = hyperparameters["filter_size_1"]
        filter_size_2 = hyperparameters["filter_size_2"]
        filter_size_3 = hyperparameters["filter_size_3"]
        filter_size_4 = hyperparameters["filter_size_4"]
        num_filters_1 = hyperparameters["num_filters_1"]
        num_filters_2 = hyperparameters["num_filters_2"]
        num_filters_3 = hyperparameters["num_filters_3"]
        num_filters_4 = hyperparameters["num_filters_4"]
        n_channels = hyperparameters["n_channels"]
        stride = hyperparameters["stride"]
        B_net = hyperparameters["B_net"]
        T_net = hyperparameters["T_net"]
        bs = hyperparameters["bs"]
        tsbs = hyperparameters["tsbs"]
        epochs = hyperparameters["epochs"]

        param = DataSet(bs, tsbs)
        model = DeepONet(self.tf_data_type)

        # Branch CNN initialisation
        W_br_1, b_br_1 = model.hyper_initial_cnn([filter_size_1, filter_size_1, n_channels, num_filters_1], num_filters_1)
        W_br_2, b_br_2 = model.hyper_initial_cnn([filter_size_2, filter_size_2, num_filters_1, num_filters_2], num_filters_2)
        W_br_3, b_br_3 = model.hyper_initial_cnn([filter_size_3, filter_size_3, num_filters_2, num_filters_3], num_filters_3)
        W_br_4, b_br_4 = model.hyper_initial_cnn([filter_size_4, filter_size_4, num_filters_3, num_filters_4], num_filters_4)
        
        # output dimension of Branch/Trunk (latent dimension)
        var = [8820] # Need to Automate this
        B_net = var + B_net

        # Branch FNN initialisation
        W_br_fnn, b_br_fnn = model.hyper_initial_fnn(B_net)

        # Trunk initialisation
        W_tr, b_tr = model.hyper_initial_fnn(T_net)

        W_b_dict = {
            "W_br1": W_br_1,
            "b_br1": b_br_1,
            "W_br2": W_br_2,
            "b_br2": b_br_2,
            "W_br3": W_br_3,
            "b_br3": b_br_3,
            "W_br4": W_br_4,
            "b_br4": b_br_4,
            "W_br_fnn": W_br_fnn,
            "b_br_fnn": b_br_fnn,
            "W_tr": W_tr,
            "b_tr": b_tr,
        }

        Train_Model_Adam = Train_Adam(model, self.tf_data_type, B_net, T_net, W_b_dict, hyperparameters)
        Test_error = Error_Test(W_b_dict, model, self.tf_data_type, save_results_to)
        optimiser = tf.keras.optimizers.Adam()

        n = 0
        start_time = time.perf_counter()
        time_step_0 = time.perf_counter()

        train_loss = np.zeros((epochs + 1, 1))
        test_loss = np.zeros((epochs + 1, 1))
        while n <= epochs:
            x_train, f_train, u_train, f_norm_train, Xmin, Xmax = param.minibatch()
            lr = 0.0001
            optimiser.lr.assign(lr)
            train_dict, train_W_b_dict = Train_Model_Adam.nn_train(
                optimiser, x_train, f_train, u_train, f_norm_train, Xmin, Xmax
            )
            loss = train_dict["loss"]

            if n % 50 == 0:
                x_test, f_test, u_test, f_norm_test, batch_id = param.testbatch()
                u_pred = Train_Model_Adam.call(x_test, f_test, f_norm_test, Xmin, Xmax)
                # batch_id, f_test, u_test, u_pred = Test_error.nn_error_test(x_test, f_test, u_test, f_norm_test, stride, tsbs, Xmin, Xmax, batch_id, param)
                # u_pred = param.decoder(u_pred)
                # u_test = param.decoder(u_test)
                err = np.mean((u_test - u_pred) ** 2 / (u_test**2 + 1e-4))
                err = np.reshape(err, (-1, 1))
                time_step_1000 = time.perf_counter()
                T = time_step_1000 - time_step_0
                print(
                    "Step: %d, Loss: %.4e, Test L2 error: %.4f, Time (secs): %.4f"
                    % (n, loss, err, T)
                )
                time_step_0 = time.perf_counter()

            train_loss[n, 0] = loss
            test_loss[n, 0] = err
            n += 1

        x_print, f_print, u_print, f_norm_print, batch_id = param.printbatch()
        batch_id, f_print, u_print, u_pred = \
            Test_error.nn_error_test(x_print, f_print, u_print, f_norm_print, stride, Xmin, Xmax, batch_id)
        err = np.mean((u_print - u_pred) ** 2 / (u_print**2 + 1e-4))
        err = np.reshape(err, (-1, 1))
        np.savetxt(save_results_to + "/err", err, fmt="%e")
        io.savemat(
            save_results_to + "/Darcy.mat",
            mdict={
                "test_id": batch_id,
                "x_test": f_print,
                "y_test": u_print,
                "y_pred": u_pred,
            },
        )
        stop_time = time.perf_counter()
        print("Elapsed time (secs): %.3f" % (stop_time - start_time))

        # Save variables (weights + biases)
        W_br_1_save, b_br_1_save = model.save_W_b(train_W_b_dict["W_br1"], train_W_b_dict["b_br1"])
        W_br_2_save, b_br_2_save = model.save_W_b(train_W_b_dict["W_br2"], train_W_b_dict["b_br2"])
        W_br_3_save, b_br_3_save = model.save_W_b(train_W_b_dict["W_br3"], train_W_b_dict["b_br3"])
        W_br_4_save, b_br_4_save = model.save_W_b(train_W_b_dict["W_br4"], train_W_b_dict["b_br4"])
        W_br_fnn_save, b_br_fnn_save = model.save_W_b(train_W_b_dict["W_br_fnn"], train_W_b_dict["b_br_fnn"])
        W_tr_save, b_tr_save = model.save_W_b(train_W_b_dict["W_tr"], train_W_b_dict["b_tr"])

        W_b_dict_save = {
            "W_br1": W_br_1_save,
            "b_br1": b_br_1_save,
            "W_br2": W_br_2_save,
            "b_br2": b_br_2_save,
            "W_br3": W_br_3_save,
            "b_br3": b_br_3_save,
            "W_br4": W_br_4_save,
            "b_br4": b_br_4_save,
            "W_br_fnn": W_br_fnn_save,
            "b_br_fnn": b_br_fnn_save,
            "W_tr": W_tr_save,
            "b_tr": b_tr_save,
        }

        io.savemat(save_variables_to + "/Weight_bias.mat", W_b_dict_save)
        print("Complete storing")

        print("Save")

        plot = Plot()
        plot.Plotting(train_loss, test_loss, save_results_to)
