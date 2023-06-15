import matplotlib.pyplot as plt
import numpy as np


class Plot:
    def __init__(self):
        pass

    def Plotting(self, train_loss, test_loss, save_results_to):
        """
        Plots the loss history.

        Args:
            train_loss (list): List of training loss.
            test_loss (list): List of testing loss.
            save_results_to (str): Directory to save results to.
        """
        plt.rcParams.update({"font.size": 15})
        num_epoch = train_loss.shape[0]
        x = np.linspace(1, num_epoch, num_epoch)
        fig = plt.figure(constrained_layout=True, figsize=(7, 5))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, train_loss[:, 0], color="blue", label="Training Loss")
        ax.set_yscale("log")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend(loc="upper left")
        fig.savefig(save_results_to + "/loss_train.png")

        ## Save test loss
        np.savetxt(save_results_to + "/loss_test", test_loss[:, 0])
        np.savetxt(save_results_to + "/epochs", x)

        fig = plt.figure(constrained_layout=True, figsize=(7, 5))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, test_loss[:, 0], color="red", label="Testing Loss")
        ax.set_yscale("log")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend(loc="upper left")
        fig.savefig(save_results_to + "/loss_test.png")

        ########## NOT LOG PlOTS
        plt.rcParams.update({"font.size": 15})
        num_epoch = train_loss.shape[0]
        x = np.linspace(1, num_epoch, num_epoch)
        fig = plt.figure(constrained_layout=True, figsize=(7, 5))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, train_loss[:, 0], color="blue", label="Training Loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend(loc="upper left")
        fig.savefig(save_results_to + "/loss_train_notlog.png")

        fig = plt.figure(constrained_layout=True, figsize=(7, 5))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.plot(x, test_loss[:, 0], color="red", label="Testing Loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend(loc="upper left")
        fig.savefig(save_results_to + "/loss_test_notlog.png")
