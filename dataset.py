import numpy as np
import scipy.io as io

np.random.seed(1234)


class DataSet:
    def __init__(self, bs, test_bs):
        self.bs = bs
        self.test_bs = test_bs
        self.F_train, self.U_train, self.F_test, self.U_test, self.X,\
            self.u_mean, self.u_std, self.f_norm_train, self.f_norm_test = self.load_data()

    def decoder(self, x):
        """
        Decoder for the output of the network.

        Args:
            x (ndarray): Input to the decoder.

        Returns:
            ndarray: Decoded output.
        """
        x = x * (self.u_std + 1.0e-9) + self.u_mean
        return x

    def load_data(self):
        """
        Loads the data from the .mat files.

        Returns:
            Tuple: tuple of training and testing data.
        """
        file = io.loadmat("./Data/Dataset_square")

        k_train = file["k_train"]
        u_train = file["u_train"]
        f_train = file["shape_train"]

        k_test = file["k_test"]
        u_test = file["u_test"]
        f_test = file["shape_test"]

        # file = io.loadmat('./Data/Dataset_rightTriangle')

        # k2_train = file['k_train']
        # u2_train = file['u_train']
        # f2_train = file['shape_train']

        # k2_test = file['k_test']
        # u2_test = file['u_test']
        # f2_test = file['shape_test']

        # file = io.loadmat('./Data/Dataset_eqTri_notch')

        # k3_train = file['k_train']
        # u3_train = file['u_train']
        # f3_train = file['shape_train']

        # k3_test = file['k_test']
        # u3_test = file['u_test']
        # f3_test = file['shape_test']

        # file = io.loadmat('./Data/Dataset_equilateralTri')

        # k4_train = file['k_train']
        # u4_train = file['u_train']
        # f4_train = file['shape_train']

        # k4_test = file['k_test']
        # u4_test = file['u_test']
        # f4_test = file['shape_test']

        # k_train = np.concatenate((k1_train, k2_train, k3_train, k4_train), axis = 0)
        # k_test = np.concatenate((k1_test, k2_test, k3_test, k4_test), axis = 0)

        # f_train = np.concatenate((f1_train, f2_train, f3_train, f4_train), axis = 0)
        # f_test = np.concatenate((f1_test, f2_test, f3_test, f4_test), axis = 0)

        # io.savemat(self.save_results+'/K.mat',
        #              mdict={'k_train': k_train,
        #                     'k_test': k_test,
        #                     'f_train': f_train,
        #                     'f_test': f_test})

        # u_train = np.concatenate((u1_train, u2_train, u3_train, u4_train), axis = 0)
        # u_test = np.concatenate((u1_test, u2_test, u3_test, u4_test), axis = 0)

        k_train = np.log(k_train)
        k_test = np.log(k_test)

        xx = file["xx"]
        yy = file["yy"]
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        X = np.hstack((xx, yy))

        s = 100
        r = s * s

        k_train_mean = np.mean(np.reshape(k_train, (-1, s, s)), 0)
        k_train_std = np.std(np.reshape(k_train, (-1, s, s)), 0)
        k_train_mean = np.reshape(k_train_mean, (-1, s, s, 1))
        k_train_std = np.reshape(k_train_std, (-1, s, s, 1))
        k_train = np.reshape(k_train, (-1, s, s, 1))
        k_train = (k_train - k_train_mean) / (k_train_std)
        k_test = np.reshape(k_test, (-1, s, s, 1))
        k_test = (k_test - k_train_mean) / (k_train_std)

        f_train = np.reshape(f_train, (-1, s, s, 1))
        f_test = np.reshape(f_test, (-1, s, s, 1))

        F_train = np.concatenate((k_train, f_train), axis=-1)
        F_test = np.concatenate((k_test, f_test), axis=-1)

        u_train_mean = np.mean(np.reshape(u_train, (-1, r)), 0)
        u_train_std = np.std(np.reshape(u_train, (-1, r)), 0)
        u_train_mean = np.reshape(u_train_mean, (-1, r, 1))
        u_train_std = np.reshape(u_train_std, (-1, r, 1))
        U_train = np.reshape(u_train, (-1, r, 1)) * 10
        # U_train = (U_train - u_train_mean)/(u_train_std + 1.0e-9)
        U_test = np.reshape(u_test, (-1, r, 1)) * 10
        # U_test = (U_test - u_train_mean)/(u_train_std + 1.0e-9)

        f_norm_train = np.reshape(f_train, (-1, s * s, 1), order="F")
        f_norm_test = np.reshape(f_test, (-1, s * s, 1), order="F")

        f_norm_train = f_norm_train.astype(np.float64)
        f_norm_test = f_norm_test.astype(np.float64)

        return F_train, U_train, F_test, U_test, X, \
            u_train_mean, u_train_std, f_norm_train, f_norm_test

    def minibatch(self):
        """
        Generates a random batch from the training data.

        Returns:
            Tuple: Training batched params.
        """
        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)
        f_train = self.F_train[batch_id]
        u_train = self.U_train[batch_id]
        f_norm_train = self.f_norm_train[batch_id]
        x_train = self.X

        Xmin = np.array([0.0, 0.0]).reshape((-1, 2))
        Xmax = np.array([1.0, 1.0]).reshape((-1, 2))

        return x_train, f_train, u_train, f_norm_train, Xmin, Xmax

    def testbatch(self):
        """
        Generates a random batch from the test data.

        Returns:
            Tuple: Testing batched params.
        """
        batch_id = np.random.choice(self.F_test.shape[0], self.test_bs, replace=False)
        f_test = self.F_test[batch_id]
        u_test = self.U_test[batch_id]
        f_norm_test = self.f_norm_test[batch_id]
        x_test = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return x_test, f_test, u_test, f_norm_test, batch_id

    def printbatch(self):
        """
        Generates a random batch from the test data.

        Returns:
            Tuple: Batched params used to calculate final errors and losses.
        """
        batch_id = np.random.choice(self.F_test.shape[0], self.test_bs, replace=False)
        f_test = self.F_test[batch_id]
        u_test = self.U_test[batch_id]
        f_norm_test = self.f_norm_test[batch_id]
        x_test = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return x_test, f_test, u_test, f_norm_test, batch_id
