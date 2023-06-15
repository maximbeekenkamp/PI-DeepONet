import tensorflow as tf
import numpy as np


class DeepONet:
    def __init__(self, tf_data_type):
        self.tf_data_type = tf_data_type

    #####################
    ### Trunk Network ###
    #####################

    def hyper_initial_fnn(self, layers):
        """
        Initialises the weights and biases for a FNN.

        Args:
            layers (list): list description of model architecture.

        Returns:
            Tuple: Initialised weights and biases.
        """
        # with tf.device('/GPU:0'): #avoid laptop error
        L = len(layers)
        W = []
        b = []
        for l in range(1, L):
            in_dim = layers[l - 1]
            out_dim = layers[l]
            std = np.sqrt(2.0 / (in_dim + out_dim))
            weight = tf.Variable(
                tf.random.normal(shape=[in_dim, out_dim], stddev=std, dtype=self.tf_data_type),
                dtype=self.tf_data_type
            )
            bias = tf.Variable(
                tf.zeros(shape=[1, out_dim], dtype=self.tf_data_type),
                dtype=self.tf_data_type
            )

            W.append(weight)
            b.append(bias)

        return W, b

    def fnn_T(self, W, b, X, Xmin, Xmax):
        """
        Forward pass of the Trunk network.

        Args:
            W (Tensor object of ndarray): Weights of the Trunk network.
            b (Tensor object of ndarray): Biases of the Trunk network.
            X (Tensor object of ndarray): Input spatial coordinates.
            Xmin (Tensor object of ndarray): Minimum value of the input spatial coordinates.
            Xmax (Tensor object of ndarray): Maximum value of the input spatial coordinates.

        Returns:
            Tensor: Output of the Trunk network.
        """
        A = 2.0 * (X - Xmin) / (Xmax - Xmin) - 1.0
        L = len(W)
        for i in range(L - 1):
            A = tf.nn.leaky_relu(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.nn.relu(tf.add(tf.matmul(A, W[-1]), b[-1]))

        return Y

    ######################
    ### Branch Network ###
    ######################

    def hyper_initial_cnn(self, shape_w, shape_b):
        """
        Initialises the weights and biases for a CNN.

        Args:
            shape_w (array): Shape of the weights.
            shape_b (array): Shape of the biases.

        Returns:
            Tuple: Tuple containing the initialised weights and biases.
        """
        std = 0.01
        # with tf.device('/GPU:0'): #avoid laptop error
        weight = tf.Variable(
            tf.random.normal(shape=shape_w, stddev=std, dtype=self.tf_data_type),
            dtype=self.tf_data_type
        )
        bias = tf.Variable(
            tf.zeros(shape=shape_b, dtype=self.tf_data_type),
            dtype=self.tf_data_type
        )
        return weight, bias

    def conv_layer(self, x, w, b, stride, actn=tf.nn.relu):
        """
        Performs a convolutional layer.

        Args:
            x (Tensor object of ndarray): Input of the convolutional layer.
            w (Tensor object of ndarray): Weights of the convolutional layer.
            b (Tensor object of ndarray): Bias of the convolutional layer.
            stride (Tensor object of ndarray): Stride of the convolutional layer.
            actn (tf function, optional): Desired activation function. Defaults to tf.nn.relu.

        Returns:
            Tuple: Tuple containing the output of the convolutional layer output, the weights and the biases.
        """
        layer = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="SAME")
        layer += b
        return actn(layer), w, b

    def avg_pool(self, x, ksize, stride):
        """
        Performs an average pooling layer.

        Args:
            x (Tensor object of ndarray): Input of the average pooling layer.
            ksize (int): Kernel size of the average pooling layer.
            stride (int): Stride of the average pooling layer.

        Returns:
            Tensor object of ndarray: Output of the average pooling layer.
        """
        pool_out = tf.nn.avg_pool(
            x,
            ksize=[1, ksize, ksize, 1],
            strides=[1, stride, stride, 1],
            padding="SAME"
        )
        return pool_out

    def max_pool(self, x, ksize, stride):
        """
        Performs a max pooling layer.

        Args:
            x (Tensor object of ndarray): Input of the max pooling layer.
            ksize (int): Kernel size of the max pooling layer.
            stride (int): Stride of the max pooling layer.

        Returns:
            Tensor object of ndarray: Output of the max pooling layer.
        """
        pool_out = tf.nn.max_pool(
            x,
            ksize=[1, ksize, ksize, 1],
            strides=[1, stride, stride, 1],
            padding="SAME",
        )
        return pool_out

    def flatten_layer(self, layer):
        """
        Flattens a layer.

        Args:
            layer (Tensor object of ndarray): Input layer.

        Returns:
            Tensor object of ndarray: Flattened layer.
        """
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat

    def fnn_B(self, W, b, X):
        """
        Forward pass of the Branch network.

        Args:
            W (Tensor object of ndarray): Weights of the Branch network.
            b (Tensor object of ndarray): Biases of the Branch network.
            X (Tensor object of ndarray): Input functions for the Branch network.

        Returns:
            Tensor object of ndarray: Output of the Branch dense network.
        """
        L = len(W)
        for i in range(L - 1):
            X = tf.nn.leaky_relu(tf.add(tf.matmul(X, W[i]), b[i]))
        Y = tf.nn.relu(tf.add(tf.matmul(X, W[-1]), b[-1]))

        return Y

    # Saving helper functions
    def save_W_b(self, W, b):
        L = len(W)
        W_ = []
        b_ = []
        for i in range(L):
            W_.append(W[i].numpy())
            x = b[i]
            b_.append(np.reshape(x, (-1)))

        return W_, b_

    def save_W(self, W):
        L = len(W)
        W_ = []
        for i in range(L):
            W_.append(W[i].numpy())

        return W_
