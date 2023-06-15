import tensorflow as tf


class Train_Adam:
    def __init__(self, model, tf_data_type, B_net, T_net, W_b_dict, hyperparameters):
        self.model = model
        self.tf_data_type = tf_data_type
        self.B_net = B_net
        self.T_net = T_net
        self.W_b_dict = W_b_dict
        self.stride = hyperparameters["stride"]
        tf.keras.mixed_precision.set_global_policy("float64")

    def train_vars(self):
        """
        Creates a list of trainable variables for the model.

        Returns:
            List: Trainable variables, consisting of the weights and biases of the model.
        """
        Y = (
            [self.W_b_dict["W_br1"]]
            + [self.W_b_dict["b_br1"]]
            + [self.W_b_dict["W_br2"]]
            + [self.W_b_dict["b_br2"]]
            + [self.W_b_dict["W_br3"]]
            + [self.W_b_dict["b_br3"]]
            + [self.W_b_dict["W_br4"]]
            + [self.W_b_dict["b_br4"]]
            + self.W_b_dict["W_br_fnn"]
            + self.W_b_dict["b_br_fnn"]
            + self.W_b_dict["W_tr"]
            + self.W_b_dict["b_tr"]
        )

        return Y

    @tf.function(jit_compile=True)
    def call(self, X, F, F_norm, Xmin, Xmax):
        """
        Forward pass of the DeepONet, containing both Branch and Trunk networks.

        Args:
            X (Tensor object of ndarray): 2D array of the input spatial coordinates. Used in the Trunk network.
            F (Tensor object of ndarray): 2D array of the input functions. Used in the Branch network.
            F_norm (Tensor object of ndarray): Masking layer for the Branch network. 1 for spaces inside the target domain, 
            0 for spaces outside the target domain.
            Xmin (Tensor object of ndarray): Minimum value of the input spatial coordinates.
            Xmax (Tensor object of ndarray): Maximum value of the input spatial coordinates.

        Returns:
            Tensor object of ndarray: Prediction of the DeepONet.
        """
        ######################
        ### Branch Network ###
        ######################
        # fnnx,y,z
        # compute derivatives using forward mode autodiff
        # combine to create jacobian
        # combine trunk net
        # pinns loss function

        # # CNN1
        # b_out, self.W_b_dict["W_br1"], self.W_b_dict["b_br1"] = \
        # self.model.conv_layer(F, self.W_b_dict["W_br1"], self.W_b_dict["b_br1"], self.stride, actn=tf.nn.relu)
        # pool = self.model.avg_pool(b_out, 2, 2)# switch to max_pool

        # # CNN2
        # b_out, self.W_b_dict["W_br2"], self.W_b_dict["b_br2"] = \
        #     self.model.conv_layer(pool, self.W_b_dict["W_br2"], self.W_b_dict["b_br2"], self.stride, actn=tf.nn.relu)
        # pool = self.model.avg_pool(b_out, 2, 2)

        # # CNN3
        # b_out, self.W_b_dict["W_br3"], self.W_b_dict["b_br3"] = \
        #     self.model.conv_layer(pool, self.W_b_dict["W_br3"], self.W_b_dict["b_br3"], self.stride, actn=tf.nn.relu)
        # pool = self.model.avg_pool(b_out, 2, 2)

        # # CNN4
        # b_out, self.W_b_dict["W_br4"], self.W_b_dict["b_br4"] = \
        #     self.model.conv_layer(pool, self.W_b_dict["W_br4"], self.W_b_dict["b_br4"], self.stride, actn=tf.nn.relu)
        # pool = self.model.avg_pool(b_out, 2, 2)
        # flat = self.model.flatten_layer(pool)
        # # FNN
        # u_B = self.model.fnn_B(self.W_b_dict["W_br_fnn"], self.W_b_dict["b_br_fnn"], flat)

        #FNN x
        b_out_x, self.W_b_dict["W_brx"], self.W_b_dict["b_brx"] = self.model.fnn_layer(F_x, self.W_b_dict["W_brx"], self.W_b_dict["b_brx"], actn=tf.nn.relu)

        #FNN y
        b_out_y, self.W_b_dict["W_bry"], self.W_b_dict["b_bry"] = self.model.fnn_layer(F_y, self.W_b_dict["W_bry"], self.W_b_dict["b_bry"], actn=tf.nn.relu)

        #FNN z
        b_out_z, self.W_b_dict["W_brz"], self.W_b_dict["b_brz"] = self.model.fnn_layer(F_z, self.W_b_dict["W_brz"], self.W_b_dict["b_brz"], actn=tf.nn.relu)

        
        #####################
        ### Trunk Network ###
        #####################

        # FNN
        u_T = self.model.fnn_T(self.W_b_dict["W_tr"], self.W_b_dict["b_tr"], X, Xmin, Xmax)

        ###

        # Combine Branch and Trunk networks
        u_nn = tf.einsum("ik,jk->ij", u_B, u_T)
        u_nn = tf.expand_dims(u_nn, axis=-1)
        u_pred = u_nn * F_norm

        return u_pred

    @tf.function(jit_compile=True)
    def nn_train(self, optimizer, X, F, U, F_norm, Xmin, Xmax):
        """
        Backward pass of the DeepONet, using the Adam optimizer.

        Args:
            optimizer (tf function): Adam optimizer.
            X (Tensor object of ndarray): 2D array of the input spatial coordinates. Used in the Trunk network.
            F (Tensor object of ndarray): 2D array of the input functions. Used in the Branch network.
            U (Tensor object of ndarray): Solution of the PDE.
            F_norm (Tensor object of ndarray): Masking layer for the Branch network. 1 for spaces inside the target domain, 
            0 for spaces outside the target domain.
            Xmin (Tensor object of ndarray): Minimum value of the input spatial coordinates.
            Xmax (Tensor object of ndarray): Maximum value of the input spatial coordinates.

        Returns:
            Tuple of Dictionaries: Returns a tuple of dictionaries. The first dictionary contains the loss
            and the predicted solution. The second contains the weights and biases of the model.
        """
        with tf.GradientTape() as tape:
            u_pred = self.call(X, F, F_norm, Xmin, Xmax)
            loss = tf.reduce_mean(tf.square(U - u_pred) / (tf.square(U) + 1e-4))

        gradients = tape.gradient(loss, self.train_vars())
        optimizer.apply_gradients(zip(gradients, self.train_vars()))

        loss_dict = {"loss": loss, "U_pred": u_pred}
        return loss_dict, self.W_b_dict
