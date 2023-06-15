import tensorflow as tf


class Error_Test:
    def __init__(self, W_b_dict, model, tf_data_type, saveresultsto):
        self.W_b_dict = W_b_dict
        self.model = model
        self.tf_data_type = tf_data_type
        self.save_results_to = saveresultsto

    def nn_error_test(self, x_test, f_test, u_test, f_norm_test, stride, Xmin, Xmax, batch_id):
        """
        Calculates the error of the model.

        Args:
            x_test (Tensor object of ndarray): Input space values for testing.
            f_test (Tensor object of ndarray): Functions for testing.
            u_test (Tensor object of ndarray): True solution for testing.
            f_norm_test (Tensor object of ndarray): Branch masking layer for testing.
            stride (Tensor object of ndarray): Stride for testing.
            Xmin (Tensor object of ndarray): Minimum value of input space for testing.
            Xmax (Tensor object of ndarray): Maximum value of input space for testing.
            batch_id (Tensor object of ndarray): Batch ID for testing.

        Returns:
            Tuple containing network params: Returns batch_id, true f, true u, and predicted u.
            
        """
        ######################
        ### Branch Network ###
        ######################

        # CNN1
        b_out, self.W_b_dict["W_br1"], self.W_b_dict["b_br1"] = \
            self.model.conv_layer(f_test, self.W_b_dict["W_br1"], self.W_b_dict["b_br1"], stride, actn=tf.nn.relu)
        pool = self.model.avg_pool(b_out, 2, 2)

        # CNN2
        b_out, self.W_b_dict["W_br2"], self.W_b_dict["b_br2"] = \
            self.model.conv_layer(pool, self.W_b_dict["W_br2"], self.W_b_dict["b_br2"], stride, actn=tf.nn.relu)
        pool = self.model.avg_pool(b_out, 2, 2)

        # CNN3
        b_out, self.W_b_dict["W_br3"], self.W_b_dict["b_br3"] = \
            self.model.conv_layer(pool, self.W_b_dict["W_br3"], self.W_b_dict["b_br3"], stride, actn=tf.nn.relu)
        pool = self.model.avg_pool(b_out, 2, 2)
        
        # CNN4
        b_out, self.W_b_dict["W_br4"], self.W_b_dict["b_br4"] = \
            self.model.conv_layer(pool, self.W_b_dict["W_br4"], self.W_b_dict["b_br4"], stride, actn=tf.nn.relu)
        pool = self.model.avg_pool(b_out, 2, 2)
        flat = self.model.flatten_layer(pool)

        # FNN
        u_B = self.model.fnn_B(self.W_b_dict["W_br_fnn"], self.W_b_dict["b_br_fnn"], flat)

        #####################
        ### Trunk Network ###
        #####################

        # FNN
        u_T = self.model.fnn_T(self.W_b_dict["W_tr"], self.W_b_dict["b_tr"], x_test, Xmin, Xmax)

        ###

        # Combine Branch and Trunk networks
        u_nn = tf.einsum("ik,jk->ij", u_B, u_T)
        u_nn = tf.expand_dims(u_nn, axis=-1)
        u_pred = u_nn * f_norm_test

        return batch_id, f_test, u_test, u_pred
