import tensorflow as tf
import multiprocessing as mp
import numpy as np

class ParameterServer(object):

    def __init__(self, model=None):

        self.model = model
        # Create a variable to track the global step
        # self.global_step = tf.Variable(0, name='global_step', trainable='False')

    def getModel(self):
        return self.model

    def getParams(self):
        """
        Return the evaluated model.
        """
        ret = self.model.getParams()
        return ret

    def applyGradients(self, gradients):
        """
        Set the params hosted on the server.
        """
        self.model.applyGradients(gradients)


class MNISTSoftmaxModel(object):

    def __init__(self, weights, bias):

        self.weights = weights
        self.bias = bias

    def getParams(self):

        """
        We eval the weights and bias there and pass the params as a dictionary.
        """
        ret_dict = {}
        ret_dict['weight_1'] = np.copy(self.weights)
        ret_dict['bias_1'] = np.copy(self.bias)
        return ret_dict

    def applyGradients(self, gradients):
        grad_weight = gradients['weight_1']
        grad_bias = gradients['bias_1']
        self.weights += grad_weight
        self.bias += grad_bias
