""" Mean to experiment downpourSGD on MNIST dataset """


import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

class DownpourSGDTrainer(object):

    def __init__(self, ):
        self.init_flags()

    def init_flags(self):
        tf.app.flags.DEFINE_string("train_dir", "./tmp/mnist_train", """Directory for training data""")
        tf.app.flags.DEFINE_string
        # Flags for defining the tf.train.ClusterSpec
        tf.app.flags.DEFINE_string("ps_hosts", "",
                                   "Comma-separated list of hostname:port pairs")
        tf.app.flags.DEFINE_string("worker_hosts", "",
                                   "Comma-separated list of hostname:port pairs")

        # Flags for defining the tf.train.Server
        tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
        tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

        self.FLAGS = tf.app.flags.FLAGS

    def calculate_loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    def setup_server(self):
        pass
