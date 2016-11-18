""" Mean to experiment downpourSGD on MNIST dataset """


import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

class DownpourSGDTrainer(object):

    """
    Implement downpour asynchronous stochastic gradient descent on classic MNIST data_set.
    Attempt to to this by
        1. Create a cluster with number of workers and one parameter server.
        2. Assign hidden layers to parameter server.
        3. Assign computation to workers.
        4. Workers sent their update to parameter server asynchronously.
        5. Vary communication variance & training batch size to explore relations.
    """

    def __init__(self, ):
        self.init_flags()

    def init_flags(self):
        self.flags = tf.app.flags
        self.flags.DEFINE_string("train_dir", "./tmp/mnist_train", """Directory for training data""")
        self.flags.DEFINE_string

        # Flags for defining the tf.train.ClusterSpec
        self.flags.DEFINE_string("ps_hosts", "",
                                   "Comma-separated list of hostname:port pairs")
        self.flags.DEFINE_string("worker_hosts", "",
                                   "Comma-separated list of hostname:port pairs")

        # Flags for defining the tf.train.Server
        self.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
        self.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

        self.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
        self.flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
        self.flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
        self.flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
        self.flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                             'Must divide evenly into the dataset sizes.')
        self.flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
        self.flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                             'for unit testing.')

    def placeholder_inputs(self, batch_size):
        images_placeholder = tf.placeholder(tf.float32, shape=([batch_size, mnist.IMAGE_PIXELS]))
        labels_placeholder = tf.placeholder(tf.int32, shape=([batch_size]))
        return images_placeholder, labels_placeholder

    def fill_feed_dict(self, data_set, image_pl, label_pl):
        image_feed, label_feed = data_set.next_batch(flags.batch_size, flags.fake_data)

        feed_dict = {image_pl: image_feed, label_pl: label_feed}
        return feed_dict

    def setup_server(self):
        """
        Set up workers with corresponding constants
        """
        FLAGS = self.flags.FLAGS

        # Pass in by  --ps_hosts=ps0.example.com:2222, ps1.example.com:2222
        ps_hosts = FLAGS.ps_hosts.split(",")
        worker_hosts = FLAGS.worker_hosts.split(",")

        # Create cluster:
        cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

        # Create and start a server: pass in by --job_name=worker --task_index=1
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

        if FLAGS.job_name == "ps":
            # Do something for parameter sharing scheme.
        elif FLAGS.job_name == "worker":
            # Assign operations to local worker by default:
            with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster
            )):
                # Bulid model:
                # Do something for parameter sharing scheme.

    def downpour_training_op(self):
        """
        Validation baseline function: run locally.
        """

        images_placeholder, labels_placeholder = self.placeholder_inputs(self.flags.batch_size)

        # Do inference:
        logits = mnist.inference(images_placeholder, self.flags.hidden1, self.flags.hidden2)
