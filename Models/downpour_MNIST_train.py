""" Mean to experiment downpourSGD on MNIST dataset """


import numpy as np
import tensorflow as tf
import time

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
        self.data_set = mnist

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

    def do_eval(self, eval_correct, images_placeholder, labels_placeholder, data_set):

        """Runs one evaluation against the full epoch of data.
        This function is to check stage for both asyn and syn loss.

        Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
          input_data.read_data_sets().
        """

        # And run one epoch of eval.
        true_count = 0

        # Set numbers divisible by total.
        steps_per_epoch = data_set.num_examples // FLAGS.batch_size
        num_examples = steps_per_epoch * FLAGS.batch_size

        for step in xrange(steps_per_epoch):
          feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
          # Eval_correct is an op;
          # Running eval_correct will call mnist.evaluation for model, which will do the
          # reduce_sum for correct labels.
          true_count += sess.run(eval_correct, feed_dict=feed_dict)

        precision = true_count / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (num_examples, true_count, precision))

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
            pass
        elif FLAGS.job_name == "worker":
            # Assign operations to local worker by default:
            with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster
            )):
                # Bulid model:
                # Do something for parameter sharing scheme.
                pass

    def downpour_training_local_op(self):
        """
        Validation baseline function: run locally.
        """
        FLAGS = self.flags.FLAGS
        images_placeholder, labels_placeholder = self.placeholder_inputs(self.flags.batch_size)

        # Do inference:
        logits = mnist.inference(images_placeholder, self.flags.hidden1, self.flags.hidden2)

        # Calculate loss after generating logits:
        loss = mnist.loss(logits, labels_placeholder)

        # Add loss to training:
        train_op = mnist.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # Initialize Variable
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)


        for step in xrange(FLAGS.max_steps):

            """
            We want to inspect loss value on each step as a local benchmark
            for fully connected network.
            """

            start_time = time.time()
            feed_dict = self.fill_feed_dict(self.data_set.train, images_placeholder, labels_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time
