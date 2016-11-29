""" Mean to experiment downpourSGD on MNIST dataset """


import numpy as np
import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

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
        FLAGS = self.flags.FLAGS
        self.data_set = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

    def init_flags(self):
        self.flags = tf.app.flags
        self.flags.DEFINE_string("train_log", "./tmp/mnist_train_logs", """dir for training log""")
        self.flags.DEFINE_string("train_dir", "./tmp/mnist_train", """Directory for training data""")

        # Flags for defining the tf.train.Server
        self.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
        self.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

        self.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
        self.flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
        self.flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
        self.flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
        self.flags.DEFINE_integer('batch_size', 200, 'Batch size.  '
                             'Must divide evenly into the dataset sizes.')
        self.flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                             'for unit testing.')

    def placeholder_inputs(self, batch_size):
        images_placeholder = tf.placeholder(tf.float32, shape=([batch_size, mnist.IMAGE_PIXELS]))
        labels_placeholder = tf.placeholder(tf.int32, shape=([batch_size]))
        return images_placeholder, labels_placeholder

    def fill_feed_dict(self, data_set, image_pl, label_pl):
        FLAGS = self.flags.FLAGS
        image_feed, label_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)

        feed_dict = {image_pl: image_feed, label_pl: label_feed}
        return feed_dict

    def do_eval(self,sess, eval_correct, images_placeholder, labels_placeholder, data_set):

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
        FLAGS = self.flags.FLAGS

        # Set numbers divisible by total.
        steps_per_epoch = data_set.num_examples // FLAGS.batch_size
        num_examples = steps_per_epoch * FLAGS.batch_size

        for step in range(steps_per_epoch):
          feed_dict = self.fill_feed_dict(data_set, images_placeholder, labels_placeholder)
          # Eval_correct is an op;
          # Running eval_correct will call mnist.evaluation for model, which will do the
          # reduce_sum for correct labels.
          true_count += sess.run(eval_correct, feed_dict=feed_dict)

        precision = true_count / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (num_examples, true_count, precision))

    def downpour_training_distributed_op(self):
        """
        Set up workers with corresponding constants
        """
        FLAGS = self.flags.FLAGS

        # Pass in by  --ps_hosts=ps0.example.com:2222, ps1.example.com:2222
        # ps_hosts = FLAGS.ps_hosts.split(",")
        # worker_hosts = FLAGS.worker_hosts.split(",")

        # Create cluster:
        cluster = tf.train.ClusterSpec({"ps": ["localhost:2222"], "worker":["localhost:2222", "localhost:2222"]})
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

        if FLAGS.job_name == "ps":
            # Do something for parameter sharing scheme.
            # Currently updating all part.
            server.join()

        elif FLAGS.job_name == "worker":
            # Assign operations to local worker by default:
            with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/replica:%d/task:%d/cpu:%d" % (0, FLAGS.task_index, 0))):
                # Bulid model:
                # Do something for parameter sharing scheme.
                # Currently updating all parameters.
                images_placeholder, labels_placeholder = self.placeholder_inputs(FLAGS.batch_size)
    
                logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)

                loss = mnist.loss(logits, labels_placeholder)

                # Create a variable to track the global step
                global_step = tf.Variable(0, name='global_step', trainable='False')

                # Add a scalar summary for the snapshot loss.
                tf.summary.scalar('loss', loss)

                # Create the gradient descent optimizer with the given learning rate.
                optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

                # Use the optimizer to apply the gradients that minimize the loss.
                # feed_dict somewhere.
                train_op = optimizer.minimize(loss, global_step=global_step)

                saver = tf.train.Saver()
                summary_op = tf.merge_all_summaries()
                init_op = tf.initialize_all_variables()

            # Create a "supervisor", which oversees the training process.
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                     logdir=FLAGS.train_log,
                                     init_op=init_op,
                                     summary_op=summary_op,
                                     saver=saver,
                                     global_step=global_step,
                                     save_model_secs=600)
            # The supervisor takes care of session initialization, restoring from
            # a checkpoint, and closing when done or an error occurs.

            with sv.managed_session(server.target, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                # Loop until the supervisor shuts down or 1000 steps have completed.
                step = 0
                while not sv.should_stop() and step < 1000:
                    # Run a training step asynchronously.
                    feed_dict = self.fill_feed_dict(self.data_set.train, images_placeholder, labels_placeholder)

                    # Run one step of the model.  The return values are the activations
                    # from the `train_op` (which is discarded) and the `loss` Op.  To
                    # inspect the values of your Ops or variables, you may include them
                    # in the list passed to sess.run() and the value tensors will be
                    # returned in the tuple from the call.
                    _, step = sess.run([train_op, loss], feed_dict=feed_dict)
                sv.stop()

    def downpour_training_local_op(self):
        """
        Validation baseline function: run locally.
        """
        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            FLAGS = self.flags.FLAGS
            images_placeholder, labels_placeholder = self.placeholder_inputs(FLAGS.batch_size)

            # Do inference:
            logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)

            # Calculate loss after generating logits:
            loss = mnist.loss(logits, labels_placeholder)

            # Add loss to training:
            train_op = mnist.training(loss, FLAGS.learning_rate)

            # Add summary
            summary = tf.merge_all_summaries()

            # Add the Op to compare the logits to the labels during evaluation.
            eval_correct = mnist.evaluation(logits, labels_placeholder)

            # Initialize Variable
            init = tf.initialize_all_variables()

            sess = tf.Session()

            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

            sess.run(init)


            for step in range(FLAGS.max_steps+1):

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

                # Write the summaries and print an overview fairly often.
                if step % 100 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                    summary_str = sess.run(summary, feed_dict = feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # Save a checkpoint and evaluate the model periodically.
                if step % 1000 == 0:
                    print('Training Data Eval:')
                    self.do_eval(sess,
                            eval_correct,
                            images_placeholder,
                            labels_placeholder,
                            self.data_set.train)
                    # Evaluate against the validation set.
                    print('Validation Data Eval:')
                    self.do_eval(sess,
                            eval_correct,
                            images_placeholder,
                            labels_placeholder,
                            self.data_set.validation)
                    # Evaluate against the test set.
                    print('Test Data Eval:')
                    self.do_eval(sess,
                            eval_correct,
                            images_placeholder,
                            labels_placeholder,
                            self.data_set.test)

trainer = DownpourSGDTrainer()
trainer.downpour_training_distributed_op()

#python downpour_MNIST_train.py --job_name=ps  --task_index=0
#python downpour_MNIST_train.py --job_name=worker  --task_index=0
#python downpour_MNIST_train.py --job_name=worker  --task_index=1
