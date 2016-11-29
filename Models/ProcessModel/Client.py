import tensorflow as tf
import numpy as np
import time

def build_mnist_graph(graph, device, MNIST_server, name):
    """
    Build a inference & training graph on mnist.
    """
    with graph.as_default():
        with graph.device(device):
            # Input placeholder.
            # We need to specify the model for parameter sharing.
            MNIST_model = MNIST_server.getModel()
            params = MNIST_model.getParams()
            print("Graph initializing...")
            print(params['bias_1'])

            X_input = tf.placeholder(tf.float32, [None, 784])
            y = tf.placeholder(tf.float32, [None, 10])

            # Load weights from model.
            weights = tf.get_variable('weights', initializer=params['weight_1'], dtype=tf.float32)
            bias = tf.get_variable('bias', initializer=params['bias_1'], dtype=tf.float32)



            # Build forward inference graph
            # global_step = MNIST_server.global_step

            logits = tf.nn.softmax(tf.matmul(X_input, weights) + bias)
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))

            # Build gradient
            opt = tf.train.GradientDescentOptimizer(0.5, name=name)
            gradients = opt.compute_gradients(cross_entropy)

            train_op = opt.apply_gradients(gradients)

            init = tf.initialize_all_variables()

            saver = tf.train.Saver()
            return train_op, gradients, weights, bias, saver, X_input, y, logits, init


class MNISTClient(object):

    """ This client do the MNIST softmax training.
        Currently go with eval everything.
    """

    def __init__(self, name, step = 20, device='/cpu:0'):
        self.communication_step = step
        self.device = device
        self.name = name

    def train(self, server, steps=10, batch_size=100):
        """
        Actually do the training.
        1. In the training graph, the client get params from the server.
        2. Run the train_op for k steps.
        3. At multiple of steps, we get the params and write to server.
        """
        """
        Initialization code based on server parameter
        """
        from tensorflow.examples.tutorials.mnist import input_data
        data_set = input_data.read_data_sets('MNIST_data', one_hot=True)
        graph = tf.Graph()

        # Build the graph.
        train_op, gradients, weights, bias, saver, X_input, y, logits, init = build_mnist_graph(graph, self.device, server, self.name)

        # Assert

        sess = tf.Session(graph=graph)
        sess.run(init)
        # print(self.weights.eval(session=sess))
        start = time.time()
        for i in range(1501):
            # Update the gradients on server.
            gradient_batch = {}
            params = server.getParams()

            if i % self.communication_step == 0:

                assign_op_1 = weights.assign(params['weight_1'])
                assign_op_2 = bias.assign(params['bias_1'])

                sess.run(assign_op_1)
                sess.run(assign_op_2)

            batch_xs, batch_ys = data_set.train.next_batch(batch_size)

            feed_dict = {X_input: batch_xs, y: batch_ys}
            sess.run(train_op, feed_dict=feed_dict)

            # Currently we use the raw expansion. Maybe improving in the future.
            gradient_batch['weight_1'] = (gradients[0][1].eval(session=sess) - params['weight_1']) / self.communication_step
            gradient_batch['bias_1'] = (gradients[1][1].eval(session=sess) - params['bias_1']) / self.communication_step

            server.applyGradients(gradient_batch)

            # if i % 1 == 0:
            #
            #     print(self.name, i)
            #     print("Gradients: ", gradient_batch['bias_1'])
            #     print(bias.eval(session=sess))

        # print("Here")
        # print(bias.eval(session=sess))
        # print("what")
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        casted = tf.cast(correct_prediction, tf.float32)

        print("The accuracy is {0}".format(sum(casted.eval(session=sess, feed_dict={X_input:data_set.test.images, y: data_set.test.labels})) / 10000.0))
        print(time.time() - start)

            # accuracy = tf.reduce_mean(casted)
                #
                # print(sess.run(accuracy, feed_dict={x:self.data_set.test.images, _y: self.data_set.test.labels}))

    def test_assignment(self):
        """
        Some testing code.
        Test the variable assignment.
        """
        init = tf.constant(np.random.rand(1, 2))
        test_w = tf.get_variable('test_w', initializer=init)
        init_var = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init_var)
            print("First time: ")
            print(test_w.eval(session=sess))
            assign_op = test_w.assign(tf.zeros([1,2], dtype=tf.float64))
            sess.run(assign_op)
            print("After assign: ")
            print(test_w.eval(session=sess))
