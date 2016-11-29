import tensorflow as tf
import numpy as np

def build_mnist_graph(graph, device, MNIST_server, name):
    """
    Build a inference & training graph on mnist.
    """
    with graph.as_default():
        with graph.device(device):
            # Input placeholder.
            # We need to specify the model for parameter sharing.
            MNIST_model = MNIST_server.model
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

    def __init__(self, server, name, device='/cpu:0'):

        from tensorflow.examples.tutorials.mnist import input_data
        self.data_set = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.server = server
        self.graph = tf.Graph()
        self.device = device

        # Build the graph.
        train_op, gradients, weights, bias, saver, X_input, y, logits, init = build_mnist_graph(self.graph, self.device, self.server, name)

        # Assert
        self.gradients = gradients
        self.train_op = train_op
        self.weights = weights
        self.bias = bias
        self.saver = saver
        self.X_input = X_input
        self.y_label = y
        self.logit = logits
        self.init_op = init

    def train(self, steps=10, batch_size=100):
        """
        Actually do the training.
        1. In the training graph, the client get params from the server.
        2. Run the train_op for k steps.
        3. At multiple of steps, we get the params and write to server.
        """
        print(self.server)
        sess = tf.Session(graph=self.graph)
        sess.run(self.init_op)
        # print(self.weights.eval(session=sess))
        for i in range(601):
            # Update the gradients on server.
            gradient_batch = {}
            params = self.server.getParams()

            if i % 50 == 0:
                print(params['bias_1'])
                assign_op_1 = self.weights.assign(params['weight_1'])
                assign_op_2 = self.bias.assign(params['bias_1'])

                sess.run(assign_op_1)
                sess.run(assign_op_2)

            batch_xs, batch_ys = self.data_set.train.next_batch(batch_size)

            feed_dict = {self.X_input: batch_xs, self.y_label: batch_ys}
            sess.run(self.train_op, feed_dict=feed_dict)

            # Currently we use the raw expansion. Maybe improving in the future.
            gradient_batch['weight_1'] = self.gradients[0][1].eval(session=sess) - params['weight_1']
            gradient_batch['bias_1'] = self.gradients[1][1].eval(session=sess) - params['bias_1']

            self.server.applyGradients(gradient_batch)
            #
            # print("Successfully upload gradients: ")
        # print(gradient_batch)
        # print(self.bias.eval(session=sess))
        correct_prediction = tf.equal(tf.argmax(self.logit, 1), tf.argmax(self.y_label, 1))
        casted = tf.cast(correct_prediction, tf.float32)

        print(sum(casted.eval(session=sess, feed_dict={self.X_input:self.data_set.test.images, self.y_label: self.data_set.test.labels})) / 10000.0)


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
