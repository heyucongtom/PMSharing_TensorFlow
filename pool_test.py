from multiprocessing import Pool
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from functools import partial
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def decorator(func):
    def wrapper(**kwargs):
        return partial(func, **kwargs)
    return wrapper


def run_steps(params):
    num_steps, weights_1, biases_1 = params
    # biases_1 = tf.placeholder(tf.float32, [10])
    # weights_1 = tf.placeholder(tf.float32, [784, 10])
    X_input = tf.placeholder(tf.float32, [None, 784])
    labels = tf.placeholder(tf.float32, [None, 10])

    # global_step = tf.Variable(0)

    logits = tf.nn.softmax(tf.matmul(X_input, weights_1) + biases_1)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    train_op = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        for _ in range(num_steps):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_op, feed_dict={X_input: batch_xs, labels: batch_ys})
        correct_prediction = tf.equal(
            tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print(sess.run(accuracy, feed_dict={
              X_input: mnist.test.images, labels: mnist.test.labels}))
    return weights_1, biases_1


if __name__ == '__main__':
    pool = Pool(4)
    weights = tf.Variable(tf.zeros([784, 10]))
    biases = tf.Variable(tf.zeros([10]))

    params_list = []
    for _ in range(4):
        params_list.append([1, tf.identity(weights), tf.identity(biases)])

    for _ in range(10000):
        print(pool.map(run_steps, params_list))
