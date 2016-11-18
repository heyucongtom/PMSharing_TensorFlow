from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)
sess.run(c)
