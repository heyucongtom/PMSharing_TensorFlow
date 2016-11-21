from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

cluster = tf.train.ClusterSpec({"ps": ["localhost:2222"], "worker":["localhost:2222"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)


# Parameter server:
with tf.device(""):

  weights_1 = tf.Variable(tf.zeros([784, 10]))
  biases_1 = tf.Variable(tf.zeros([10]))

# Worker
with tf.device(""):
  X_input = tf.placeholder(tf.float32, [None, 784])
  labels = tf.placeholder(tf.float32, [None, 10])

  logits = tf.nn.relu(tf.matmul(X_input, weights_1) + biases_1)

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

  opt = tf.train.GradientDescentOptimizer(0.5)
  gradient = opt.compute_gradients(cross_entropy)
  train_op = opt.apply_gradients(gradient)

with tf.Session("") as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    for _ in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_op, feed_dict={X_input: batch_xs, labels:batch_ys})

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={X_input:mnist.test.images, labels: mnist.test.labels}))
