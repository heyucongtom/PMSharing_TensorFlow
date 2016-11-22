from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

cluster = tf.train.ClusterSpec({"ps": ["localhost:2222"], "worker":["localhost:2222", "localhost:2222"]})
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
# server = tf.train.Server.create_local_server()

# Parameter server:
if FLAGS.job_name == "ps":
    server.join()

if FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/replica:%d/task:%d/cpu:%d" % (0, FLAGS.task_index, 0))):
        weights_1 = tf.Variable(tf.zeros([784, 10]))
        biases_1 = tf.Variable(tf.zeros([10]))

        X_input = tf.placeholder(tf.float32, [None, 784])

        global_step = tf.Variable(0)

        logits = tf.nn.softmax(tf.matmul(X_input, weights_1) + biases_1)
        labels = tf.placeholder(tf.float32, [None, 10])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

        # opt = tf.train.GradientDescentOptimizer(0.5)
        # gradient = opt.compute_gradients(cross_entropy)
        # train_op = opt.apply_gradients(gradient)
        train_op = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy, global_step=global_step)

    with tf.Session(server.target, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(1000000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_op, feed_dict={X_input: batch_xs, labels:batch_ys})

            if i % 1000 == 0:

                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                print(sess.run(accuracy, feed_dict={X_input:mnist.test.images, labels: mnist.test.labels}))

    # python distributed_trial3.py  --job_name=ps --task_index=0
    # python distributed_trial3.py  --job_name=worker --task_index=0
    # python distributed_trial3.py  --job_name=worker --task_index=1
