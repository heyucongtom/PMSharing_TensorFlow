import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Flags for defining the tf.train.ClusterSpec
ps_hosts = []

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("hidden_units", 100,
                            "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                           "Directory for storing mnist data")
tf.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")

FLAGS = tf.app.flags.FLAGS

IMAGE_PIXELS = 28

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Variables of the hidden layer
      hid_w = tf.Variable(
          tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                              stddev=1.0 / IMAGE_PIXELS), name="hid_w")
      hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

      # Variables of the softmax layer
      sm_w = tf.Variable(
          tf.truncated_normal([FLAGS.hidden_units, 10],
                              stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
          name="sm_w")
      sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

      x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
      y_ = tf.placeholder(tf.float32, [None, 10])

      hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
      hid = tf.nn.relu(hid_lin)

      y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
      loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss)

      saver = tf.train.Saver()
      summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()

    # Create a "supervisor", which oversees the training process.
    # sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
    #                          logdir="/tmp/train_logs",
    #                          init_op=init_op,
    #                          summary_op=summary_op,
    #                          saver=saver,
    #                          global_step=global_step,
    #                          save_model_secs=600)

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    print(1)
    with tf.Session("grpc://worker7:2222") as sess:
        print(2)
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_feed = {x: batch_xs, y_: batch_ys}
        for t in range(10000):
            print(3)
            sess.run(train_op, feed_dict=train_feed)
            print(4)
            if t % 100 == 0:
                print("Done step %d" % t)
      # Loop until the supervisor shuts down or 1000000 steps have completed.
        # step = 0
        # while not sv.should_stop() and step < 10000:
        #
        # # Run a training step asynchronously.
        # # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # # perform *synchronous* training.
        #
        # batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        # train_feed = {x: batch_xs, y_: batch_ys}
        #
        # _, step = sess.run([train_op, global_step], feed_dict=train_feed)
        # if step % 100 == 0:
        #     print("Done step %d" % step)

    # Ask for all the services to stop.
    # sv.stop()

if __name__ == "__main__":
  tf.app.run()

  #python distributed_tutorial.py      --ps_hosts=ps0.example.com:2222     --worker_hosts=worker0.example.com:2222      --job_name=ps --task_index=0
  #python distributed_tutorial.py      --ps_hosts=ps0.example.com:2222     --worker_hosts=worker0.example.com:2222     --job_name=worker --task_index=0
