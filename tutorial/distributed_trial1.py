import tensorflow as tf

cluster = tf.train.ClusterSpec({"local": ["localhost:2222"]})
server = tf.train.Server(cluster,
                        job_name="local",
                        task_index=0)

with tf.Session(server.target) as sess:
    with tf.device("/job:local/replica:0/task:0"):
        const1 = tf.constant("Hello I am the first constant")
        const2 = tf.constant("Hello I am the second constant")
    print(sess.run([const1, const2]))
    
