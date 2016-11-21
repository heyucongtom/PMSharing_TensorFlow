import tensorflow as tf

def main() :
   with tf.device("/job:ps/task:0/cpu:0"):
      a = tf.Variable(1)
   with tf.device("/job:ps/task:0/cpu:1"):
      b = tf.Variable(1)
   with tf.Session("grpc://localhost:22223",config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
      init_ab = tf.initialize_all_variables();
      sess.run(init_ab)
      result = sess.run(a+b)
      print(result)

if __name__ == '__main__':
    main()
