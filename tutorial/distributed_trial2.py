import tensorflow as tf
import datetime
import numpy as np
n = 4
c1 = tf.Variable([])
c2 = tf.Variable([])
def matpow(M, n):
    if n < 1:
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

t1 = datetime.datetime.now()

with tf.device("/job:worker/task:0"):
    A = np.random.rand(1e2, 1e2).astype('float32')
    c1 = matpow(A,n)

with tf.device("/job:worker/task:1"):
    B = np.random.rand(1e2, 1e2).astype('float32')
    c2 = matpow(B,n)

with tf.Session("grpc://192.168.555.254:2500") as sess:
    sum = c1 + c2

t2 = datetime.datetime.now()
print("Multi node computation time: " + str(t2-t1))
