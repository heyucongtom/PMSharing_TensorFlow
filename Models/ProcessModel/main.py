"""
Run the Model.
"""
import multiprocessing as mp
import numpy as np
from ParameterServer import ParameterServer, MNISTSoftmaxModel
from Client import MNISTClient
from multiprocessing.managers import BaseManager

class MyManager(BaseManager): pass

def Manager():
    m = MyManager()
    m.start()
    return m

MyManager.register('Server', ParameterServer)

def train(server, i):
    name = "CL" + str(i)
    client = MNISTClient(name=name)
    client.train(server)

if __name__ == '__main__':

    """MNIST Part"""
    init_weights = np.zeros([784, 10]).astype(np.float32, copy=False)
    init_bias = np.zeros(10).astype(np.float32, copy=False)
    # Build the model
    model = MNISTSoftmaxModel(init_weights, init_bias)

    # Build the parameter server hosting the model.
    # ps = ParameterServer(model=model)

    # Build the client.
    client_lst = []

    """
    How to share the same parameter server?
    If not sharing, then shall output two similar training result if programed correctly.
    """

    # Server is shared by a custom manager
    # Reference: http://stackoverflow.com/questions/28612412/how-can-i-share-a-class-between-processes-in-python
    manager = Manager()
    server = manager.Server(model)
    pool = mp.Pool(mp.cpu_count())
    for i in range(4):
        # TODO: pool.async_apply will encounter computation overflow
        pool.apply_async(func=train, args=(server, i))
    pool.close()
    pool.join()
