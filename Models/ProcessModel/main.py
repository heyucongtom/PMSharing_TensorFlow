"""
Run the Model.
"""
import multiprocessing as mp
import numpy as np
from ParameterServer import ParameterServer, MNISTSoftmaxModel
from Client import MNISTClient

def func(client, ):


if __name__ == '__main__':
    """MNIST Part"""
    init_weights = np.zeros([784, 10]).astype(np.float32, copy=False)
    init_bias = np.zeros(10).astype(np.float32, copy=False)
    # Build the model
    model = MNISTSoftmaxModel(init_weights, init_bias)

    # Build the parameter server hosting the model.
    ps = ParameterServer(model=model)

    # Build the client.
    client_lst = []

    """
    How to share the same parameter server?
    If not sharing, then shall output two similar training result if programed correctly.
    """
    manager = mp.Manager()
    Global = manager.Namespace()


    # TODO: Sharing the server!
    # Currently it would fork the server on each process.
    # Currently the result shall be two parallel process finishing at the same time,
    # with the same accuracy.
    # File Usage: python main.py

    Client1 = MNISTClient(server=ps, name='CL1')

    Client2 = MNISTClient(server=ps, name='CL2')



    p1 = mp.Process(target=Client1.train)
    p2 = mp.Process(target=Client2.train)

    p1.start()
    import time
    time.sleep(5)
    p2.start()
    p1.join()
    p2.join()
