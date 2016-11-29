"""
Run the Model.
"""
import multiprocessing as mp
import numpy as np
from ParameterServer import ParameterServer, MNISTSoftmaxModel
from Client import MNISTClient

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
    Client1 = MNISTClient(server=ps, name='CL1')
    Client1.train()

    Client2 = MNISTClient(server=ps, name='CL2')
    Client2.train()
