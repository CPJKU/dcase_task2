
from __future__ import print_function

import numpy as np


class DataPool(object):
    """
    Class that holds and provides data for neural network training

    __getitem__ has to return a list of inputs even when only one item is used (as for example with auto-encoders)

    e.g. [x, y], [x, y, z] or only [x]
    """
    def __init__(self, *x):
        """ Constructor """
        if x.__class__ is tuple:
            self.x = list(x)
        else:
            self.x = [x]

        self.shape = [len(x[0])]
        self.n_inputs = len(self.x)

    def __getitem__(self, item):
        """ Make object accessible by indexing """

        if item.__class__ == int:
            item = slice(item, item + 1)

        return [self.x[i_input][item] for i_input in xrange(self.n_inputs)]

    def shuffle(self):
        """ Shuffle data """
        rand_idx = np.random.permutation(self.shape[0])
        for i_input in xrange(len(self.x)):
            self.x[i_input] = self.x[i_input][rand_idx]


if __name__ == '__main__':
    """ main """

    # init some random data
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    # init data pool
    data_pool = DataPool(X, y)

    # test data structure
    x, y = data_pool[0:10]
    print(x.shape, y.shape)

    # test data structure
    x, y = data_pool[[1, 4, 5]]
    print(x.shape, y.shape)

    # test data structure
    x, y = data_pool[5]
    print(x.shape, y.shape)

    # init data pool
    data_pool = DataPool(X)
    x, = data_pool[0:10]
    print(x.shape)
