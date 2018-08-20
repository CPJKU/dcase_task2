
from __future__ import print_function

import numpy as np


def get_batch_iterator():
    """
    Standard batch iterator
    """
    def batch_iterator(batch_size, k_samples, shuffle, fill_last_batch=True):
        return BatchIterator(batch_size=batch_size, k_samples=k_samples, shuffle=shuffle, fill_last_batch=fill_last_batch)

    return batch_iterator


def get_augmented_batch_iterator(flip_left_right=True, flip_up_down=False, crop=None, pad=None, pad_mode='constant'):
    """
    Image classification batch iterator which randomly flips images left and right
    """
    
    def prepare(x, y):

        # flipping
        if flip_left_right:
            fl = np.random.randint(0, 2, x.shape[0])
            for i in xrange(x.shape[0]):
                if fl[i] == 1:
                    x[i] = x[i, :, :, ::-1]

        if flip_up_down:
            fl = np.random.randint(0, 2, x.shape[0])
            for i in xrange(x.shape[0]):
                if fl[i] == 1:
                    x[i] = x[i, :, ::-1, :]

        # pad images
        if pad is not None:
            x = np.pad(x, ((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])), mode=pad_mode)

        # random cropping
        if crop is not None:
            x_new = np.zeros((x.shape[0], x.shape[1], crop[0], crop[1]), dtype=np.float32)
            for i in xrange(x.shape[0]):
                r0 = np.random.randint(0, (x.shape[2] - crop[0]) // 2)
                r1 = r0 + crop[0]
                c0 = np.random.randint(0, (x.shape[3] - crop[1]) // 2)
                c1 = c0 + crop[1]
                x_new[i] = x[i, :, r0:r1, c0:c1]
            x = x_new

        return x, y
    
    def batch_iterator(batch_size, k_samples, shuffle):
        return BatchIterator(batch_size=batch_size, prepare=prepare, k_samples=k_samples, shuffle=shuffle)

    return batch_iterator


def get_segmentation_flip_batch_iterator(flip_left_right=True, flip_up_down=False):
    """
    Image segmentation batch iterator which randomly flips images (and mask) left and right
    """

    def prepare(x, y):

        # flipping
        if flip_left_right:
            fl = np.random.randint(0, 2, x.shape[0])
            for i in xrange(x.shape[0]):
                if fl[i] == 1:
                    x[i] = x[i, :, :, ::-1]
                    y[i] = y[i, :, :, ::-1]

        if flip_up_down:
            fl = np.random.randint(0, 2, x.shape[0])
            for i in xrange(x.shape[0]):
                if fl[i] == 1:
                    x[i] = x[i, :, ::-1, :]
                    y[i] = y[i, :, ::-1, :]

        return x, y

    def batch_iterator(batch_size, k_samples, shuffle):
        return BatchIterator(batch_size=batch_size, prepare=prepare, k_samples=k_samples, shuffle=shuffle)

    return batch_iterator


class BatchIterator(object):
    """
    Prototype for batch iterator
    """

    def __init__(self, batch_size, re_iterate=1, prepare=None, k_samples=None, shuffle=True, fill_last_batch=True):
        self.batch_size = batch_size

        if prepare is None:
            def prepare(*data):
                return data
        self.prepare = prepare
        
        self.re_iterate = re_iterate
        self.k_samples = k_samples
        self.shuffle = shuffle
        self.fill_last_batch = fill_last_batch
        self.epoch_counter = 0
        self.n_epochs = None

    def __call__(self, data_pool):
        self.data_pool = data_pool
        if self.k_samples is None:
            self.k_samples = self.data_pool.shape[0]
        self.n_batches = self.re_iterate * (self.k_samples // self.batch_size)
        self.n_epochs = self.data_pool.shape[0] // self.k_samples

        if self.shuffle:
            self.data_pool.shuffle()

        return self

    def __iter__(self):

        # compute current epoch index
        idx_epoch = np.mod(self.epoch_counter, self.n_epochs)

        # reiterate entire data-set
        for _ in xrange(self.re_iterate):
                
            # use only k samples per epoch
            for i_b in xrange((self.k_samples + self.batch_size - 1) / self.batch_size):

                # slice batch data
                start = i_b * self.batch_size + idx_epoch * self.k_samples
                stop = (i_b + 1) * self.batch_size + idx_epoch * self.k_samples
                stop = np.min([stop, self.data_pool.shape[0]])
                sl = slice(start, stop)
                xb = self.data_pool[sl]

                # get missing samples
                if self.fill_last_batch:
                    n_sampels = xb[0].shape[0]
                    if n_sampels < self.batch_size:
                        n_missing = self.batch_size - n_sampels

                        x_con = self.data_pool[0:n_missing]
                        for i_input in xrange(len(xb)):
                            xb[i_input] = np.concatenate((xb[i_input], x_con[i_input]))

                yield self.transform(xb)

            # increase epoch counter
            self.epoch_counter += 1

        # shuffle train data after full set iteration
        if self.shuffle and (idx_epoch + 1) == self.n_epochs:
            self.data_pool.shuffle()
    
    def transform(self, data):
        return self.prepare(*data)


def threaded_generator(generator, num_cached=10):
    """
    Threaded generator
    """
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    queue = Queue.Queue(maxsize=num_cached)
    end_marker = object()

    # define producer
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(end_marker)

    # start producer
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer
    item = queue.get()
    while item is not end_marker:
        yield item
        queue.task_done()
        item = queue.get()


def generator_from_iterator(iterator):
    """
    Compile generator from iterator
    """
    for x in iterator:
        yield x


def threaded_generator_from_iterator(iterator, num_cached=10):
    """
    Compile threaded generator from iterator
    """
    generator = generator_from_iterator(iterator)
    return threaded_generator(generator, num_cached)


def batch_compute1(X, compute, batch_size):
    """ Batch compute data """

    # init results
    R = None

    # get number of samples
    n_samples = X.shape[0]

    # get input shape
    in_shape = list(X.shape)[1:]

    # get number of batches
    n_batches = int(np.ceil(float(n_samples) / batch_size))

    # iterate batches
    for i_batch in xrange(n_batches):

        # extract batch
        start_idx = i_batch * batch_size
        excerpt = slice(start_idx, start_idx + batch_size)
        E = X[excerpt]

        # append zeros if batch is to small
        n_missing = batch_size - E.shape[0]
        if n_missing > 0:
            E = np.vstack((E, np.zeros([n_missing] + in_shape, dtype=X.dtype)))

        # compute results on batch
        r = compute(E)

        # init result array
        if R is None:
            R = np.zeros([n_samples] + list(r.shape[1:]), dtype=r.dtype)

        # store results
        R[start_idx:start_idx+r.shape[0]] = r[0:batch_size-n_missing]

    return R


if __name__ == '__main__':
    """ main """
    from data_pool import DataPool

    # init some random data
    X = np.random.randn(100, 1, 20, 20)
    y = np.arange(0, 100)

    # # init data pool
    # data_pool = DataPool(X, y)
    # bi = get_augmented_batch_iterator(flip_left_right=True, flip_up_down=False, crop=None, pad=(2, 5), pad_mode='constant')
    # iterator = bi(batch_size=1, k_samples=100, shuffle=False)
    # for x, y in iterator(data_pool):
    #     print(y, x.shape, y.shape)

    # # init data pool
    # data_pool = DataPool(X)
    # iterator = BatchIterator(batch_size=1)
    # for x, in iterator(data_pool):
    #     print(x.shape)

    # init data pool
    data_pool = DataPool(X, y)
    iterator = BatchIterator(batch_size=10, shuffle=True, k_samples=20)
    for epoch in xrange(100):
        for x, y in iterator(data_pool):
            print(y)

        print("-" * 10)
