
from __future__ import print_function

import numpy as np

# from lasagne.utils import floatX
floatX = np.float32


class SubSequenceSlicer(object):
    """
    Slices Sub-Sequences out of Sequence Collection

    Produces a list containing lists of sub-sequences
    """

    def __init__(self, sequences, seq_len=100, step_size=1):
        """
        Constructor

        :param sequences: list of lists containing one or more sequences
        :param seq_len: length of sub-sequences to be returned
        :param step_size: step size for sub-sequence sliding window
        """

        self.sequences = sequences

        self.seq_len = seq_len
        self.step_size = step_size

        # number of sub-sequences
        self.n_inputs = len(self.sequences)
        self.shape = None
        self.train_tuples = None

        # get dimensions of input sequences
        self.ndims = [self.sequences[i][0].shape[1] for i in xrange(self.n_inputs)]

        # get number of sequences
        self.n_sequences = len(self.sequences[0])

        # prepare data
        self.prepare_train_entities()

        # shuffle data
        self.shuffle()

    def prepare_train_entities(self):
        """
        Prepare train entities

        A train entity is a tuple such as (sequence_idx, sub_sequence_start_idx)
        """
        self.train_tuples = np.zeros((0, 2), dtype=np.int)

        # iterate all sequences
        for i in xrange(self.n_sequences):

            # compile current train entities
            sequence = self.sequences[0][i]
            tr_indices = np.arange(0, sequence.shape[0] - self.seq_len, self.step_size).astype(np.int)
            cur_entities = np.asarray(zip(np.repeat(i, len(tr_indices)), tr_indices), dtype=np.int)

            self.train_tuples = np.vstack((self.train_tuples, cur_entities))

        # number of train samples
        self.shape = [self.train_tuples.shape[0], self.seq_len]

    def shuffle(self, indices=None):
        """ Reset batch generator (random shuffle train data) """
        if indices is None:
            indices = np.random.permutation(self.shape[0])
        self.train_tuples = self.train_tuples[indices]

    def __getitem__(self, item):
        """ Make class accessible by index or slice """

        if item.__class__ == int:
            item = slice(item, item + 1)

        # get train tuples
        batch_entities = self.train_tuples[item]

        # initialize batch data (list of sequences)
        x_batch = [np.zeros((len(batch_entities), self.seq_len, dim), dtype=floatX) for dim in self.ndims]

        # collect batch data
        for i_batch_entity, (i_sequence, t) in enumerate(batch_entities):

            # collect sub-sequence
            for i_input in xrange(self.n_inputs):
                sequence = self.sequences[i_input][i_sequence]
                x_batch[i_input][i_batch_entity, :, :] = sequence[slice(t, t + self.seq_len)]

        # convert back to list
        x_batch = [[x[i] for i in xrange(x.shape[0])] for x in x_batch]

        return x_batch

    def __len__(self):
        return self.shape[0]


class SequenceDataPool(object):
    """
    Base Data Pool for Sequential Data
    """

    def __init__(self, sequence_list, max_len, use_mask=False):
        """
        Instantiates the Sequence Data Pool.

        Parameters
        ----------
        sequence_list : list containing lists of sequences
        max_len : integer
            maximum sequence length in data set
        """

        self.sequence_list = sequence_list

        self.shape = [self.sequence_list.shape[0]]
        self.max_len = max_len
        self.use_mask = use_mask

        self.n_inputs = self.sequence_list.n_inputs

        # get dimensions of individual inputs
        self.ndims = [self.sequence_list[i][0][0].shape[-1] for i in xrange(self.n_inputs)]

    def __getitem__(self, key):
        """
        Make class accessible by index or slice
        """

        # get batch
        if key.__class__ != slice:
            key = slice(key, key+1)

        # fix out of bounds
        end_idx = np.min([key.stop, self.sequence_list.shape[0]])
        key = slice(key.start, end_idx)

        batch_sequences = self.sequence_list[key]
        n_batch = key.stop - key.start

        x_batch = self.init_batch_data(n_batch)

        # collect batch data
        for i_sample in xrange(n_batch):
            self.set_batch_entry(i_sample, batch_sequences, x_batch)

        if self.use_mask:
            return list(x_batch)
        else:
            return [x_batch[0], x_batch[2]]

    def shuffle(self):
        """
        Reset batch generator (shuffle samples)
        """
        # TODO:
        pass

    def init_batch_data(self, nbatch):
        """
        Initialize batch data
        """
        pass

    def set_batch_entry(self, i_sample, i_sequence, target):
        """
        Set one sample of batch
        """
        pass


class LastStepOneHotPredictionDataPool(SequenceDataPool):
    """
    Produces data + last time step prediction targets for sequence data

    produces    x ... input sequence with shape (n_batch, max_len, n_features)
                m ... mask sequence with shape (n_batch, max_len)
                y ... integer target values (n_batch, )
    """

    def init_batch_data(self, n_batch):
        """ Initialize batch data """
        x_batch = np.zeros((n_batch, self.max_len-1, self.ndims[0]), dtype=np.float32)
        m_batch = np.zeros((n_batch, self.max_len-1), dtype=np.float32)
        y_batch = np.zeros(n_batch, dtype=np.int32)

        return x_batch, m_batch, y_batch

    def set_batch_entry(self, i_sample, sequence_list, x_batch):
        """ Set one sample of batch """

        # unwrap data
        x_batch, m_batch, y_batch = x_batch
        sequence = sequence_list[0][i_sample]

        x_batch[i_sample, 0:sequence.shape[0]-1, :] = sequence[0:-1]
        m_batch[i_sample, 0:sequence.shape[0] - 1] = 1
        y_batch[i_sample] = np.argmax(sequence[sequence.shape[0]-1])


class StepOneHotPredictionDataPool(SequenceDataPool):
    """
    Produces data + time step prediction targets for sequence data

    produces    x ... input sequence with shape (n_batch, max_len, n_features)
                m ... mask sequence with shape (n_batch, max_len)
                y ... integer target values (n_batch, max_len)
    """

    def init_batch_data(self, n_batch):
        """ Initialize batch data """
        x_batch = np.zeros((n_batch, self.max_len-1, self.ndims[0]), dtype=np.float32)
        m_batch = np.zeros((n_batch, self.max_len-1), dtype=np.float32)
        y_batch = np.zeros((n_batch, self.max_len-1), dtype=np.int32)

        return x_batch, m_batch, y_batch

    def set_batch_entry(self, i_sample, sequence_list, x_batch):
        """ Set one sample of batch """

        # unwrap data
        x_batch, m_batch, y_batch = x_batch
        sequence = sequence_list[0][i_sample]

        x_batch[i_sample, 0:sequence.shape[0]-1, :] = sequence[0:-1]
        m_batch[i_sample, 0:sequence.shape[0] - 1] = 1
        y_batch[i_sample, 0:sequence.shape[0]-1] = np.argmax(sequence[1::, :], axis=1)


class StepRegressionDataPool(SequenceDataPool):
    """
    Produces data + regression targets on time step level
    """

    def init_batch_data(self, n_batch):
        """ Initialize batch data """
        x_batch = np.zeros((n_batch, self.max_len, self.ndim), dtype=np.float32)
        m_batch = np.zeros((n_batch, self.max_len), dtype=np.float32)
        y_batch = np.zeros((n_batch, self.max_len, self.target_dim), dtype=np.float32)

        return x_batch, m_batch, y_batch

    def set_batch_entry(self, x_batch, m_batch, y_batch, i_sample, sequence, target):
        """ Set one sample of batch """
        x_batch[i_sample, 0:sequence.shape[0], :] = sequence
        m_batch[i_sample, 0:sequence.shape[0]] = 1
        y_batch[i_sample, 0:sequence.shape[0], :] = target.reshape((-1, self.target_dim))


class SequenceRegressionDataPool(SequenceDataPool):
    """
    Produces data + regression targets on sequence level
    """

    def init_batch_data(self, n_batch):
        """ Initialize batch data """
        x_batch = np.zeros((n_batch, self.max_len, self.ndim), dtype=np.float32)
        m_batch = np.zeros((n_batch, self.max_len), dtype=np.float32)
        y_batch = np.zeros((n_batch, self.target_dim), dtype=np.float32)

        return x_batch, m_batch, y_batch

    def set_batch_entry(self, x_batch, m_batch, y_batch, i_sample, sequence, target):
        """ Set one sample of batch """
        x_batch[i_sample, 0:sequence.shape[0], :] = sequence
        m_batch[i_sample, 0:sequence.shape[0]] = 1
        y_batch[i_sample] = target


class SequenceClfDataPool(SequenceDataPool):
    """
    Produces data + classification targets on sequence level
    """

    def init_batch_data(self, n_batch):
        """ Initialize batch data """
        x_batch = np.zeros((n_batch, self.max_len, self.ndim), dtype=np.float32)
        m_batch = np.zeros((n_batch, self.max_len), dtype=np.float32)
        y_batch = np.zeros((n_batch,), dtype=np.int32)

        return x_batch, m_batch, y_batch

    def set_batch_entry(self, x_batch, m_batch, y_batch, i_sample, sequence, target):
        """ Set one sample of batch """
        x_batch[i_sample, 0:sequence.shape[0], :] = sequence
        m_batch[i_sample, 0:sequence.shape[0]] = 1
        y_batch[i_sample] = target


class StepPredictionDataPool(SequenceDataPool):
    """
    Produces data + next step prediction targets on time step level
    """

    def init_batch_data(self, n_batch):
        """ Initialize batch data """
        x_batch = np.zeros((n_batch, self.max_len, self.ndim), dtype=np.float32)
        y_batch = np.zeros((n_batch, self.max_len, self.ndim), dtype=np.float32)
        m_batch = np.zeros((n_batch, self.max_len), dtype=np.float32)

        return x_batch, m_batch, y_batch

    def set_batch_entry(self, x_batch, m_batch, y_batch, i_sample, sequence, target=None):
        """ Set one sample of batch """
        x_batch[i_sample, 1:sequence.shape[0], :] = sequence[0:-1]
        y_batch[i_sample, 0:sequence.shape[0], :] = sequence
        m_batch[i_sample, 0:sequence.shape[0]] = 1


class LastStepPredictionDataPool(SequenceDataPool):
    """
    Produces data + next step prediction targets for only last value
    """

    def init_batch_data(self, n_batch):
        """ Initialize batch data """
        x_batch = np.zeros((n_batch, self.max_len, self.ndim), dtype=np.float32)
        y_batch = np.zeros((n_batch, self.ndim), dtype=np.float32)
        m_batch = np.zeros((n_batch, self.max_len), dtype=np.float32)

        return x_batch, m_batch, y_batch

    def set_batch_entry(self, x_batch, m_batch, y_batch, i_sample, sequence, target=None):
        """ Set one sample of batch """
        x_batch[i_sample, 0:sequence.shape[0]-1, :] = sequence[0:-1]
        y_batch[i_sample, :] = sequence[-1]
        m_batch[i_sample, 0:sequence.shape[0]-1] = 1


class StepRegressionDataPoolConv(SequenceDataPool):
    """
    Contains data + annotations on sequence level
    """

    def __init__(self, file_names, max_len, y_train=None, shuffle=True, temporal_context=1):
        super(StepRegressionDataPoolConv, self).__init__(file_names, max_len, y_train, shuffle)
        self.temporal_context = temporal_context

    def init_batch_data(self, nbatch):
        """
        Initialize batch data
        """
        x_batch = np.zeros((nbatch, self.max_len, 3 * self.ndim), dtype=np.float32)
        m_batch = np.zeros((nbatch, self.max_len), dtype=np.float32)
        y_batch = np.zeros((nbatch, self.max_len), dtype=np.float32)

        return x_batch, m_batch, y_batch

    def set_batch_entry(self, x_batch, m_batch, y_batch, i_sample, sequence, target):
        """
        Set one sample of batch
        """
        # collect temporal context vectors
        for t in xrange(sequence.shape[0]):

            start = np.max([0, t - self.temporal_context])
            stop = t + self.temporal_context + 1
            sl = slice(start, stop)
            spectrogram_excerpt = sequence[sl, :]

            if t - self.temporal_context < 0:
                missing = np.zeros((np.abs(t - self.temporal_context), sequence.shape[1]), dtype=np.float32)
                spectrogram_excerpt = np.vstack((missing, spectrogram_excerpt))

            if sl.stop > sequence.shape[0]:
                missing = np.zeros((sl.stop - sequence.shape[0], sequence.shape[1]), dtype=np.float32)
                spectrogram_excerpt = np.vstack((spectrogram_excerpt, missing))

            num_el = self.ndim * (self.temporal_context * 2 + 1)
            x_batch[i_sample, t, :] = spectrogram_excerpt.reshape((-1, num_el))

        # collect mask and ground truth
        m_batch[i_sample, 0:sequence.shape[0]] = 1
        y_batch[i_sample, 0:sequence.shape[0]] = target


if __name__ == '__main__':
    """ main """

    # generate test data
    sequences = []
    for _ in xrange(20):
        length = np.random.randint(20, 50)
        sequences.append(np.random.randn(length, 1))

    inp_sequences = sequences
    out_sequences = sequences

    seq_slicer = SubSequenceSlicer([inp_sequences], seq_len=13, step_size=2)
    data_pool = StepOneHotPredictionDataPool(seq_slicer, max_len=13, use_mask=True)
    x, m, y = data_pool[0:10]

    from batch_iterators import BatchIterator
    bi = BatchIterator(batch_size=10, k_samples=50)
    for epoch in xrange(10):
        for x, m, y in bi(data_pool):
            print(x.shape, m.shape, y.shape)
