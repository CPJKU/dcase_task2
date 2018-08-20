
import signal
import numpy as np
from multiprocessing import Pool


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class AugmentedAudioFileClassificationDataPool(object):
    """
    Data Pool for audio file classification
    """

    def __init__(self, files, targets, audio_processor, n_workers=1, shuffle=True, use_cache=False, n_classes=None,
                 target_type=np.int32, use_masks=False, mask_len=None):
        """
        Constructor
        """

        self.files = files
        self.targets = targets
        self.shape = [len(self.files)]
        self.use_masks = use_masks
        self.mask_len = mask_len

        self.audio_processor = audio_processor

        # initialize data pool
        self.n_workers = n_workers
        if self.n_workers > 1:
            self.pool = Pool(self.n_workers, init_worker)

        # initialize augmentation cache
        self.use_cache = use_cache
        self.cache = dict()

        # get number of classes
        self.n_classes = n_classes if n_classes is not None else len(np.unique(targets))
        self.target_type = target_type

        # check if files are audios or precomputed spectrograms
        if self.files[-1].endswith(".npy"):
            for file in self.files:
                self.cache[file] = np.load(file)

        # shuffle data
        if shuffle:
            self.shuffle()

    def shuffle(self):
        """ reset batch generator """
        indices = np.random.permutation(self.shape[0])
        self.files = self.files[indices]
        self.targets = self.targets[indices]

    def __getitem__(self, key):
        """ make class accessible by index or slice """

        # get batch
        if key.__class__ == int:
            indices = range(key, key + 1)
        elif key.__class__ == slice:
            indices = range(key.start, np.min([self.shape[0], key.stop]))
        else:
            indices = key

        # convert to list of requested indices
        indices = np.asarray(indices, dtype=np.int)

        # prepare list of files
        compute_file_list = []
        y = []
        for file_id in indices:

            if not self.use_cache or self.files[file_id] not in self.cache:
                compute_file_list.append(self.files[file_id])

            y.append(self.targets[file_id])

        # parallel computation of spectrogram for each file
        if self.n_workers > 1:
            ret_val = self.pool.map(self.audio_processor, compute_file_list)
        else:
            ret_val = []
            for file in compute_file_list:
                ret_val.append(self.audio_processor(file))

        # collect results from computation and cache
        X = []
        for file_id in indices:

            key = self.files[file_id]

            # check if spectrogram is in cache
            if key in self.cache:
                spec = self.cache[key]

            # spectrogram was computed
            else:
                spec = ret_val[compute_file_list.index(key)].astype(np.float32)

                # add spectrogram to cache
                if self.use_cache:
                    self.cache[key] = spec

            X.append(spec)

        # compute masks and pad spectrogram lengths
        if self.use_masks:
            m = []
            for i, spec in enumerate(X):

                # pad spectrogram
                n_missing = self.mask_len - spec.shape[-1]
                to_pad = [(0, 0) for _ in range(spec.ndim)]
                to_pad[-1] = (0, n_missing)
                spec = np.pad(spec, to_pad, mode="constant", constant_values=0)

                # compute mask
                tmp = np.zeros(self.mask_len, dtype=np.float32)
                tmp[0:spec.shape[-1]] = 1.0
                m.append(tmp)

        # assure that all spectrograms have the same number of frames
        spec_lengths = [len(spec) for spec in X]
        assert all(x == spec_lengths[0] for x in spec_lengths), "not all spectrograms have the same lengths (%s)" % str(np.unique(spec_lengths))

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=self.target_type)

        if self.use_masks:
            m = np.asarray(m, dtype=np.float32)
            return [X, m, y]
        else:
            return [X, y]


class AugmentedAudioFileMultiClassificationDataPool(AugmentedAudioFileClassificationDataPool):
    """
    Data Pool for multi-label audio file classification
    """

    def __getitem__(self, key):
        """ make class accessible by index or slice """

        # get batch
        if key.__class__ != slice:
            key = slice(key, key + 1)

        # fix out of bounds
        key = slice(key.start, np.min([self.shape[0], key.stop]))

        # get batch size
        batch_size = key.stop - key.start

        # prepare list of files
        compute_file_list = []
        y = np.zeros((batch_size, self.n_classes), dtype=np.float32)
        for idx, file_id in enumerate(range(key.start, key.stop)):

            if not self.use_cache or self.files[file_id] not in self.cache:
                compute_file_list.append(self.files[file_id])

            for t in self.targets[file_id]:
                y[idx, t] = 1

        # parallel computation of spectrogram for each file
        ret_val = self.pool.map(self.audio_processor, compute_file_list)

        # collect results from computation and cache
        X = []
        for file_id in range(key.start, key.stop):

            key = self.files[file_id]

            # check if spectrogram is in cache
            if key in self.cache:
                spec = self.cache[key]

            # spectrogram was computed
            else:
                spec = ret_val[compute_file_list.index(key)].astype(np.float32)

                # todo: fix this
                if spec.shape[0] < (2 * 313):
                    spec = np.pad(spec, ((0, (2 * 313) - spec.shape[0]), (0, 0)), mode='constant')

                # add spectrogram to cache
                if self.use_cache:
                    self.cache[key] = spec

            X.append(spec)

        X = np.asarray(X, dtype=np.float32)

        return [X, y]


class AugmentedRawAndSpecAudioFileClassificationDataPool(AugmentedAudioFileClassificationDataPool):
    """
    Data Pool for audio file classification
    """

    def __getitem__(self, key):
        """ make class accessible by index or slice """

        # get batch
        if key.__class__ != slice:
            key = slice(key, key + 1)

        # fix out of bounds
        key = slice(key.start, np.min([self.shape[0], key.stop]))

        # prepare list of files
        compute_file_list = []
        y = []
        for file_id in range(key.start, key.stop):

            if not self.use_cache or self.files[file_id] not in self.cache:
                compute_file_list.append(self.files[file_id])

            y.append(self.targets[file_id])

        # parallel computation of spectrogram for each file
        ret_val = self.pool.map(self.audio_processor, compute_file_list)

        # collect results from computation and cache
        X = []
        for file_id in range(key.start, key.stop):

            key = self.files[file_id]

            # check if spectrogram is in cache
            if key in self.cache:
                spec = self.cache[key]

            # spectrogram was computed
            else:
                spec = ret_val[compute_file_list.index(key)]

                # add spectrogram to cache
                if self.use_cache:
                    self.cache[key] = spec

            X.append(spec)

        # assure that all spectrograms have the same number of frames
        spec_lengths = [len(spec) for spec in X]
        assert all(x == spec_lengths[0] for x in spec_lengths), "not all spectrograms have the same lengths (%s)" % str(np.unique(spec_lengths))

        X = np.asarray(X, dtype=np.object)
        y = np.asarray(y, dtype=np.int32)

        return [X, y]
