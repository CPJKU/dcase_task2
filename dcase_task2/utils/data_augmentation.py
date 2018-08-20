
import numpy as np


def prepare(X, y):
    """ Prepare data for network processing """
    if len(X.shape) == 3:
        X = X[:, np.newaxis]
    X = X[:, 0:1, :, :].copy()
    return X, y


def prepare_spec_dims(X, y):
    if len(X.shape) == 3:
        X = X[:, np.newaxis]
    return X, y


def get_prepare_random_slice(n_frames, min_frames=None):

    def prepare_random_slice(X, y):

        if min_frames is not None:
            frames = np.random.randint(min_frames, n_frames + 1)
        else:
            frames = n_frames

        # apply random cyclic shift
        max_idx = X.shape[3] - frames
        X_new = np.zeros((X.shape[0], 1, X.shape[2], frames), dtype=np.float32)
        for i in xrange(X.shape[0]):
            start = np.random.randint(0, max_idx)
            stop = start + frames
            X_new[i] = X[i, :, :, start:stop]

        return X_new, y

    return prepare_random_slice


def get_prepare_one_hot(n_classes):

    def prepare_one_hot(X, y):
        batch_size = X.shape[0]

        one_hot = np.zeros((batch_size, n_classes), dtype=np.float32)
        for i in range(batch_size):
            one_hot[i, y[i]] = 1.0

        return X.astype(np.float32), one_hot.astype(np.float32)

    return prepare_one_hot


def get_prepare_mixup(n_classes, to_one_hot=True, alpha=0.2):

    def prepare_mixup(X, y):
        """
        Mixup data augmentation
        """
        batch_size, c, h, w = X.shape
        l = np.random.beta(alpha, alpha, batch_size)
        X_l = l.reshape(batch_size, 1, 1, 1)
        y_l = l.reshape(batch_size, 1)

        # mix observations
        X1, X2 = X[:], X[::-1]
        X = X1 * X_l + X2 * (1.0 - X_l)

        if to_one_hot:
            one_hot = np.zeros((batch_size, n_classes), dtype=np.float32)
            for i in range(batch_size):
                one_hot[i, y[i]] = 1.0
        else:
            one_hot = y

        # mix labels
        y1 = one_hot[:]
        y2 = one_hot[::-1]
        y = y1 * y_l + y2 * (1.0 - y_l)

        return X.astype(np.float32), y.astype(np.float32)

    return prepare_mixup


def get_prepare_concat_mixup(n_classes, to_one_hot=True, alpha=0.2, p=1.0):

    def prepare_concat_mixup(X, y):
        """
        Mixup data augmentation
        """
        batch_size, c, h, w = X.shape

        do_mixup = np.random.random() < p
        if do_mixup:
            l = np.random.beta(alpha, alpha, batch_size)
            y_l = l.reshape(batch_size, 1)

            X1 = X[:]
            X2 = X[::-1]
            w1 = int(w * (1.0 - alpha))
            X = np.concatenate((X1[:, :, :, :w1], X2[:, :, :, w1::]), axis=-1)

        if to_one_hot:
            one_hot = np.zeros((batch_size, n_classes), dtype=np.float32)
            for i in range(batch_size):
                one_hot[i, y[i]] = 1.0
        else:
            one_hot = y

        # mixup lables
        if do_mixup:
            y1 = one_hot[:]
            y2 = one_hot[::-1]
            y = y1 * y_l + y2 * (1 - y_l)
        else:
            y = one_hot

        return X.astype(np.float32), y.astype(np.float32)

    return prepare_concat_mixup


def get_prepare_random_erasing(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1.0/0.3, v_l=0.0, v_h=1.0):

    def prepare_random_erasing(X, y):
        """
        Random erasing data augmentation
        """

        def erase(input_img):

            _, img_h, img_w = input_img.shape

            # randomly apply random erasing
            if np.random.rand() > p:
                return input_img

            while True:
                s = np.random.uniform(s_l, s_h) * img_h * img_w
                r = np.random.uniform(r_1, r_2)
                w = int(np.sqrt(s / r))
                h = int(np.sqrt(s * r))

                left = np.random.randint(0, img_w)
                top = np.random.randint(0, img_h)

                if left + w <= img_w and top + h <= img_h:
                    break

            # get number random patch
            c = np.random.uniform(v_l, v_h)
            input_img[:, top:top + h, left:left + w] = c

            return input_img

        for i in range(X.shape[0]):
            X[i] = erase(X[i])

        return X.astype(np.float32), y

    return prepare_random_erasing


def get_prepare_event_oversampling(n_frames=128):

    def prepare_event_oversampling(X, y):

        # apply random cyclic shift
        X_new = np.zeros((X.shape[0], 1, X.shape[2], n_frames), dtype=np.float32)
        for i in xrange(X.shape[0]):

            # compute frame sample probabilities
            sample_probs = X[i, :, :, :].mean(axis=(0, 1))
            sample_probs -= sample_probs.min()
            sample_probs /= sample_probs.sum()

            # sample center frame
            center_frame = np.random.choice(range(X.shape[-1]), p=sample_probs)

            # set sample window
            start = center_frame - n_frames // 2
            start = np.clip(start, 0, X.shape[-1] - n_frames)
            stop = start + n_frames

            X_new[i] = X[i, :, :, start:stop]

        return X_new, y

    return prepare_event_oversampling
