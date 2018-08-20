#!/usr/bin/env python
import numpy as np
import theano.tensor as T

import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import batch_norm_dnn as batch_norm
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from lasagne.layers import DropoutLayer, FlattenLayer, GlobalPoolLayer, NonlinearityLayer

from dcase_task2.lasagne_wrapper.training_strategy import TrainingStrategy, RefinementStrategy
from dcase_task2.lasagne_wrapper.learn_rate_shedules import get_linear
from dcase_task2.lasagne_wrapper.parameter_updates import get_update_adam
from dcase_task2.lasagne_wrapper.batch_iterators import BatchIterator

from dcase_task2.utils.data_tut18_task2 import SPEC_BINS
from dcase_task2.utils.data_augmentation import prepare, get_prepare_mixup, get_prepare_one_hot, get_prepare_random_slice


N_FRAMES = 3*128
BATCH_SIZE = 100

INPUT_SHAPE = [1, SPEC_BINS, N_FRAMES]

init_conv = lasagne.init.HeNormal

prepare_mixup = get_prepare_mixup(41, to_one_hot=True, alpha=0.3)
prepare_one_hot = get_prepare_one_hot(41)
prepare_random_slice = get_prepare_random_slice(N_FRAMES)


def compile_train_strategy(fine_tune=False):
    """
    Compile training strategy either for initial training or for iterative fine-tuning
    """

    if fine_tune:
        ini_learning_rate = np.float32(0.0001)
        max_epochs = 30
        start_decay = 5
        best_model_by_accurary = False
    else:
        ini_learning_rate = np.float32(0.001)
        max_epochs = 500
        start_decay = 100
        best_model_by_accurary = False

    patience = max_epochs

    def prepare_valid(X, y):
        X, y = prepare(X, y)
        X, y = prepare_one_hot(X, y)
        return X, y

    def prepare_train(X, y):
        X, y = prepare_random_slice(X, y)
        if fine_tune:
            X, y = prepare_one_hot(X, y)
        else:
            X, y = prepare_mixup(X, y)
        return X.astype(np.float32), y.astype(np.float32)

    def get_valid_batch_iterator():
        """
        Get batch iterator
        """

        def batch_iterator(batch_size, k_samples, shuffle):
            return BatchIterator(batch_size=batch_size // 4, prepare=prepare_valid, k_samples=k_samples,
                                 shuffle=shuffle,
                                 fill_last_batch=False)

        return batch_iterator

    def get_train_batch_iterator():
        """
        Get batch iterator
        """

        def batch_iterator(batch_size, k_samples, shuffle):
            return BatchIterator(batch_size=batch_size, prepare=prepare_train, k_samples=k_samples, shuffle=shuffle)

        return batch_iterator

    def get_train_strategy():
        return TrainingStrategy(
            batch_size=BATCH_SIZE,
            ini_learning_rate=ini_learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            y_tensor_type=T.matrix,
            L2=None,
            adapt_learn_rate=get_linear(start_decay, ini_learning_rate, max_epochs-start_decay),
            update_function=get_update_adam(),
            valid_batch_iter=get_valid_batch_iterator(),
            train_batch_iter=get_train_batch_iterator(),
            shuffle_train=True,
            best_model_by_accurary=best_model_by_accurary,
            refinement_strategy=RefinementStrategy(n_refinement_steps=0, refinement_patience=0, learn_rate_multiplier=0.0))

    return get_train_strategy()


def build_model(batch_size=BATCH_SIZE):
    """ Compile net architecture """
    nonlin = lasagne.nonlinearities.rectify

    # --- input layers ---
    l_in = lasagne.layers.InputLayer(shape=(None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]), name='Input')
    net = l_in

    nf = 64

    # --- conv layers ---
    net = Conv2DLayer(net, num_filters=nf, filter_size=5, stride=2, pad=2, W=init_conv(gain="relu"), nonlinearity=nonlin)
    net = batch_norm(net, alpha=0.1)
    net = Conv2DLayer(net, num_filters=nf, filter_size=3, stride=1, pad=1, W=init_conv(gain="relu"), nonlinearity=nonlin)
    net = batch_norm(net, alpha=0.1)
    net = MaxPool2DLayer(net, pool_size=2)
    net = DropoutLayer(net, p=0.3)

    net = Conv2DLayer(net, num_filters=2*nf, filter_size=3, stride=1, pad=1, W=init_conv(gain="relu"), nonlinearity=nonlin)
    net = batch_norm(net, alpha=0.1)
    net = Conv2DLayer(net, num_filters=2*nf, filter_size=3, stride=1, pad=1, W=init_conv(gain="relu"), nonlinearity=nonlin)
    net = batch_norm(net, alpha=0.1)
    net = MaxPool2DLayer(net, pool_size=2)
    net = DropoutLayer(net, p=0.3)

    net = Conv2DLayer(net, num_filters=4*nf, filter_size=3, stride=1, pad=1, W=init_conv(gain="relu"), nonlinearity=nonlin)
    net = batch_norm(net, alpha=0.1)
    net = DropoutLayer(net, p=0.3)
    net = Conv2DLayer(net, num_filters=4*nf, filter_size=3, stride=1, pad=1, W=init_conv(gain="relu"), nonlinearity=nonlin)
    net = batch_norm(net, alpha=0.1)
    net = DropoutLayer(net, p=0.3)
    net = Conv2DLayer(net, num_filters=6*nf, filter_size=3, stride=1, pad=1, W=init_conv(gain="relu"), nonlinearity=nonlin)
    net = batch_norm(net, alpha=0.1)
    net = DropoutLayer(net, p=0.3)
    net = Conv2DLayer(net, num_filters=6*nf, filter_size=3, stride=1, pad=1, W=init_conv(gain="relu"), nonlinearity=nonlin)
    net = batch_norm(net, alpha=0.1)
    net = MaxPool2DLayer(net, pool_size=2)
    net = DropoutLayer(net, p=0.3)

    net = Conv2DLayer(net, num_filters=8 * nf, filter_size=3, stride=1, pad=1, W=init_conv(gain="relu"), nonlinearity=nonlin)
    net = batch_norm(net, alpha=0.1)
    net = Conv2DLayer(net, num_filters=8 * nf, filter_size=3, stride=1, pad=1, W=init_conv(gain="relu"), nonlinearity=nonlin)
    net = batch_norm(net, alpha=0.1)
    net = MaxPool2DLayer(net, pool_size=(1, 2))
    net = DropoutLayer(net, p=0.3)

    net = Conv2DLayer(net, num_filters=8 * nf, filter_size=3, stride=1, pad=1, W=init_conv(gain="relu"), nonlinearity=nonlin)
    net = batch_norm(net, alpha=0.1)
    net = Conv2DLayer(net, num_filters=8 * nf, filter_size=3, stride=1, pad=1, W=init_conv(gain="relu"), nonlinearity=nonlin)
    net = batch_norm(net, alpha=0.1)
    net = MaxPool2DLayer(net, pool_size=(1, 2))
    net = DropoutLayer(net, p=0.3)

    net = Conv2DLayer(net, num_filters=8*nf, filter_size=3, pad=0, W=init_conv(gain="relu"), nonlinearity=nonlin)
    net = batch_norm(net, alpha=0.1)
    net = DropoutLayer(net, p=0.5)
    net = Conv2DLayer(net, num_filters=8*nf, filter_size=1, pad=0, W=init_conv(gain="relu"), nonlinearity=nonlin)
    net = batch_norm(net, alpha=0.1)
    net = DropoutLayer(net, p=0.5)

    # --- feed forward part ---
    net = Conv2DLayer(net, num_filters=41, filter_size=1, W=init_conv(gain="relu"), nonlinearity=None)
    net = batch_norm(net, alpha=0.1)
    net = GlobalPoolLayer(net)
    net = FlattenLayer(net)
    net = NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.softmax)

    return net


if __name__ == "__main__":
    from dcase_task2.lasagne_wrapper.utils import print_net_architecture
    print_net_architecture(build_model(None), detailed=True)
