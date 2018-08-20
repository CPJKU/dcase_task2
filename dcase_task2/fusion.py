
from __future__ import print_function

import os
import pickle
import theano
import argparse
import theano.tensor as T
import lasagne as lnn
import numpy as np
from tqdm import tqdm

from utils.apk import mapk


PATIENCE = 1000


def main():
    # load data
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_args = subparsers.add_parser('train')
    train_args.add_argument('--weights', type=str, required=True, help='Store weights to this file')
    train_args.add_argument('--max_steps', type=int, help='Maximum optimization steps for fusion', default=100000)
    train_args.add_argument('predictions', nargs='+')
    train_args.set_defaults(train=True)

    test_args = subparsers.add_parser('test')
    test_args.add_argument('--weights', type=str, help='Load weights from this file')
    test_args.add_argument('--out', type=str, help='Store predictions to this file')
    test_args.add_argument('predictions', nargs='+')
    test_args.set_defaults(train=False)
    args = parser.parse_args()

    X = []
    y_true = None
    files = None
    n_models = len(args.predictions)
    for pred_file in args.predictions:

        # load predicted probabilities
        with open(pred_file, "rb") as fp:
            p = pickle.load(fp)

        X.append(p['y_probs'])

        # sanity check if ground truth is the same
        if y_true is None:
            y_true = p['y_true']
        else:
            assert (y_true == p['y_true']).all()

        # sanity check if files are the same
        pf_files = [os.path.basename(f).strip() for f in p['files']]
        if files is None:
            files = pf_files
        else:
            assert (files == pf_files)

        print('Acc {}: {:.2f} ({} files)'.format(pred_file, (X[-1].argmax(1) == y_true).mean() * 100, len(p['y_true'])))

    # get number of classes
    N_CLASSES = X[0].shape[-1]
    n_observations = len(files)
    X = np.hstack(X).reshape(n_observations, n_models, N_CLASSES).astype(np.float32)

    # estimate label consensus
    print("\nEstimating label consensus ...")
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            consensus = float(np.sum(X[:, i, :].argmax(1) == X[:, j, :].argmax(1))) / X.shape[0]
            print("Consensus %d / %d: %.2f" % (i, j, consensus))

    # estimate performace
    pred_avg = X.mean(axis=1)
    print('\nAcc average: {:.2f}'.format((pred_avg.argmax(1) == y_true).mean() * 100))

    actual = [[y] for y in y_true]
    predicted = []
    for yp in pred_avg:
        predicted.append(list(np.argsort(yp)[::-1][0:3]))
    print("MAP@%d: %.3f" % (3, mapk(actual, predicted, 3)))

    x = T.tensor3('x')
    y = T.ivector('y')

    # model weights and class biases
    W = theano.shared(np.ones((n_models, 1)).astype(np.float32) / n_models, name='W')
    b = theano.shared(np.zeros(N_CLASSES).astype(np.float32), name='b')

    # compute fused prediction
    y_hat = lnn.nonlinearities.softmax(x.transpose(0, 2, 1).dot(W)[:, :, 0] + b)

    if args.train:
        loss = lnn.objectives.categorical_crossentropy(y_hat, y).mean()
        updates = lnn.updates.adam(loss, [W, b])
        train = theano.function(inputs=[x, y], outputs=loss, updates=updates)

        epochs = tqdm(range(args.max_steps))
        l_prev = np.inf
        wait = 0
        for _ in epochs:
            l = train(X, y_true)
            epochs.set_description('Loss: {:.5f}'.format(float(l)))
            if l < l_prev:
                l_prev = l
            else:
                wait += 1
                if wait > PATIENCE:
                    break

        pickle.dump([W.get_value(), b.get_value()],
                    open(args.weights, 'wb'))

    if not args.train and args.weights is not None:
        W_, b_ = pickle.load(open(args.weights, 'rb'))
        W.set_value(W_)
        b.set_value(b_)

    y_fused = y_hat.eval({x: X})
    print('\nAcc fused: {:.2f}'.format((y_fused.argmax(1) == y_true).mean() * 100))

    # compute map@3
    actual = [[y] for y in y_true]
    predicted = []
    for yp in y_fused:
        predicted.append(list(np.argsort(yp)[::-1][0:3]))
    print("MAP@%d: %.3f" % (3, mapk(actual, predicted, 3)))

    # dump fusion output
    if not args.train and args.out:
        with open(args.out, "wb") as fp:
            pickle.dump({'y_probs': y_fused, 'y_true': y_true, 'files': files}, fp)


if __name__ == "__main__":
    """ main """
    main()
