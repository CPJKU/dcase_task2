
from __future__ import print_function

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dcase_task2.lasagne_wrapper.network import Network
from dcase_task2.lasagne_wrapper.batch_iterators import BatchIterator
from dcase_task2.lasagne_wrapper.utils import BColors

from config.settings import EXP_ROOT
from train import select_model, load_data, get_dump_file_paths
from utils.apk import mapk
from utils.data_tut18_task2 import load_data_lb, ID_CLASS_MAPPING

# bash coloring
col = BColors()


def print_result(fold, y, y_predicted, id_class_mapping):
    """ print result matrix """

    n_classes = len(np.unique(y))

    p, r, f, s = precision_recall_fscore_support(y, y_predicted, labels=None, pos_label=1, average=None)
    a = [(accuracy_score(y[y == c], y_predicted[y == c])) for c in xrange(n_classes)]

    # count occurrences of classes
    count = Counter(y)

    print("\n")
    if fold is not None:
        print("Results on fold %d" % fold)
    print("\n")
    print("%30s  |  %s  |  %5s  |  %4s  |  %4s  |   %4s   |" % ("LABEL", "CNT", "ACC ", "PR ", "RE ", "F1 "))
    print('-' * 70)
    for c in xrange(n_classes):
        print("%30s  |  %03d  |  %0.3f  |  %.2f  |  %.2f  |  %.3f   |" % (id_class_mapping[c], count[c], a[c], p[c], r[c], f[c]))
    print('-' * 70)
    print("%30s  |  %03d  |  %0.3f  |  %.2f  |  %.2f  |  %.3f   |" % ('average', len(y), np.mean(a), np.mean(p), np.mean(r), np.mean(f)))
    print('=' * 70)
    print("Overall Accuracy: %.3f %%" % (100.0 * accuracy_score(y, y_predicted)))
    print('=' * 70)


def prepare_slices(X, n_frames, n_slices=25):

    max_idx = X.shape[3] - n_frames - 1
    start_idxs = np.linspace(0, max_idx, n_slices).astype(np.int)
    X_new = np.zeros((n_slices, 1, X.shape[2], n_frames), dtype=np.float32)
    for i, start in enumerate(start_idxs):
        stop = start + n_frames
        X_new[i] = X[0, :, :, start:stop]

    return X_new


if __name__ == '__main__':
    """ main """
    # add argument parser
    parser = argparse.ArgumentParser(description='Train multi-modality model.')
    parser.add_argument('--model', help='select model to evaluate.')
    parser.add_argument('--params', help='select model parameters to evaluate (otherwise take defaults).', default=None)
    parser.add_argument('--data', help='select evaluation data.')
    parser.add_argument('--fold', help='train split.', type=int, default=0)
    parser.add_argument('--set', help='train, valid, test, private, public, private_public.', type=str, default="valid")
    parser.add_argument('--dump_results', help='dump results to pickl file.', action='store_true')
    parser.add_argument('--show', help='show spectrogram with prediction result.', action='store_true')
    parser.add_argument('--stats', help='create some prediction stats plots (only works with valid set).', action='store_true')
    parser.add_argument('--tag', help='add tag to result files.', type=str, default=None)
    parser.add_argument('--n_slices', help='perform sliding window forward passes.', type=int, default=None)

    # tut18 task2
    parser.add_argument('--train_file', help='train data file.', type=str, default="train.csv")
    parser.add_argument('--max_len', help='maximum spectrogram length.', type=int, default=None)
    parser.add_argument('--min_len', help='minimum spectrogram length.', type=int, default=None)
    parser.add_argument('--no_len_fix', help='fix lengths of spectrograms.', action='store_false')
    parser.add_argument('--train_on_all', help='use all files for training.', action='store_true')
    parser.add_argument('--validate_unverified', help='validate also on unverified samples.', action='store_true')

    args = parser.parse_args()

    # select model
    model = select_model(args.model)

    # load data
    print("Loading data ...")
    if args.set in ["private", "public", "private_public"]:
        spec_dir = args.data.split("-")[1]
        data = load_data_lb(spec_dir, lb_set=args.set)
        id_class_mapping = ID_CLASS_MAPPING
    else:
        data, id_class_mapping = load_data(args.data, args.fold, args=args)

    # set model dump file
    print("Loading model parameters ...")
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    dump_file, log_file = get_dump_file_paths(out_path, args.fold)

    # apply tag to dump files
    if args.tag:
        dump_file = dump_file.replace(".pkl", "_%s.pkl" % args.tag)
        log_file = log_file.replace(".pkl", "_%s.pkl" % args.tag)
        print("tagged parameter dump file %s" % dump_file)

    # overwrite model parameters
    if args.params:
        dump_file = args.params
        print("overwriting parameter dump file", os.path.basename(dump_file))

    # compile network
    net = model.build_model(batch_size=1)

    # initialize neural network
    my_net = Network(net, print_architecture=False)

    # load model parameters network
    my_net.load(dump_file)

    # init batch iterator
    bi = BatchIterator(batch_size=1, k_samples=None, shuffle=False, prepare=model.prepare)

    # iterate samples for prediction
    print("Predicting on test set ...")
    prediction_stats = []
    y_true, y_probs, y_predicted = [], [], []
    for i, (X, y) in enumerate(bi(data[args.set])):
        print("Processing file %d / %d" % (i + 1, data[args.set].shape[0]), end='\r')
        sys.stdout.flush()

        X_orig = X.copy()

        # fix spectrogram lengths
        if args.min_len:
            tmp = X.copy()
            while X.shape[-1] < args.min_len:
                X = np.concatenate((X, tmp), axis=-1)
        if args.max_len:
            X = X[:, :, :, 0:args.max_len]

        if args.n_slices:
            X_slc = prepare_slices(X, model.N_FRAMES, args.n_slices)
            p_pred = my_net.predict_proba(X_slc)
            p_pred = p_pred.mean(axis=0)
        else:
            p_pred = my_net.predict_proba(X)[0]

        # compute predicted class label
        y_pred = np.argmax(p_pred)

        # book keeping
        y_predicted.append(y_pred)
        y_probs.append(p_pred)
        y_true.append(y[0])

        # book keeping for further analysis
        stats = {
            "correct": y[0] == y_pred,
            "label": y[0],
            "prediction": y_pred,
            "length": X_orig.shape[-1],
        }
        prediction_stats.append(stats)

        if args.show and y[0] != y_pred:

            plt.figure("Predictions")
            plt.clf()

            plt.subplot(4, 1, 1)
            plt.imshow(X[0, 0], origin="lower", interpolation="nearest", cmap="viridis", aspect="auto")
            plt.title("t: %d, p: %d" % (y[0], y_pred))

            plt.subplot(4, 1, 2)
            plt.plot(p_pred, "o-")
            plt.ylim([0, 1])
            plt.grid("on")

            for j, idx in enumerate(np.linspace(0, X.shape[-1] - model.N_FRAMES, 5).astype(np.int)):
                X_ex = X[:, :, :, idx:idx + model.N_FRAMES]

                plt.subplot(4, 5, j + 11)
                plt.imshow(X_ex[0, 0], origin="lower", interpolation="nearest", cmap="viridis", aspect="auto")
                plt.subplot(4, 5, j + 16)
                plt.plot(my_net.predict_proba(X_ex)[0], "o-")
                plt.plot(y[0], 0, "ro")
                plt.ylim([0, 1])
                plt.grid("on")

            plt.show(block=True)

    # create some prediction analysis plots
    if args.stats:
        n_classes = len(np.unique(y_true))

        labels_correct = [st["label"] for st in prediction_stats if st["correct"]]
        labels_wrong = [st["label"] for st in prediction_stats if not st["correct"]]

        hist_c, bins = np.histogram(labels_correct, bins=range(0, 42))
        hist_w, bins = np.histogram(labels_wrong, bins=range(0, 42))

        fig_name = "correct-wrong-labels-%d-%s" % (args.fold, model.EXP_NAME)
        plt.figure(fig_name)
        plt.bar(bins[0:-1], hist_c, color="b", alpha=0.7)
        plt.bar(bins[0:-1], hist_w, bottom=hist_c, color="m", alpha=0.7)
        plt.grid("on")
        plt.xlim([0, n_classes])
        plt.title(fig_name)
        plt.savefig(fig_name + ".png")

        len_correct = [st["length"] for st in prediction_stats if st["correct"]]
        len_wrong = [st["length"] for st in prediction_stats if not st["correct"]]

        _, bins = np.histogram(len_correct + len_wrong, bins=100)
        bins = bins.astype(np.int)
        hist_c, _ = np.histogram(len_correct, bins=bins)
        hist_w, _ = np.histogram(len_wrong, bins=bins)

        fig_name = "correct-wrong-length-%d-%s" % (args.fold, model.EXP_NAME)
        plt.figure(fig_name)
        plt.subplots_adjust(bottom=0.2)
        plt.bar(range(len(hist_c)), hist_c, color="b", alpha=0.7)
        plt.bar(range(len(hist_c)), hist_w, color="m", alpha=0.7)
        plt.grid("on")
        plt.title(fig_name)
        plt.xticks(range(0, len(hist_c), 3), bins[0:-1:3], rotation='vertical')
        plt.savefig(fig_name + ".png")

        fig_name = "scatter-plot-%d-%s" % (args.fold, model.EXP_NAME)
        plt.figure(fig_name)
        for stats in prediction_stats:
            if stats["correct"]:
                plt.plot(stats["label"], stats["length"], "bo", alpha=0.5)
        for stats in prediction_stats:
            if not stats["correct"]:
                plt.plot(stats["label"], stats["length"], "mo", alpha=1.0)
        plt.grid("on")
        plt.xlim([-1, n_classes])
        plt.title(fig_name)
        plt.savefig(fig_name + ".png")

    # convert to arrays
    y_predicted = np.asarray(y_predicted, dtype=np.int32)
    y_probs = np.asarray(y_probs, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int32)

    # present results
    print_result(args.fold, y_true, y_predicted, id_class_mapping)

    # compute map@3
    predict_k = 3
    actual = [[y] for y in y_true]
    predicted = []
    for yp in y_probs:
        predicted.append(list(np.argsort(yp)[::-1][0:predict_k]))
    print("\n MAP@%d: %.3f" % (predict_k, mapk(actual, predicted, predict_k)))

    # dump result matrices
    if args.dump_results:

        # compile dump file name
        set_name = args.set if not args.validate_unverified else args.set + "all"
        file_name = os.path.basename(dump_file).replace("params", "probs_%s_%s" % (set_name, args.data))

        # dump results to model folder
        pkl_file = os.path.join(out_path, file_name)
        print("Dumping results to %s" % pkl_file)
        with open(pkl_file, "wb") as fp:
            pickle.dump({'y_probs': y_probs, 'y_true': y_true, 'files': data[args.set].files}, fp)
