
from __future__ import print_function

import os
import sys
import pickle
import numpy as np
import argparse

from dcase_task2.lasagne_wrapper.network import Network
from dcase_task2.lasagne_wrapper.batch_iterators import BatchIterator

from config.settings import EXP_ROOT
from config.settings import DATA_ROOT as DATA_ROOT
from train import select_model
from utils.data_tut18_task2 import get_files_and_labels
from utils.data_tut18_task2 import load_data as load_data_tut18_task2
from utils.data_tut18_task2 import ID_CLASS_MAPPING as id_class_mapping
from utils.apk import mapk
from utils.data_augmentation import get_prepare_random_slice


def get_dump_file_paths(out_path, fold):
    par = 'params.pkl' if fold is None else 'params_%d.pkl' % fold
    log = 'results.pkl' if fold is None else 'results_%d.pkl' % fold
    dump_file = os.path.join(out_path, par)
    log_file = os.path.join(out_path, log)
    print("parameter dump file", dump_file)
    return dump_file, log_file


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Train multi-modality model.')
    parser.add_argument('--model', help='select model to train.', default=None)
    parser.add_argument('--spec_dir', help='spectrogram directory.', default="specs_train_processor4")
    parser.add_argument('--ver_thresh', help='verification threshold.', type=float, default=None)
    parser.add_argument('--top_probs_thresh', help='top probability verification threshold.', type=float, default=2.0)
    parser.add_argument('--k_per_class', help='select top samples per class.', type=int, default=0)
    parser.add_argument('--tag', help='add tag to result files.', type=str, default=None)
    parser.add_argument('--stochastic', help='perform stochastic forward passes.', action='store_true')
    parser.add_argument('--train_file', help='add tag to result files.', type=str, default="train_self_verified.csv")
    parser.add_argument('--valid_predictions', nargs='+')

    # tut18 task2
    parser.add_argument('--max_len', help='maximum spectrogram length.', type=int, default=None)
    parser.add_argument('--min_len', help='minimum spectrogram length.', type=int, default=None)
    parser.add_argument('--no_len_fix', help='fix lengths of spectrograms.', action='store_false')

    args = parser.parse_args()

    # get annotations
    tr_file = os.path.join(DATA_ROOT, "train.csv")
    files, labels, verified = get_files_and_labels(tr_file, spec_dir="specs_train_b128_h512")
    files = np.asarray([os.path.basename(f).split(".")[0] for f in files])

    # copy verification status
    verified_updated = verified.copy()
    verified_probs = np.ones((len(files), 41), dtype=np.float32)

    # select model
    if args.model:
        model = select_model(args.model)

        # prepare random slicing
        prepare_random_slice = get_prepare_random_slice(model.N_FRAMES)

    # iterate folds
    for fold in [1, 2, 3, 4]:

        # load data
        if not args.valid_predictions:
            print("\nLoading data of fold %d ..." % fold)

            data = load_data_tut18_task2(fold=fold, n_workers=1, spec_dir=args.spec_dir,
                                         train_verified=True, train_unverified=True, normalize=False,
                                         fix_lengths=args.no_len_fix, max_len=args.min_len, min_len=args.max_len,
                                         validate_verified=False, load_test=False)

            # compile network
            net = model.build_model()

            # initialize neural network
            my_net = Network(net, print_architecture=False)

            # set model dump file
            out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
            dump_file, log_file = get_dump_file_paths(out_path, fold)

            # apply tag to dump files
            if args.tag:
                dump_file = dump_file.replace(".pkl", "_%s.pkl" % args.tag)
                log_file = log_file.replace(".pkl", "_%s.pkl" % args.tag)
                print("tagged parameter dump file %s" % dump_file)

            # load model parameters network
            my_net.load(dump_file)

            # init batch iterator
            bi = BatchIterator(batch_size=1, k_samples=None, shuffle=False, prepare=model.prepare)

            # iterate validation data
            print("Predicting on validation set ...")
            y_true, y_pred, y_probs = [], [], []
            for i, (X, y) in enumerate(bi(data["valid"])):
                print("Processing file %d / %d" % (i + 1, data["valid"].shape[0]), end='\r')
                sys.stdout.flush()

                # get current file name
                cur_file = os.path.basename(data["valid"].files[i]).split(".")[0]
                file_idx = np.nonzero(files == cur_file)[0][0]
                cur_label = labels[file_idx]

                assert y[0] == cur_label, "label miss-match!"

                # skip officially verified files
                if verified[file_idx]:
                    continue

                # fix spectrogram lengths
                tmp = X.copy()
                while X.shape[-1] < args.min_len:
                    X = np.concatenate((X, tmp), axis=-1)
                X = X[:, :, :, 0:args.max_len]

                # predict on file
                if args.stochastic:
                    X_slc = []
                    for k in range(25):
                        X_slc.append(prepare_random_slice(X, y)[0])
                    X_slc = np.concatenate(X_slc, axis=0)
                    p_pred = my_net.predict_proba(X_slc)
                    p_pred = p_pred.mean(axis=0)
                    y_p = p_pred.argmax()
                else:
                    p_pred = my_net.predict_proba(X)[0]
                    y_p = p_pred.argmax()

                # update verification status
                y_true.append(y[0])
                y_pred.append(y_p)
                y_probs.append(p_pred)

                # keep probability of prediction
                verified_probs[file_idx] = p_pred

        else:

            # load and average all predictions
            y_probs = []
            for pred_file in args.valid_predictions:
                pred_file %= fold
                with open(pred_file, "rb") as fp:
                    p = pickle.load(fp)
                y_probs.append(p['y_probs'])
                fold_files = p['files']
                y_true = p['y_true']

            # average predictions
            y_probs = np.asarray(y_probs)
            y_probs = y_probs.mean(axis=0)

            for i, file_path in enumerate(fold_files):
                # get current file name
                cur_file = os.path.basename(file_path).split(".")[0]
                file_idx = np.nonzero(files == cur_file)[0][0]
                verified_probs[file_idx] = y_probs[i]

        # check performance of model
        actual = [[y] for y in y_true]
        predicted = []
        for yp in y_probs:
            predicted.append(list(np.argsort(yp)[::-1][0:3]))
        print("\nMAP@%d: %.3f" % (3, mapk(actual, predicted, 3)))

    # predict verification status
    if args.k_per_class:
        y_p = np.argmax(verified_probs, axis=1)
        for c in range(41):
            cls_idxs = np.nonzero((labels == c) & ~verified & (y_p == labels))[0]
            prob_sorted = np.argsort(verified_probs[cls_idxs, c])[::-1]
            prob_sorted = prob_sorted[0:min(len(prob_sorted), args.k_per_class)]
            sorted_probs = verified_probs[cls_idxs, c][prob_sorted]
            for k, idx in enumerate(prob_sorted):
                verified_updated[cls_idxs[idx]] = True
                if args.ver_thresh and sorted_probs[k] < args.ver_thresh:
                    verified_updated[cls_idxs[idx]] = False

    else:
        cls_counts = np.zeros(41)
        ver_idxs = np.nonzero(~verified)[0]
        for i in ver_idxs:
            p_pred = verified_probs[i]
            y_p = np.argmax(p_pred)

            has_label_match = y_p == labels[i]
            has_high_prob = p_pred[y_p] >= args.ver_thresh
            top_probs = np.sort(p_pred)[::-1][0:2]
            has_low_confusion = (top_probs[0] / top_probs[1]) > args.top_probs_thresh

            if has_label_match and has_high_prob and has_low_confusion:
                p_verified = True
                cls_counts[labels[i]] += 1
            else:
                p_verified = False

            if cls_counts[labels[i]] > args.k_per_class:
                p_verified = False

            # update global verification status
            verified_updated[i] = p_verified

    # check verification match
    orig_verification = verified[~verified]
    pred_verification = verified_updated[~verified]
    match = float(np.sum(orig_verification == pred_verification)) / len(orig_verification)
    ver_rate = float(np.sum(verified_updated != verified)) / len(orig_verification)
    print("")
    print("%d samples considered for verification." % len(orig_verification))
    print("Verification Match: %.3f" % match)
    print("Verification Rate:  %.3f" % ver_rate)

    # write verification file
    print("")
    print("%d file verified as good enough!" % np.sum(verified_updated != verified))

    l, c = np.unique(labels[verified_updated != verified], return_counts=True)
    for i in range(len(l)):
        print("%d: %d" % (l[i], c[i]))

    # write new txt file
    verified_updated = verified_updated.astype(np.int)
    txt_file = os.path.join(DATA_ROOT, args.train_file)
    with open(txt_file, "wb") as fp:
        # write header row
        fp.write("fname,label,manually_verified\n")

        for i in range(len(files)):
            line = "%s.wav,%s,%d\n" % (files[i], id_class_mapping[labels[i]], verified_updated[i])
            fp.write(line)
