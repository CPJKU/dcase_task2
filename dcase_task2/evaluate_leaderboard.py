
from __future__ import print_function

import os
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from dcase_task2.eval import print_result
from dcase_task2.utils.apk import mapk
from dcase_task2.utils.data_tut18_task2 import CLASS_ID_MAPPING, ID_CLASS_MAPPING
from dcase_task2.config.settings import DATA_ROOT as DATA_ROOT


def load_gt(lb_set="private"):
    """
    Load files of leaderboard splits
    """

    lb_set_list = lb_set.split("_")

    # load post competition ground truth
    with open(os.path.join(DATA_ROOT, "test_post_competition.csv"), 'r') as fp:
        train_list = fp.read()

    files = []
    labels = []
    for i, line in enumerate(train_list.split("\n")):

        if i == 0 or line == '':
            continue

        split_line = line.split(",")
        
        file_usage = split_line[2].lower()
        if file_usage not in lb_set_list:
            continue

        if split_line[0] != '':
            file_name = split_line[0].strip()

            files.append(file_name)
            labels.append(CLASS_ID_MAPPING[split_line[1].strip()])

        else:
            pass

    return np.asarray(files, dtype=np.string_), np.asarray(labels, dtype=np.int32)


def load_prediction(submission_file, file_list=None):
    """
    Load prediction (submission file)
    """
    with open(submission_file, 'r') as fp:
        train_list = fp.read()

    files = []
    labels = []
    for i, line in enumerate(train_list.split("\n")):

        if i == 0 or line == '':
            continue

        split_line = line.split(",")

        if split_line[0] != '':
            file_name = split_line[0].strip()
            
            if file_list is not None and file_name not in file_list:
                continue

            files.append(file_name)
            labels.append([CLASS_ID_MAPPING[l] for l in split_line[1].strip().split(" ")])

        else:
            pass

    return np.asarray(files, dtype=np.string_), np.asarray(labels, dtype=np.int)


def plot_confusion_matrix(preds, truth, normalize=False):
    """
    Create confusion matrix plot
    """
    
    cnf_matrix = confusion_matrix(truth, preds)

    # Concatenate labels of prediction and true arrays
    labels = list(np.unique(truth))
    for p in np.unique(preds):
        try:
            labels.index(p)
        except:
            labels.append(p)
    labels = sorted(labels)

    # Normalize confusion matrix
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

    # plot confusion matrix
    plt.figure("Confusion Matrix", figsize=(15, 15))
    plt.clf()
    plt.subplot(111)
    plt.subplots_adjust(bottom=0.13, top=0.98, right=0.99)
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    tick_marks = np.arange(len(labels))
    if len(labels) > 2:
        labels = [ID_CLASS_MAPPING[l] for l in labels]
    plt.xticks(tick_marks, labels, rotation="vertical")
    plt.yticks(tick_marks, labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Evaluate leaderboard submission.')
    parser.add_argument('--submission_file', help='path to submission file.', default=None)
    parser.add_argument('--set', help='leaderboard set (private, public, private_public).', type=str, default="private")
    args = parser.parse_args()
    
    gt_files, gt_labels = load_gt(lb_set=args.set)
    pr_files, pr_labels = load_prediction(submission_file=args.submission_file, file_list=gt_files)
    
    sorted_idx = np.argsort(gt_files)
    gt_files = gt_files[sorted_idx]
    gt_labels = gt_labels[sorted_idx]
    
    sorted_idx = np.argsort(pr_files)
    pr_files = pr_files[sorted_idx]
    pr_labels = pr_labels[sorted_idx]

    # print results to command line
    print_result(None, gt_labels, pr_labels[:, 0], ID_CLASS_MAPPING)
    actual = [[y] for y in gt_labels]
    print("MAP@%d: %.5f" % (3, mapk(actual, pr_labels, 3)))
    print('=' * 70)

    # create some evaluation plots
    print("Creating f-score plot (fscores.pdf)")
    n_classes = len(np.unique(y))
    p, r, f, s = precision_recall_fscore_support(gt_labels, pr_labels[:, 0], labels=None, pos_label=1, average=None)
    plt.figure("F-Score", figsize=(7, 4))
    plt.clf()
    plt.subplots_adjust(bottom=0.45, left=0.08, right=0.99, top=0.99)
    ax = plt.subplot(111)
    plt.bar(range(41), f)
    plt.xlim([-1, 41])
    ax.yaxis.grid(True)
    plt.ylabel("F-Score")
    plt.xticks(np.arange(41), [ID_CLASS_MAPPING[l].replace("_", " ") for l in range(41)], rotation='vertical')
    plt.savefig("fscores.pdf")

    print("Creating confusion matrix plot (cmatrix.pdf)")
    plot_confusion_matrix(pr_labels[:, 0], gt_labels)
    plt.savefig("cmatrix.pdf")
