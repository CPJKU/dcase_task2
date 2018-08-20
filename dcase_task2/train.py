
from __future__ import print_function

import os
import argparse
import numpy as np

from dcase_task2.lasagne_wrapper.network import Network

from utils.data_tut18_task2 import load_data as load_data_tut18_task2
from utils.data_tut18_task2 import ID_CLASS_MAPPING as id_class_mapping_tut18_task2

from config.settings import EXP_ROOT

# seed seed for reproducibility
np.random.seed(4711)


def select_model(model_path):
    """ select model """

    model_str = os.path.basename(model_path)
    model_str = model_str.split('.py')[0]
    import_root = ".".join((model_path.split(os.path.sep))[:-1])
    exec("from %s import %s as model" % (import_root, model_str))

    model.EXP_NAME = model_str
    return model


def load_data(data_set, fold, args):
    """ select data """

    if "tut18T2ver" in data_set:
        normalize = "norm" in data_set
        spec_dir = data_set.split("-")[1]
        data = load_data_tut18_task2(fold=fold, n_workers=1, spec_dir=spec_dir,
                                     train_verified=True, train_unverified=False, normalize=normalize,
                                     fix_lengths=args.no_len_fix, max_len=args.max_len, min_len=args.min_len,
                                     train_file=args.train_file, train_on_all=args.train_on_all,
                                     validate_verified=not args.validate_unverified)
        id_class_mapping = id_class_mapping_tut18_task2

    elif "tut18T2unver" in data_set:
        normalize = "norm" in data_set
        spec_dir = data_set.split("-")[1]
        data = load_data_tut18_task2(fold=fold, n_workers=1, spec_dir=spec_dir,
                                     train_verified=False, train_unverified=True, normalize=normalize,
                                     fix_lengths=args.no_len_fix, max_len=args.max_len, min_len=args.min_len,
                                     train_file=args.train_file, train_on_all=args.train_on_all,
                                     validate_verified=not args.validate_unverified)
        id_class_mapping = id_class_mapping_tut18_task2

    elif "tut18T2" in data_set:
        normalize = "norm" in data_set
        spec_dir = data_set.split("-")[1]
        data = load_data_tut18_task2(fold=fold, n_workers=1, spec_dir=spec_dir,
                                     train_verified=True, train_unverified=True, normalize=normalize,
                                     fix_lengths=args.no_len_fix, max_len=args.max_len, min_len=args.min_len,
                                     train_file=args.train_file, train_on_all=args.train_on_all,
                                     validate_verified=not args.validate_unverified)
        id_class_mapping = id_class_mapping_tut18_task2

    return data, id_class_mapping


def get_dump_file_paths(out_path, fold):
    par = 'params.pkl' if fold is None else 'params_%d.pkl' % fold
    log = 'results.pkl' if fold is None else 'results_%d.pkl' % fold
    dump_file = os.path.join(out_path, par)
    log_file = os.path.join(out_path, log)
    return dump_file, log_file


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Train audio tagging network.')
    parser.add_argument('--model', help='select model to train.')
    parser.add_argument('--data', help='select model to train.')
    parser.add_argument('--fold', help='train split.', type=int, default=None)
    parser.add_argument('--ini_params', help='path to pretrained parameters.', type=str, default=None)
    parser.add_argument('--tag', help='add tag to result files.', type=str, default=None)
    parser.add_argument('--fine_tune', help='use fine-tune train configuration.', action='store_true')

    # tut18 task2
    parser.add_argument('--train_file', help='train data file.', type=str, default="train.csv")
    parser.add_argument('--max_len', help='maximum spectrogram length.', type=int, default=None)
    parser.add_argument('--min_len', help='minimum spectrogram length.', type=int, default=None)
    parser.add_argument('--no_len_fix', help='do not fix lengths of spectrograms.', action='store_false')
    parser.add_argument('--train_on_all', help='use all files for training.', action='store_true')
    parser.add_argument('--validate_unverified', help='validate also on unverified samples.', action='store_true')

    args = parser.parse_args()

    # select model
    model = select_model(args.model)

    # load data
    print("\nLoading data ...")
    data, _ = load_data(args.data, args.fold, args)

    # set model dump file
    print("\nPreparing model ...")
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    dump_file, log_file = get_dump_file_paths(out_path, args.fold)

    # change parameter dump files
    if not args.fine_tune:
        dump_file = dump_file.replace(".pkl", "_it0.pkl")
        log_file = log_file.replace(".pkl", "_it0.pkl")
        print("parameter file", dump_file)
        print("log file", log_file)

    # compile network
    net = model.build_model()

    # initialize neural network
    my_net = Network(net)

    # load initial parametrization
    if args.ini_params:
        ini_params = args.ini_params % args.fold
        ini_params = dump_file.replace(os.path.basename(dump_file).split(".")[0], ini_params)
        my_net.load(ini_params)
        print("initial parameter file %s" % ini_params)

    # add tag to results
    if args.tag:
        dump_file = dump_file.replace(".pkl", "_%s.pkl" % args.tag)
        log_file = log_file.replace(".pkl", "_%s.pkl" % args.tag)
        print("tagged parameter file %s" % dump_file)

    # train network
    train_strategy = model.compile_train_strategy(args.fine_tune)
    my_net.fit(data, train_strategy, log_file=log_file, dump_file=dump_file)
