
from __future__ import print_function

import os
import pickle
import argparse
import numpy as np

from utils.data_tut18_task2 import ID_CLASS_MAPPING

if __name__ == "__main__":
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Generate DACASE submission file.')
    parser.add_argument('--prediction_file', help='prediction file.')
    parser.add_argument('--out_file', help='name of submission.', default=None)
    args = parser.parse_args()

    # check if outfile is specified
    if args.out_file is None:
        print("Stopping script: Outfile not specified!")
        exit(0)

    # load prediction
    with open(args.prediction_file, "rb") as fp:
        prediction = pickle.load(fp)

    # get file names and predictions
    files_names = prediction['files']
    y_probs = prediction['y_probs']

    # write submission file
    with open(args.out_file, 'wb') as fp:

        # write header
        fp.write("fname,label\n")

        # iterate files
        for i in range(len(files_names)):
            file_name = os.path.basename(files_names[i]).replace(".npy", ".wav")

            labels = list(np.argsort(y_probs[i])[::-1][0:3])
            tags = [ID_CLASS_MAPPING[l] for l in labels]
            tags = " ".join(tags)

            entry = "%s,%s\n" % (file_name, tags)
            fp.write(entry)
    print("Submission written to: %s" % args.out_file)
