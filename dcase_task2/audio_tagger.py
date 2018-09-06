
from __future__ import print_function

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from madmom.processors import IOProcessor, process_online

from prepare_spectrograms import processor_pipeline2
from lasagne_wrapper.network import Network
from config.settings import EXP_ROOT
from train import get_dump_file_paths, select_model
from utils.data_tut18_task2 import ID_CLASS_MAPPING


# initialize sliding window
SLIDING_WINDOW = np.zeros((128, 256), dtype=np.float32)
FRAME_COUNT = 0
TEXT_BOX = None
PREDICT_EVERY_K = None


def output_processor(data, output):
    """
    Output data processor
    """
    global FRAME_COUNT
    global TEXT_BOX
    global SLIDING_WINDOW
    global TAGGER
    global PREDICT_EVERY_K

    # check if there is audio content
    frame = data[0]
    if np.any(np.isnan(frame)):
        frame = np.zeros_like(frame, dtype=np.float32)

    # increase frame count
    FRAME_COUNT += 1
    FRAME_COUNT = np.mod(FRAME_COUNT, PREDICT_EVERY_K)
    do_predict = FRAME_COUNT == 0

    # update sliding window
    SLIDING_WINDOW[:, 0:-1] = SLIDING_WINDOW[:, 1::]
    SLIDING_WINDOW[:, -1] = frame

    # predict
    if do_predict:
        pred_spec = SLIDING_WINDOW[np.newaxis, np.newaxis]
        probs = TAGGER.predict_proba(pred_spec)[0]
        sorted_cls_idxs = np.argsort(probs)[::-1]

    # show current sliding window
    resz_spec = 2
    spec = SLIDING_WINDOW[::-1, :].copy() / 3.0
    spec = cv2.resize(spec, (spec.shape[1] * resz_spec, spec.shape[0] * resz_spec))
    spec = plt.cm.viridis(spec)[:, :, 0:3]
    spec = (spec * 255).astype(np.uint8)
    spec_rgb = cv2.cvtColor(spec, cv2.COLOR_RGB2BGR)
    if spec_rgb.shape[1] < 512:
        p = (512 - spec_rgb.shape[1]) // 2
        spec_rgb = np.pad(spec_rgb, ((0, 0), (p, p), (0, 0)), mode="constant")

    # visualize predictions
    if TEXT_BOX is None:
        TEXT_BOX = np.zeros_like(spec_rgb)
    font_face = cv2.FONT_HERSHEY_SIMPLEX

    if do_predict:
        TEXT_BOX = np.zeros_like(spec_rgb)

        if not np.sum(pred_spec) == 0:

            # show top labels
            for i in range(5):

                # plot label
                text = ID_CLASS_MAPPING[sorted_cls_idxs[i]]
                text_size = cv2.getTextSize(text, fontFace=font_face, fontScale=1.0, thickness=1)[0]
                text_org = (10, (text_size[1] + 20) * (i + 1))
                cv2.putText(TEXT_BOX, text, text_org, fontFace=font_face, fontScale=1.0, color=(255, 255, 255), thickness=1)

                # plot probability bar
                max_len = int(TEXT_BOX.shape[1] * 0.95)
                line_length = int(np.round(probs[sorted_cls_idxs[i]] * max_len))
                row_coord = text_org[1] + 10
                cv2.line(TEXT_BOX, (10, row_coord), (10 + line_length, row_coord), (255, 255, 0), 5)

        else:

            # show message
            for i, text in enumerate(["Waiting for", "spectrogram content ..."]):
                text_size = cv2.getTextSize(text, fontFace=font_face, fontScale=1.0, thickness=1)[0]
                text_org = (10, (text_size[1] + 20) * (i + 1))
                cv2.putText(TEXT_BOX, text, text_org, fontFace=font_face, fontScale=1.0, color=(255, 255, 255), thickness=1)

    # combine views
    screen_rgb = np.concatenate((spec_rgb, TEXT_BOX), axis=0)

    cv2.imshow("Machine Listener: General-Purpose Audio Tagger", screen_rgb)
    cv2.waitKey(1)


if __name__ == "__main__":
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Run live audio tagging system.')
    parser.add_argument('--model', help='Select model.', default='models/vgg_gap_spec2_no_pp.py')
    parser.add_argument('--params', help='Select model parameters to evaluate (otherwise take defaults).',
                        default=None)
    parser.add_argument('--predict_every_k', help='Update prediction every k frames.',
                        type=int, default=10)
    args = parser.parse_args()

    print("\nINFO: Make sure that you use a model trained on the correct spectrogram version." \
          "SOX preprocessing is not implemented for this live-audio-tagger."\
          "Otherwise the model's performance will be very poor.\n")

    # initialize network
    print("Initializing tagging network ...")
    model = select_model(args.model)
    net = model.build_model(batch_size=1)

    # initialize neural network
    TAGGER = Network(net, print_architecture=False)

    # load model parameters network
    if args.params is not None:
        dump_file = args.params
    else:
        out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
        dump_file, log_file = get_dump_file_paths(out_path, 1)
        dump_file = dump_file.replace(".pkl", "_it0.pkl")
    print("Loading model from: %s" % dump_file)
    TAGGER.load(dump_file)

    # set prediction rate
    PREDICT_EVERY_K = args.predict_every_k

    # dummy prediction to compile model
    print("Compiling prediction function ...")
    TAGGER.predict(SLIDING_WINDOW[np.newaxis, np.newaxis])

    print("Starting prediction loop ...")
    processor = IOProcessor(in_processor=processor_pipeline2, out_processor=output_processor)
    process_online(processor, infile=None, outfile=None, sample_rate=32000)
