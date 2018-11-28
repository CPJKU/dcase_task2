
from __future__ import print_function

import argparse

import os
import time
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor, LogarithmicFilteredSpectrogramProcessor
from madmom.audio.filters import LogFilterbank
from madmom.processors import SequentialProcessor, Processor


class LibrosaProcessor(Processor):

    def __init__(self):
        pass

    def process(self, file_path, **kwargs):
        n_fft = 1024
        sr = 32000
        mono = True
        log_spec = False
        n_mels = 128

        hop_length = 192
        fmax = None

        if mono:
            sig, sr = librosa.load(file_path, sr=sr, mono=True)
            sig = sig[np.newaxis]
        else:
            sig, sr = librosa.load(file_path, sr=sr, mono=False)
            # sig, sf_sr = sf.read(file_path)
            # sig = np.transpose(sig, (1, 0))
            # sig = np.asarray([librosa.resample(s, sf_sr, sr) for s in sig])

        spectrograms = []
        for y in sig:

            # compute stft
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True,
                                pad_mode='reflect')

            # keep only amplitures
            stft = np.abs(stft)

            # spectrogram weighting
            if log_spec:
                stft = np.log10(stft + 1)
            else:
                freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
                stft = librosa.perceptual_weighting(stft**2, freqs, ref=1.0, amin=1e-10, top_db=99.0)

            # apply mel filterbank
            spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax)

            # keep spectrogram
            spectrograms.append(np.asarray(spectrogram))

        spectrograms = np.asarray(spectrograms)

        return spectrograms

processor_version1 = LibrosaProcessor()

sig_proc = SignalProcessor(num_channels=1, sample_rate=32000, norm=True)
fsig_proc = FramedSignalProcessor(frame_size=1024, hop_size=128, origin='future')
spec_proc = SpectrogramProcessor(frame_size=1024)
filt_proc = LogarithmicFilteredSpectrogramProcessor(filterbank=LogFilterbank, num_bands=26, fmin=20, fmax=14000)
processor_pipeline2 = [sig_proc, fsig_proc, spec_proc, filt_proc]
processor_version2 = SequentialProcessor(processor_pipeline2)


if __name__ == "__main__":
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Pre-compute spectrograms for training and testing.')
    parser.add_argument('--audio_path', help='path to audio files.')
    parser.add_argument('--spec_path', help='path where to store spectrograms.')
    parser.add_argument('--show', help='show spectrogram plots.', type=int, default=None)
    parser.add_argument('--dump', help='dump spectrograms.', action='store_true')
    parser.add_argument('--spec_version', help='spectrogram version to compute (1 or 2).', type=int, default=1)
    parser.add_argument('--no_preprocessing', help='compute spectrogram for original audios.', action='store_true')
    args = parser.parse_args()

    # create destination directory
    if not os.path.exists(args.spec_path):
        os.makedirs(args.spec_path)

    # select spectrogram processor
    proc = processor_version1 if args.spec_version == 1 else processor_version2

    # get timestamp
    tstp = str(time.time())

    # get list of audio files
    file_list = glob.glob(os.path.join(args.audio_path, "*.wav"))

    if args.show:
        file_list = np.asarray(file_list)

    # compute spectrograms
    for i, file in enumerate(file_list):
        print("%04d / %d | %s" % (i + 1, len(file_list), file))

        # show at max args.show spectrograms
        if args.show and i >= args.show:
            break

        # use original audio
        if args.no_preprocessing:
            aug_audio_file = file
        # sox silence clipping
        else:
            aug_cmd = "norm -0.1 silence 1 0.025 0.15% norm -0.1 reverse silence 1 0.025 0.15% reverse"
            aug_audio_file = "%stmp.wav" % tstp
            os.system("sox %s %s %s" % (file, aug_audio_file, aug_cmd))

            assert os.path.exists(aug_audio_file), "SOX Problem ... clipped wav does not exist!"

        # compute spectrogram
        try:
            spectrogram = proc.process(aug_audio_file)
        except:
            print("Audio clipping failed! Computing spectrogram on original audio.")
            try:
                spectrogram = proc.process(file)
            except Exception:
                print("Could not process %s. Ignoring." % file)
                os.remove(aug_audio_file)
                continue

        # fix spectrogram dimensions
        if spectrogram.ndim == 2:
            spectrogram = spectrogram.T[np.newaxis]

        # clean up temporary files
        os.remove(aug_audio_file)

        # plot spectrogram
        if args.show:

            print("Spectrogram Shape:", spectrogram.shape)

            plt.figure("General-Purpose ")
            plt.clf()
            plt.subplots_adjust(right=0.98, left=0.1, bottom=0.1, top=0.99)
            plt.imshow(spectrogram[0], origin="lower", interpolation="nearest", cmap="viridis")
            plt.xlabel("%d frames" % spectrogram.shape[2])
            plt.ylabel("%d bins" % spectrogram.shape[1])
            plt.colorbar()
            plt.show()

            plt.show(block=True)

        # save spectrograms
        if args.dump:
            spec_file = os.path.join(args.spec_path, os.path.basename(file).replace(".wav", ".npy"))
            np.save(spec_file, spectrogram)
