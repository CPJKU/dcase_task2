
import os
import glob as glob
import numpy as np

from data_pools import AugmentedAudioFileClassificationDataPool
from dcase_task2.config.settings import DATA_ROOT as DATA_ROOT

SPEC_CONTEXT = 128
SPEC_BINS = 128

N_WORKERS = 1
N_FRAMES = 128

classes = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime",
           "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano",
           "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica",
           "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors",
           "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle",
           "Writing"]

CLASS_ID_MAPPING = dict()
for i, cls in enumerate(classes):
    CLASS_ID_MAPPING[cls] = i
ID_CLASS_MAPPING = dict(zip(CLASS_ID_MAPPING.values(), CLASS_ID_MAPPING.keys()))


def get_files_and_labels(txt_file, spec_dir):
    """
    Load files of split
    """
    with open(txt_file, 'r') as fp:
        train_list = fp.read()

    files = []
    labels = []
    verified = []
    for i, line in enumerate(train_list.split("\n")):

        if i == 0:
            continue

        split_line = line.split(",")

        if split_line[0] != '':
            file_name = split_line[0].strip()
            file_path = os.path.join(DATA_ROOT, spec_dir, file_name).replace(".wav", ".npy")

            files.append(file_path)
            labels.append(CLASS_ID_MAPPING[split_line[1].strip()])
            verified.append(np.bool(split_line[2] == '1'))

        else:
            pass

    return np.asarray(files, dtype=np.string_), np.asarray(labels, dtype=np.int32), np.asarray(verified, dtype=np.bool)


def load_data(fold=1, n_workers=N_WORKERS, spec_dir="specs_train_v1", train_verified=True, train_unverified=True,
              normalize=False, fix_lengths=True, max_len=None, min_len=None, validate_verified=True,
              train_file="train.csv", load_test=True, train_on_all=False):
    """ load data """

    if not min_len:
        min_len = N_FRAMES

    # get annotations
    files, labels, verified = get_files_and_labels(os.path.join(DATA_ROOT, train_file), spec_dir)
    _, _, verified_for_val = get_files_and_labels(os.path.join(DATA_ROOT, "train.csv"), spec_dir)

    # stratified split
    np.random.seed(4711)
    r_idx = np.random.permutation(len(files))
    files, labels, verified = files[r_idx], labels[r_idx], verified[r_idx]
    verified_for_val = verified_for_val[r_idx]

    verified_indices = np.nonzero(verified)[0]
    unverified_indices = np.nonzero(~verified)[0]

    from sklearn.model_selection import StratifiedKFold
    sss = StratifiedKFold(n_splits=4, random_state=0)
    sss.get_n_splits(files[verified], labels[verified])
    for i_fold, (train_index_ver, test_index_ver) in enumerate(sss.split(files[verified], labels[verified])):
        if i_fold + 1 == fold:
            break

    sss = StratifiedKFold(n_splits=4, random_state=0)
    sss.get_n_splits(files[~verified], labels[~verified])
    for i_fold, (train_index_unver, test_index_unver) in enumerate(sss.split(files[~verified], labels[~verified])):
        if i_fold + 1 == fold:
            break

    train_index = np.concatenate((verified_indices[train_index_ver], unverified_indices[train_index_unver]))
    test_index = np.concatenate((verified_indices[test_index_ver], unverified_indices[test_index_unver]))

    if train_on_all:
        train_index = np.concatenate((train_index, test_index))

    # split into train and validation data
    tr_files, tr_labels, tr_verified = files[train_index], labels[train_index], verified[train_index]
    va_files, va_labels, va_verified = files[test_index], labels[test_index], verified_for_val[test_index]

    # load only verified labels
    train_idx = np.zeros_like(tr_verified, dtype=np.bool)
    if train_verified:
        train_idx = train_idx | tr_verified
    if train_unverified:
        train_idx = train_idx | (tr_verified == False)
    tr_files = tr_files[train_idx]
    tr_labels = tr_labels[train_idx]

    # keep only verified examples for validation
    if validate_verified:
        va_files = va_files[va_verified]
        va_labels = va_labels[va_verified]

    # create data pools
    pool = AugmentedAudioFileClassificationDataPool

    train_pool = pool(tr_files, tr_labels, None, n_workers=n_workers, shuffle=True, use_cache=True)
    valid_pool = pool(va_files, va_labels, None, n_workers=n_workers, shuffle=False, use_cache=True)

    if load_test:
        test_pool = load_data_test(spec_dir=spec_dir.replace("train", "test"))["test"]
    else:
        test_pool = None

    # fix spectrogram lengths
    print("Fixing spectrogram lengths ...")
    if max_len is None:
        max_len = np.max([s.shape[-1] for s in train_pool.cache.values()])

    def fix_pool(pool, test_mode):

        for k in pool.cache.keys():

            # copy spectrogram
            spec = pool.cache[k].copy()
            tmp = spec.copy()

            while spec.shape[-1] < max_len:

                if test_mode and spec.shape[-1] >= min_len:
                    break

                spec = np.concatenate((spec, tmp), axis=-1)

            # clip spectrogram if too long
            pool.cache[k] = spec[:, :, 0:max_len]

        return pool

    if fix_lengths:
        train_pool = fix_pool(train_pool, test_mode=False)
        valid_pool = fix_pool(valid_pool, test_mode=False)
        if load_test:
            test_pool = fix_pool(test_pool, test_mode=True)

    # normalize data
    if normalize:
        print("Normalizing data ...")

        specs = train_pool.cache.values()
        specs = np.concatenate(specs, axis=2).astype(np.float32)

        sub = specs.mean(axis=(0, 2), keepdims=True)[0]
        div = specs.std(axis=(0, 2), keepdims=True)[0]

        # sub = specs.min()
        # div = np.max(specs - sub)

        for key in train_pool.cache.keys():
            train_pool.cache[key] -= sub
            train_pool.cache[key] /= div

        for key in valid_pool.cache.keys():
            valid_pool.cache[key] -= sub
            valid_pool.cache[key] /= div

        if load_test:
            for key in test_pool.cache.keys():
                test_pool.cache[key] -= sub  # [0:1]
                test_pool.cache[key] /= div  # [0:1]

    print("Train %d" % train_pool.shape[0])
    print("Valid %d" % valid_pool.shape[0])
    if load_test:
        print("Test  %d" % test_pool.shape[0])

    return {'train': train_pool, 'valid': valid_pool, 'test': test_pool}


def load_data_test(spec_dir):

    # get test files
    split_dir = os.path.join(DATA_ROOT, spec_dir)
    te_files = glob.glob(split_dir + "/*.npy")
    te_files = np.asarray(te_files, dtype=np.string_)

    # create dummy labels
    te_labels = np.asarray([-1 for _ in te_files], dtype=np.int32)

    # init test set pool
    pool = AugmentedAudioFileClassificationDataPool
    test_pool = pool(te_files, te_labels, None, n_workers=1, shuffle=False, use_cache=True)

    return {'test': test_pool}


def load_data_lb(spec_dir, lb_set="private"):

    # select leader-board fractions
    set_list = lb_set.split("_")

    # load post competition test data ground truth
    with open(os.path.join(DATA_ROOT, "test_post_competition.csv"), 'r') as fp:
        test_list = fp.read()

    te_files, te_labels = [], []
    for i, line in enumerate(test_list.split("\n")):

        if i == 0 or line == '':
            continue

        split_line = line.split(",")

        file_usage = split_line[2]
        if file_usage.lower() not in set_list:
            continue

        if split_line[0] != '':
            file_name = split_line[0].strip()

            te_files.append(file_name)
            te_labels.append(CLASS_ID_MAPPING[split_line[1].strip()])

        else:
            pass

    # compile file paths
    split_dir = os.path.join(DATA_ROOT, spec_dir.replace("train", "test"))
    te_files = np.asarray([os.path.join(split_dir, f.replace(".wav", ".npy")) for f in te_files], dtype=np.string_)
    te_labels = np.asarray(te_labels, dtype=np.int32)

    # init test set pool
    pool = AugmentedAudioFileClassificationDataPool
    test_pool = pool(te_files, te_labels, None, n_workers=1, shuffle=False, use_cache=True)

    return {lb_set: test_pool}


if __name__ == '__main__':
    """ main """
    data = load_data(spec_dir="specs_train_rosa2", train_verified=True, train_unverified=True, fold=4,
                     max_len=None, min_len=None, fix_lengths=False)

    lengths = []
    s = "train"
    for i in range(data[s].shape[0]):
        X, y = data[s][i:i+1]
        lengths.append(X.shape[-1])

    lengths = np.asarray(lengths)

    print "Median: %.1f" % np.median(lengths)
    print "Mean:   %.1f" % np.mean(lengths)
    print "Min:    %.1f" % np.min(lengths)
    print "Max:    %.1f" % np.max(lengths)

    import matplotlib.pyplot as plt
    plt.figure("Snippet Lengths", figsize=(6, 4))
    plt.subplots_adjust(left=0.15, right=0.98, top=0.98)
    plt.clf()
    n, bins, patches = plt.hist(lengths, 300, normed=1, facecolor='gray', alpha=1.0)
    plt.xlabel("Spectrogram Length")
    plt.ylabel("Fraction of Samples")
    plt.grid("on")
    plt.savefig("snippet_lengths.pdf")
