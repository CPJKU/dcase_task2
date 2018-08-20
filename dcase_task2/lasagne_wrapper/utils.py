
from __future__ import print_function

import types
import pickle
import numpy as np


# init color printer
class BColors:
    """
    Colored command line output formatting
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self):
        """ Constructor """
        pass

    def print_colored(self, string, color):
        """ Change color of string """
        return color + string + BColors.ENDC


def pb(s):
    col = BColors()
    return col.print_colored(s, BColors.OKBLUE)


def print_net_architecture(net, tag=None, detailed=False):
    """ Print network architecture """
    import lasagne

    col = BColors()
    print('\n')

    if tag is not None:
        print(col.print_colored('Net-Architecture: %s' % tag, BColors.UNDERLINE))
    else:
        print(col.print_colored('Net-Architecture:', BColors.UNDERLINE))

    layers = lasagne.layers.helper.get_all_layers(net)
    max_len = np.max([len(l.__class__.__name__) for l in layers]) + 7
    for l in layers:
        class_name = l.__class__.__name__
        output_shape = str(l.output_shape)

        if isinstance(l, lasagne.layers.DropoutLayer):
            class_name += "(%.2f)" % l.p

        class_name = class_name.ljust(max_len)
        output_shape = output_shape.ljust(25)

        if isinstance(l, lasagne.layers.InputLayer):
            class_name = col.print_colored(class_name, BColors.OKBLUE)

        if isinstance(l, lasagne.layers.MergeLayer):
            class_name = col.print_colored(class_name, BColors.WARNING)

        layer_details = ""
        if detailed:
            layer_details = []

            # add nonlinearity
            if hasattr(l, "nonlinearity"):
                if isinstance(l.nonlinearity, types.FunctionType):
                    layer_details.append(pb("NL: ") + str(l.nonlinearity.__name__))
                else:
                    layer_details.append(pb("NL: ") + str(l.nonlinearity.__class__.__name__))

            # print weight shape if possible
            if hasattr(l, 'W'):
                weight_shape = str(l.W.get_value().shape)
                layer_details.append(pb("W: ") + weight_shape)

            # print bias shape if possible
            if hasattr(l, 'b'):
                bias_shape = str(l.b.get_value().shape) if l.b is not None else "None"
                layer_details.append(pb("b: ") + bias_shape)

            # print scaler shape if possible
            if hasattr(l, 'gamma'):
                bias_shape = str(l.beta.get_value().shape) if l.beta is not None else "None"
                layer_details.append(pb("gamma: ") + bias_shape)

            # print bias shape if possible
            if hasattr(l, 'beta'):
                bias_shape = str(l.beta.get_value().shape) if l.beta is not None else "None"
                layer_details.append(pb("beta: ") + bias_shape)

            layer_details = ", ".join(layer_details)

        print(class_name, output_shape, layer_details)

    # print total number of parameters
    n_params = lasagne.layers.helper.count_params(net)
    print("\nTotal number of parameters: %d" % n_params)


class EpochLogger(object):
    """
    Convenient logging of epoch stats
    """

    def __init__(self):

        self.epoch_stats = dict()
        self.within_epoch_stats = dict()
        self.previous_epoch = None

    def append(self, key, value):

        if key not in self.within_epoch_stats:
            self.within_epoch_stats[key] = []

        self.within_epoch_stats[key].append(value)

    def summarize_epoch(self):

        self.previous_epoch = self.within_epoch_stats.copy()

        for key, values in self.within_epoch_stats.iteritems():

            if key not in self.epoch_stats:
                self.epoch_stats[key] = []

            self.epoch_stats[key].append(np.mean(values))
            self.within_epoch_stats[key] = []

    def summarize_epoch_py3(self):

        self.previous_epoch = self.within_epoch_stats.copy()

        for key, values in self.within_epoch_stats.items():

            if key not in self.epoch_stats:
                self.epoch_stats[key] = []

            self.epoch_stats[key].append(np.mean(values))
            self.within_epoch_stats[key] = []

    def dump(self, dump_file):
        with open(dump_file, "wb") as fp:
            pickle.dump(self.epoch_stats, fp, -1)

    def load(self, dump_file):
        with open(dump_file, "rb") as fp:
            self.epoch_stats = pickle.load(fp)

if __name__ == "__main__":
    """ main """

    # test epoch logger
    import numpy as np
    epoch_logger = EpochLogger()
    for epoch in xrange(10):
        for iteration in xrange(100):
            epoch_logger.append("loss", np.random.randn(1))
            epoch_logger.append("acc", 100 * np.random.randn(1))

        epoch_logger.summarize_epoch()

    epoch_logger.dump("test.pkl")

    epoch_logger = EpochLogger()
    epoch_logger.load("test.pkl")
    print(epoch_logger.epoch_stats)
