
from __future__ import print_function

from lasagne.init import floatX
import theano.tensor as T

from dcase_task2.lasagne_wrapper.optimization_objectives import mean_categorical_crossentropy, mean_pixel_binary_crossentropy, \
    mean_pixel_categorical_crossentropy
from dcase_task2.lasagne_wrapper.learn_rate_shedules import get_constant
from dcase_task2.lasagne_wrapper.parameter_updates import get_update_adam
from dcase_task2.lasagne_wrapper.batch_iterators import get_batch_iterator


def get_classification_TrainingStrategy(**kwargs):
    """
    Defines training strategy for neural network
    """
    return TrainingStrategy(**kwargs)


def get_binary_segmentation_TrainingStrategy(**kwargs):
    """
    Defines training strategy for neural network
    """    
    return TrainingStrategy(y_tensor_type=T.tensor4, objective=mean_pixel_binary_crossentropy, report_dices=True, **kwargs)


def get_categorical_segmentation_TrainingStrategy(**kwargs):
    """
    Defines training strategy for neural network
    """    
    return TrainingStrategy(y_tensor_type=T.itensor4, objective=mean_pixel_categorical_crossentropy, **kwargs)


def get_next_step_one_hot_TrainingStrategy(**kwargs):
    """
    Defines training strategy for next step prediction of classes (e.g. one hot vector character rnns)
    """
    return TrainingStrategy(**kwargs)


class RefinementStrategy(object):
    """
    Defines refinement strategy for neural network
    Once a model does not improve anymore during training this will be applied
    """

    def __init__(self, n_refinement_steps=2, refinement_patience=5, learn_rate_multiplier=0.1):
        """
        Constructor
        """
        self.n_refinement_steps = n_refinement_steps
        self.refinement_patience = refinement_patience
        self.learn_rate_multiplier = learn_rate_multiplier

    def adapt_learn_rate(self, lr):
        """ Update learning rate """
        self.n_refinement_steps -= 1
        return floatX(self.learn_rate_multiplier * lr)


class TrainingStrategy(object):
    """
    Defines training strategy for neural network
    """

    def __init__(self, batch_size=100, ini_learning_rate=0.001, max_epochs=100, patience=10, y_tensor_type=T.ivector,
                 L2=1e-4, objective=mean_categorical_crossentropy, adapt_learn_rate=get_constant(),
                 update_function=get_update_adam(), valid_batch_iter=get_batch_iterator(),
                 train_batch_iter=get_batch_iterator(), use_weights=False, samples_per_epoch=None,
                 shuffle_train=True, report_dices=False, refinement_strategy=RefinementStrategy(),
                 best_model_by_accurary=False, debug_mode=False, layer_update_filter=None, report_map=3):
        """
        Constructor
        """
        self.batch_size = batch_size
        self.ini_learning_rate = ini_learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.y_tensor_type = y_tensor_type
        self.L2 = L2
        self.objective = objective
        self.adapt_learn_rate = adapt_learn_rate
        self.update_function = update_function
        self.valid_batch_iter = valid_batch_iter
        self.train_batch_iter = train_batch_iter
        self.use_weights = use_weights
        self.samples_per_epoch = samples_per_epoch
        self.shuffle_train = shuffle_train
        self.report_dices = report_dices
        self.refinement_strategy = refinement_strategy
        self.best_model_by_accurary = best_model_by_accurary
        self.debug_mode = debug_mode
        self.layer_update_filter = layer_update_filter
        self.report_map = report_map

    def update_learning_rate(self, lr, epoch):
        """ Update learning rate """
        return self.adapt_learn_rate(lr, epoch)

    def update_parameters(self, all_grads, all_params, learning_rate):
        """ Compute updates from gradients """
        return self.update_function(all_grads, all_params, learning_rate)

    def build_valid_batch_iterator(self):
        """ Compile batch iterator """
        return self.valid_batch_iter(self.batch_size, k_samples=None, shuffle=False)

    def build_train_batch_iterator(self):
        """ Compile batch iterator """
        return self.train_batch_iter(self.batch_size, k_samples=self.samples_per_epoch, shuffle=self.shuffle_train)
