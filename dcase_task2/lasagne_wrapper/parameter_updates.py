
from lasagne.updates import momentum, adam, sgd, nesterov_momentum


def get_update_momentum(m=0.9):
    """
    Compute update with momentum
    """

    def update(all_grads, all_params, learning_rate):
        """ Compute updates from gradients """
        return momentum(all_grads, all_params, learning_rate, momentum=m)

    return update


def get_update_adam():
    """
    Compute update with momentum
    """

    def update(all_grads, all_params, learning_rate):
        """ Compute updates from gradients """
        return adam(all_grads, all_params, learning_rate)

    return update


def get_update_sgd():
    """
    Compute update with momentum
    """

    def update(all_grads, all_params, learning_rate):
        """ Compute updates from gradients """
        return sgd(all_grads, all_params, learning_rate)

    return update


def get_update_nesterov_momentum(m=0.9):
    """
    Compute update with nesterov momentum
    """

    def update(all_grads, all_params, learning_rate):
        return nesterov_momentum(all_grads, all_params, learning_rate,
                                 momentum=m)

    return update
