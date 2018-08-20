
import lasagne
import theano.tensor as T


def mean_categorical_crossentropy(pred, targ, weight=None):
    _EPSILON = 10e-8
    pred = T.clip(pred, _EPSILON, 1.0 - _EPSILON)
    return T.mean(lasagne.objectives.categorical_crossentropy(pred, targ))


def time_distributed_mean_categorical_crossentropy(pred, targ, weight=None):

    pred = T.reshape(pred, (-1, pred.shape[-1]))

    if targ.ndim == 3:
        targ = T.reshape(targ, (-1, targ.shape[-1]))
    else:
        targ = targ.flatten()

    _EPSILON = 10e-8
    pred = T.clip(pred, _EPSILON, 1.0 - _EPSILON)
    
    loss = lasagne.objectives.categorical_crossentropy(pred, targ)
    
    if weight is not None:
        weight = weight.flatten()
        return lasagne.objectives.aggregate(loss, weight, mode='normalized_sum')
    # else:
    return T.mean(loss)


def mean_pixel_binary_crossentropy(pred, targ, weight=None):
    """ optimization target """
    pred = pred.flatten()
    targ = targ.flatten()

    _EPSILON = 10e-8
    pred = T.clip(pred, _EPSILON, 1.0 - _EPSILON)
    pxl_loss = lasagne.objectives.binary_crossentropy(pred, targ)

    if weight is not None:
        weight = weight.flatten()
        pxl_loss *= weight

    return T.mean(pxl_loss)


def _reshape_to_softmax(tensor, Nc=1):
    # use theano for this
    batch_size = tensor.shape[0]
    tensor = tensor.reshape((batch_size, Nc, -1))
    tensor = tensor.dimshuffle((0, 2, 1))
    tensor = tensor.reshape((-1, Nc))
    return tensor
    
    
def mean_pixel_categorical_crossentropy(pred, targ, weight=None):
    """ optimization target """
    pred = _reshape_to_softmax(pred)
    targ = _reshape_to_softmax(targ)

    _EPSILON = 10e-8
    pred = T.clip(pred, _EPSILON, 1.0 - _EPSILON)
    pxl_loss = lasagne.objectives.categorical_crossentropy(pred, targ)
    
    if weight is not None:
        weight = weight.flatten()
        pxl_loss *= weight

    return T.mean(pxl_loss)


def mean_squared_error(pred, targ, weight=None):
    """ mean squared error """
    pred = pred.flatten()
    targ = targ.flatten()

    pxl_loss = lasagne.objectives.squared_error(pred, targ)

    if weight is not None:
        weight = weight.flatten()
        pxl_loss *= weight

    return T.mean(pxl_loss)
