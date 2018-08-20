
from __future__ import print_function

import os
import sys
import time
import pickle
import itertools
import numpy as np

import theano
import lasagne
from lasagne.utils import floatX

import theano.tensor as T

from dcase_task2.lasagne_wrapper.utils import BColors, print_net_architecture
from dcase_task2.lasagne_wrapper.data_pool import DataPool
from dcase_task2.lasagne_wrapper.batch_iterators import threaded_generator_from_iterator


class Network(object):
    """
    Neural Network
    """

    def __init__(self, net, print_architecture=True):
        """
        Constructor
        """
        self.net = net
        self.compute_output = None
        self.compute_output_dict = dict()
        self.saliency_function = None
        self.iter_funcs = None

        # get input shape of network
        l_in = lasagne.layers.helper.get_all_layers(self.net)[0]
        self.input_shape = l_in.output_shape
        
        if print_architecture:
            print_net_architecture(net, detailed=True)

    def fit(self, data, training_strategy, dump_file=None, log_file=None):
        """ Train model """
        print("Training neural network...")
        col = BColors()

        # create data pool if raw data is given
        if "X_train" in data:
            data_pools = dict()
            data_pools['train'] = DataPool(data['X_train'], data['y_train'])
            data_pools['valid'] = DataPool(data['X_valid'], data['y_valid'])
        else:
            data_pools = data

        # check if out_path exists
        if dump_file is not None:
            out_path = os.path.dirname(dump_file)
            if out_path != '' and not os.path.exists(out_path):
                os.mkdir(out_path)

        # log model evolution
        if log_file is not None:
            out_path = os.path.dirname(log_file)
            if out_path != '' and not os.path.exists(out_path):
                os.mkdir(out_path)

        # adaptive learning rate
        learn_rate = training_strategy.ini_learning_rate
        learning_rate = theano.shared(floatX(learn_rate))
        learning_rate.set_value(training_strategy.adapt_learn_rate(training_strategy.ini_learning_rate, 0))

        # initialize evaluation output
        pred_tr_err, pred_val_err, overfitting = [], [], []
        tr_accs, va_accs = [], []

        print("Compiling theano train functions...")
        if self.iter_funcs is None:
            self.iter_funcs = self._create_iter_functions(y_tensor_type=training_strategy.y_tensor_type,
                                                          objective=training_strategy.objective, learning_rate=learning_rate,
                                                          l_2=training_strategy.L2,
                                                          compute_updates=training_strategy.update_parameters,
                                                          use_weights=training_strategy.use_weights,
                                                          debug_mode=training_strategy.debug_mode,
                                                          layer_update_filter=training_strategy.layer_update_filter)

        print("Starting training...")
        now = time.time()
        try:

            # initialize early stopping
            last_improvement = 0
            best_model = lasagne.layers.get_all_param_values(self.net)

            # iterate training epochs
            best_va_dice = 0.0
            prev_tr_loss, prev_va_loss = 1e7, 1e7
            prev_acc_tr, prev_acc_va = 0.0, 0.0
            prev_map_tr, prev_map_va = 0.0, 0.0
            for epoch in self._train(self.iter_funcs, data_pools, training_strategy.build_train_batch_iterator(),
                                     training_strategy.build_valid_batch_iterator(), training_strategy.report_dices,
                                     debug_mode=training_strategy.debug_mode, predict_k=training_strategy.report_map):

                print("Epoch {} of {} took {:.3f}s".format(epoch['number'], training_strategy.max_epochs, time.time() - now))
                now = time.time()

                # --- collect train output ---

                tr_loss, va_loss = epoch['train_loss'], epoch['valid_loss']
                train_acc, valid_acc = epoch['train_acc'], epoch['valid_acc']
                train_map, valid_map = epoch['train_map'], epoch['valid_map']
                train_dices, valid_dices = epoch['train_dices'], epoch['valid_dices']
                overfit = epoch['overfitting']

                # prepare early stopping
                if training_strategy.best_model_by_accurary:
                    improvement = valid_acc > prev_acc_va
                else:
                    improvement = va_loss < prev_va_loss

                if improvement:
                    last_improvement = 0
                    best_model = lasagne.layers.get_all_param_values(self.net)
                    best_epoch = epoch['number']
                    best_opt_state = [_u.get_value() for _u in self.iter_funcs['updates'].keys()]

                    # dump net parameters during training
                    if dump_file is not None:
                        with open(dump_file, 'wb') as fp:
                            pickle.dump(best_model, fp)

                last_improvement += 1

                # print train output
                txt_tr = 'costs_tr %.5f ' % tr_loss
                if tr_loss < prev_tr_loss:
                    txt_tr = col.print_colored(txt_tr, BColors.OKGREEN)
                    prev_tr_loss = tr_loss

                txt_tr_acc = '(%.3f)' % train_acc
                if train_acc > prev_acc_tr:
                    txt_tr_acc = col.print_colored(txt_tr_acc, BColors.OKGREEN)
                    prev_acc_tr = train_acc
                txt_tr += txt_tr_acc + ', '
                
                txt_val = 'costs_val %.5f ' % va_loss
                if va_loss < prev_va_loss:
                    txt_val = col.print_colored(txt_val, BColors.OKGREEN)
                    prev_va_loss = va_loss

                txt_va_acc = '(%.3f)' % valid_acc
                if valid_acc > prev_acc_va:
                    txt_va_acc = col.print_colored(txt_va_acc, BColors.OKGREEN)
                    prev_acc_va = valid_acc
                txt_val += txt_va_acc + ', '

                txt_tr_map = 'tr-map@%d %.3f' % (training_strategy.report_map, train_map)
                if train_map > prev_map_tr:
                    txt_tr_map = col.print_colored(txt_tr_map, BColors.OKGREEN)
                    prev_map_tr = train_map

                txt_va_map = 'va-map@%d %.3f' % (training_strategy.report_map, valid_map)
                if valid_map > prev_map_va:
                    txt_va_map = col.print_colored(txt_va_map, BColors.OKGREEN)
                    prev_map_va = valid_map

                txt_map = "%s, %s" % (txt_tr_map, txt_va_map)

                print('  lr: %.7f, patience: %d' % (learn_rate, training_strategy.patience - last_improvement + 1))
                print('  ' + txt_tr + txt_val + 'tr/val %.3f' % overfit)
                print('  ' + txt_map)

                # report dice coefficients
                if training_strategy.report_dices:

                    train_str = '  train  |'
                    for key in np.sort(train_dices.keys()):
                        train_str += ' %.2f: %.3f |' % (key, train_dices[key])
                    print(train_str)
                    train_acc = np.max(train_dices.values())

                    valid_str = '  valid  |'
                    for key in np.sort(valid_dices.keys()):
                        txt_va_dice = ' %.2f: %.3f |' % (key, valid_dices[key])
                        if valid_dices[key] > best_va_dice and valid_dices[key] == np.max(valid_dices.values()):
                            best_va_dice = valid_dices[key]
                            txt_va_dice = col.print_colored(txt_va_dice, BColors.OKGREEN)
                        valid_str += txt_va_dice
                    print(valid_str)
                    valid_acc = np.max(valid_dices.values())

                # report map@k
                if training_strategy.report_map:
                    pass

                # collect model evolution data
                tr_accs.append(train_acc)
                va_accs.append(valid_acc)
                pred_tr_err.append(tr_loss)
                pred_val_err.append(va_loss)
                overfitting.append(overfit)
                
                # save results
                exp_res = dict()
                exp_res['pred_tr_err'] = pred_tr_err
                exp_res['tr_accs'] = tr_accs
                exp_res['pred_val_err'] = pred_val_err
                exp_res['va_accs'] = va_accs
                exp_res['overfitting'] = overfitting
                
                if log_file is not None:
                    with open(log_file, 'w') as fp:
                        pickle.dump(exp_res, fp)                
                
                # --- early stopping: preserve best model ---
                if last_improvement > training_strategy.patience:
                    print(col.print_colored("Early Stopping!", BColors.WARNING))
                    status = "Epoch: %d, Best Validation Loss: %.5f: Acc: %.5f" % (
                    best_epoch, prev_va_loss, prev_acc_va)
                    print(col.print_colored(status, BColors.WARNING))

                    if training_strategy.refinement_strategy.n_refinement_steps <= 0:
                        break

                    else:

                        status = "Loading best parameters so far and refining (%d) with decreased learn rate ..." % \
                                 training_strategy.refinement_strategy.n_refinement_steps
                        print(col.print_colored(status, BColors.WARNING))

                        # reset net to best weights
                        lasagne.layers.set_all_param_values(self.net, best_model)

                        # reset optimizer
                        for _u, value in zip(self.iter_funcs['updates'].keys(), best_opt_state):
                            _u.set_value(value)

                        # update learn rate
                        learn_rate = training_strategy.refinement_strategy.adapt_learn_rate(learn_rate)
                        training_strategy.patience = training_strategy.refinement_strategy.refinement_patience
                        last_improvement = 0

                # maximum number of epochs reached
                if epoch['number'] >= training_strategy.max_epochs:
                    break

                # update learning rate
                learn_rate = training_strategy.adapt_learn_rate(learn_rate, epoch['number'])
                learning_rate.set_value(learn_rate)

        except KeyboardInterrupt:
            pass

        # set net to best weights
        lasagne.layers.set_all_param_values(self.net, best_model)

        # return best validation loss
        if training_strategy.best_model_by_accurary:
            return prev_acc_va
        else:
            return prev_va_loss

    def predict_proba(self, input):
        """
        Predict on test samples
        """

        # prepare input for prediction
        if not isinstance(input, list):
            input = [input]

        # reshape to network input
        if input[0].ndim < len(self.input_shape):
            input[0] = input[0].reshape([1] + list(input[0].shape))

        if self.compute_output is None:
            self.compute_output = self._compile_prediction_function()

        return self.compute_output(*input)

    def predict(self, input):
        """
        Predict class labels on test samples
        """
        return np.argmax(self.predict_proba(input), axis=1)

    def compute_layer_output(self, input, layer):
        """
        Compute output of given layer
        layer: either a string (name of layer) or a layer object
        """

        # prepare input for prediction
        if not isinstance(input, list):
            input = [input]

        # reshape to network input
        if input[0].ndim < len(self.input_shape):
            input[0] = input[0].reshape([1] + list(input[0].shape))

        # get layer by name
        if not isinstance(layer, lasagne.layers.Layer):
            for l in lasagne.layers.helper.get_all_layers(self.net):
                if l.name == layer:
                    layer = l
                    break

        # compile prediction function for target layer
        if layer not in self.compute_output_dict:
            self.compute_output_dict[layer] = self._compile_prediction_function(target_layer=layer)

        return self.compute_output_dict[layer](*input)

    def compute_saliency(self, input, nonlin=lasagne.nonlinearities.rectify):
        """
        Compute saliency maps using guided backprop
        """

        # prepare input for prediction
        if not isinstance(input, list):
            input = [input]

        # reshape to network input
        if input[0].ndim < len(self.input_shape):
            input[0] = input[0].reshape([1] + list(input[0].shape))

        if not self.saliency_function:
            self.saliency_function = self._compile_saliency_function(nonlin)

        return self.saliency_function(*input)

    def save(self, file_path):
        """
        Save model to disk
        """
        with open(file_path, 'w') as fp:
            params = lasagne.layers.get_all_param_values(self.net)
            pickle.dump(params, fp, -1)

    def load(self, file_path):
        """
        load model from disk
        """
        with open(file_path, 'r') as fp:
            params = pickle.load(fp)
        lasagne.layers.set_all_param_values(self.net, params)

    def _compile_prediction_function(self, target_layer=None):
        """
        Compile theano prediction function
        """

        # get network output nad compile function
        if target_layer is None:
            target_layer = self.net

        # collect input vars
        all_layers = lasagne.layers.helper.get_all_layers(target_layer)
        input_vars = []
        for l in all_layers:
            if isinstance(l, lasagne.layers.InputLayer):
                input_vars.append(l.input_var)

        net_output = lasagne.layers.get_output(target_layer, deterministic=True)
        return theano.function(inputs=input_vars, outputs=net_output)

    def _create_iter_functions(self, y_tensor_type, objective, learning_rate, l_2, compute_updates, use_weights,
                               debug_mode, layer_update_filter):
        """ Create functions for training, validation and testing to iterate one epoch. """

        # init target tensor
        targets = y_tensor_type('y')
        weights = y_tensor_type('w').astype("float32")

        # get input layer
        all_layers = lasagne.layers.helper.get_all_layers(self.net)

        # collect input vars
        input_vars = []
        for l in all_layers:
            if isinstance(l, lasagne.layers.InputLayer):
                input_vars.append(l.input_var)

        # compute train costs
        tr_output = lasagne.layers.get_output(self.net, deterministic=False)

        if use_weights:
            tr_cost = objective(tr_output, targets, weights)
            tr_input = input_vars + [targets, weights]
        else:
            tr_cost = objective(tr_output, targets)
            tr_input = input_vars + [targets]

        # regularization costs
        tr_reg_cost = 0

        # regularize RNNs
        for l in all_layers:

            # if l.name == "norm_reg_rnn":
            #
            #     H = lasagne.layers.get_output(l, deterministic=False)
            #     H_l2 = T.sqrt(T.sum(H ** 2, axis=-1))
            #     norm_diffs = (H_l2[:, 1:] - H_l2[:, :-1]) ** 2
            #     norm_preserving_loss = T.mean(norm_diffs)
            #
            #     beta = 1.0
            #     tr_cost += beta * norm_preserving_loss

            if l.name == "norm_reg_rnn":

                H = lasagne.layers.get_output(l, deterministic=False)
                steps = T.arange(1, l.output_shape[1])

                def compute_norm_diff(k, H):
                    n0 = ((H[:, k - 1, :]) ** 2).sum(1).sqrt()
                    n1 = ((H[:, k, :]) ** 2).sum(1).sqrt()
                    return (n1 - n0) ** 2

                norm_diffs, _ = theano.scan(fn=compute_norm_diff, outputs_info=None,
                                            non_sequences=[H], sequences=[steps])

                beta = 1.0
                norm_preserving_loss = T.mean(norm_diffs)
                tr_reg_cost += beta * norm_preserving_loss

        # compute validation costs
        va_output = lasagne.layers.get_output(self.net, deterministic=True)

        # estimate accuracy
        if y_tensor_type == T.ivector:
            va_acc = 100.0 * T.mean(T.eq(T.argmax(va_output, axis=1), targets), dtype=theano.config.floatX)
            tr_acc = 100.0 * T.mean(T.eq(T.argmax(tr_output, axis=1), targets), dtype=theano.config.floatX)

        elif y_tensor_type == T.vector:
            va_acc = 100.0 * T.mean(T.eq(T.ge(va_output.flatten(), 0.5), targets), dtype=theano.config.floatX)
            tr_acc = 100.0 * T.mean(T.eq(T.ge(tr_output.flatten(), 0.5), targets), dtype=theano.config.floatX)

        else:
            va_acc = 100.0 * T.mean(T.eq(T.argmax(va_output, axis=1), T.argmax(targets, axis=1)), dtype=theano.config.floatX)
            tr_acc = 100.0 * T.mean(T.eq(T.argmax(tr_output, axis=1), T.argmax(targets, axis=1)), dtype=theano.config.floatX)

        # collect all parameters of net and compute updates
        all_params = lasagne.layers.get_all_params(self.net, trainable=True)

        # filter parameters to update by layer name
        if layer_update_filter:
            all_params = [p for p in all_params if layer_update_filter in p.name]

        # add weight decay
        if l_2 is not None:
            all_layers = lasagne.layers.get_all_layers(self.net)
            tr_reg_cost += l_2 * lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2)

        # compute updates
        all_grads = lasagne.updates.get_or_compute_grads(tr_cost + tr_reg_cost, all_params)
        updates = compute_updates(all_grads, all_params, learning_rate)

        # compile iter functions
        tr_outputs = [tr_cost, tr_output]
        if tr_acc is not None:
            tr_outputs.append(tr_acc)
        iter_train = theano.function(tr_input, tr_outputs, updates=updates)

        va_inputs = input_vars + [targets]
        va_cost = objective(va_output, targets)
        va_outputs = [va_cost, va_output]
        if va_acc is not None:
            va_outputs.append(va_acc)
        iter_valid = theano.function(va_inputs, va_outputs)

        # network debugging
        compute_grad_norms = None
        compute_layer_outputs = None
        if debug_mode:

            # compile gradient norm computation for weights
            grad_norms = []
            for i, p in enumerate(all_params):
                if "W" in p.name:
                    g = all_grads[i]
                    grad_norm = T.sqrt(T.sum(g**2))
                    grad_norms.append(grad_norm)
            compute_grad_norms = theano.function(tr_input, grad_norms)

            # compute output of each layer
            layer_outputs = lasagne.layers.get_output(all_layers)
            compute_layer_outputs = theano.function(input_vars, layer_outputs)

        return dict(train=iter_train, valid=iter_valid, test=iter_valid, updates=updates,
                    compute_grad_norms=compute_grad_norms,
                    compute_layer_outputs=compute_layer_outputs)

    def _compile_saliency_function(self, nonlin=lasagne.nonlinearities.rectify):
        """
        Compiles a function to compute the saliency maps and predicted classes
        for a given mini batch of input images.

        in_vars = lin.input_var
        """

        class ModifiedBackprop(object):

            def __init__(self, nonlinearity):
                self.nonlinearity = nonlinearity
                self.ops = {}  # memoizes an OpFromGraph instance per tensor type

            def __call__(self, x):
                # OpFromGraph is oblique to Theano optimizations, so we need to move
                # things to GPU ourselves if needed.
                if theano.sandbox.cuda.cuda_enabled:
                    maybe_to_gpu = theano.sandbox.cuda.as_cuda_ndarray_variable
                else:
                    maybe_to_gpu = lambda x: x
                # We move the input to GPU if needed.
                x = maybe_to_gpu(x)
                # We note the tensor type of the input variable to the nonlinearity
                # (mainly dimensionality and dtype); we need to create a fitting Op.
                tensor_type = x.type
                # If we did not create a suitable Op yet, this is the time to do so.
                if tensor_type not in self.ops:
                    # For the graph, we create an input variable of the correct type:
                    inp = tensor_type()
                    # We pass it through the nonlinearity (and move to GPU if needed).
                    outp = maybe_to_gpu(self.nonlinearity(inp))
                    # Then we fix the forward expression...
                    op = theano.OpFromGraph([inp], [outp])
                    # ...and replace the gradient with our own (defined in a subclass).
                    op.grad = self.grad
                    # Finally, we memoize the new Op
                    self.ops[tensor_type] = op
                # And apply the memorized Op to the input we got.
                return self.ops[tensor_type](x)

        class GuidedBackprop(ModifiedBackprop):
            def grad(self, inputs, out_grads):
                (inp,) = inputs
                (grd,) = out_grads
                dtype = inp.dtype
                return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)

        def fix_nonlins(l_out, nonlin):
            """ Replace relus with guided-back-prop """
            nonlin_layers = [layer for layer in lasagne.layers.get_all_layers(l_out)
                             if getattr(layer, 'nonlinearity', None) is nonlin]
            modded_nonlin = GuidedBackprop(nonlin)  # important: only instantiate this once!
            for layer in nonlin_layers:
                layer.nonlinearity = modded_nonlin

            return l_out

        # fix non-linearities
        l_out = fix_nonlins(self.net, nonlin=nonlin)

        # collect input vars
        all_layers = lasagne.layers.helper.get_all_layers(l_out)
        input_vars = []
        for l in all_layers:
            if isinstance(l, lasagne.layers.InputLayer):
                input_vars.append(l.input_var)

        outp = lasagne.layers.get_output(l_out.input_layer, deterministic=True)
        max_outp = T.max(outp, axis=1)
        saliency = theano.grad(max_outp.sum(), wrt=input_vars)

        return theano.function(input_vars, saliency)

    def _train(self, iter_funcs, data_pools, train_batch_iter, valid_batch_iter, estimate_dices, debug_mode,
               predict_k=3):
        """
        Train the model with `dataset` with mini-batch training.
        Each mini-batch has `batch_size` recordings.
        """
        col = BColors()
        from dcase_task2.lasagne_wrapper.segmentation_utils import dice
        from dcase_task2.lasagne_wrapper.evaluation import mapk

        for epoch in itertools.count(1):

            # evaluate various thresholds
            if estimate_dices:
                threshs = [0.3, 0.4, 0.5, 0.6, 0.7]

                tr_dices = dict()
                for thr in threshs:
                    tr_dices[thr] = []

                va_dices = dict()
                for thr in threshs:
                    va_dices[thr] = []

            else:
                tr_dices = None
                va_dices = None

            # iterate train batches
            batch_train_losses, batch_train_accs = [], []
            batch_train_maps = []
            iterator = train_batch_iter(data_pools['train'])
            generator = threaded_generator_from_iterator(iterator)

            batch_times = np.zeros(5, dtype=np.float32)
            start, after = time.time(), time.time()
            for i_batch, train_input in enumerate(generator):
                batch_res = iter_funcs['train'](*train_input)
                batch_train_losses.append(batch_res[0])

                # collect classification accuracies
                if len(batch_res) > 2:
                    batch_train_accs.append(batch_res[2])

                    # compute map
                    y_b = train_input[1].argmax(axis=1) if train_input[1].ndim == 2 else train_input[1]
                    pred = batch_res[1]
                    actual = [[y] for y in y_b]
                    predicted = []
                    for yp in pred:
                        predicted.append(list(np.argsort(yp)[::-1][0:predict_k]))
                    batch_train_maps.append(mapk(actual, predicted, predict_k))

                # estimate dices for various thresholds
                if estimate_dices:
                    y_b = train_input[1]
                    pred = batch_res[1]
                    for thr in threshs:
                        for i in xrange(pred.shape[0]):
                            seg = pred[i, 0] > thr
                            tr_dices[thr].append(100 * dice(seg, y_b[i, 0]))

                # train time
                batch_time = time.time() - after
                after = time.time()
                train_time = (after - start)

                # estimate updates per second (running avg)
                batch_times[0:4] = batch_times[1:5]
                batch_times[4] = batch_time
                ups = 1.0 / batch_times.mean()

                # report loss during training
                perc = 100 * (float(i_batch) / train_batch_iter.n_batches)
                dec = int(perc // 4)
                progbar = "|" + dec * "#" + (25 - dec) * "-" + "|"
                vals = (perc, progbar, train_time, ups, np.mean(batch_train_losses))
                loss_str = " (%d%%) %s time: %.2fs, ups: %.2f, loss: %.5f" % vals
                print(col.print_colored(loss_str, col.WARNING), end="\r")
                sys.stdout.flush()

            # some debug plots on gradients and hidden activations
            if debug_mode:
                import matplotlib.pyplot as plt

                # compute gradient norm for last batch
                grad_norms = iter_funcs['compute_grad_norms'](*train_input)

                plt.figure("Gradient Norms")
                plt.clf()
                plt.plot(grad_norms, "g-", linewidth=3, alpha=0.7)
                plt.grid('on')
                plt.title("Gradient Norms")
                plt.ylabel("Gradient Norm")
                plt.xlabel("Weight $W_l$")
                plt.draw()

                # compute layer output for last batch
                layer_outputs = iter_funcs['compute_layer_outputs'](*train_input[:-1])

                n_outputs = len(layer_outputs)
                sub_plot_dim = np.ceil(np.sqrt(n_outputs))

                plt.figure("Hidden Activation Distributions")
                plt.clf()
                plt.subplots_adjust(bottom=0.05, top=0.98)
                for i, l_out in enumerate(layer_outputs):
                    l_out = np.asarray(l_out).flatten()
                    h, b = np.histogram(l_out, bins='auto')

                    plt.subplot(sub_plot_dim, sub_plot_dim, i + 1)
                    plt.plot(b[:-1], h, "g-", linewidth=3, alpha=0.7,
                             label="%.2f $\pm$ %.5f" % (l_out.mean(), l_out.std()))
                    span = (b[-1] - b[0])
                    x_min = b[0] - 0.05 * span
                    x_max = b[-1] + 0.05 * span
                    plt.xlim([x_min, x_max])
                    plt.legend(fontsize=10)
                    plt.grid('on')
                    plt.yticks([])
                plt.draw()

                plt.pause(0.1)

            print("\x1b[K", end="\r")
            print(' ')
            avg_train_loss = np.mean(batch_train_losses)
            if len(batch_train_accs) > 0:
                avg_train_acc = np.mean(batch_train_accs)
                avg_train_maps = np.mean(batch_train_maps)
            else:
                avg_train_acc = avg_train_maps = 0.0
            if estimate_dices:
                for thr in threshs:
                    tr_dices[thr] = np.mean(tr_dices[thr])

            # evaluate classification power of model

            # iterate validation batches
            batch_valid_losses, batch_valid_accs = [], []
            batch_valid_maps = []
            iterator = valid_batch_iter(data_pools['valid'])
            generator = threaded_generator_from_iterator(iterator)

            batch_wights = []
            for va_input in generator:
                batch_res = iter_funcs['valid'](*va_input)
                batch_valid_losses.append(batch_res[0])
                batch_wights.append(np.float(va_input[0].shape[0]))

                # collect classification accuracies
                if len(batch_res) > 2:
                    batch_valid_accs.append(batch_res[2])

                    # compute map
                    y_b = va_input[1].argmax(axis=1) if va_input[1].ndim == 2 else va_input[1]
                    pred = batch_res[1]
                    actual = [[y] for y in y_b]
                    predicted = []
                    for yp in pred:
                        predicted.append(list(np.argsort(yp)[::-1][0:predict_k]))
                    batch_valid_maps.append(mapk(actual, predicted, predict_k))

                # estimate dices for various thresholds
                if estimate_dices:
                    y_b = va_input[1]
                    pred = batch_res[1]
                    for thr in threshs:
                        for i in xrange(pred.shape[0]):
                            seg = pred[i, :] > thr
                            va_dices[thr].append(100 * dice(seg, y_b[i, :]))

                    # # todo: remove this!
                    # if np.sum(y_b[0, :-1]) > 0:
                    #     print(np.sum(y_b[0, :-1]))
                    #     import matplotlib.pyplot as plt
                    #     plt.figure("Pred", figsize=(16, 8))
                    #     plt.clf()
                    #     c = pred.shape[1]
                    #     for i in range(c):
                    #         plt.subplot(2, c, i + 1)
                    #         plt.imshow(pred[0, i], vmin=0, vmax=1)
                    #         plt.subplot(2, c, i + c + 1)
                    #         plt.imshow(y_b[0, i], vmin=0, vmax=1)
                    #     plt.savefig("epoch_%d.png" % epoch)

            batch_wights = np.asarray(batch_wights) / np.sum(batch_wights)
            batch_valid_losses = np.asarray(batch_valid_losses)
            if len(batch_valid_accs) > 0:
                batch_valid_accs = np.asarray(batch_valid_accs)
                batch_valid_maps = np.asarray(batch_valid_maps)
            else:
                batch_valid_accs = 0.0
                batch_valid_maps = 0.0

            avg_valid_loss = np.average(batch_valid_losses, weights=batch_wights)
            avg_valid_accs = np.average(batch_valid_accs, weights=batch_wights) if len(batch_valid_accs) > 0 else 0.0
            avg_valid_maps = np.average(batch_valid_maps, weights=batch_wights) if len(batch_valid_maps) > 0 else 0.0
            if estimate_dices:
                for thr in threshs:
                    va_dices[thr] = np.average(np.asarray(va_dices[thr]), weights=batch_wights)

            # collect results
            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
                'train_acc': avg_train_acc,
                'valid_loss': avg_valid_loss,
                'valid_acc': avg_valid_accs,
                'valid_dices': va_dices,
                'train_dices': tr_dices,
                'train_map': avg_train_maps,
                'valid_map': avg_valid_maps,
                'overfitting': avg_train_loss / avg_valid_loss,
            }


class SegmentationNetwork(Network):
    """
    Segmentation Neural Network
    """
    
    def predict_proba(self, input, squeeze=True, overlap=0.5):
        """
        Predict on test samples
        """
        if self.compute_output is None:
            self.compute_output = self._compile_prediction_function()
        
        # get network input shape
        l_in = lasagne.layers.helper.get_all_layers(self.net)[0]
        in_shape = l_in.output_shape[-2::]
        
        # standard prediction
        if input.shape[-2::] == in_shape:
            proba = self.compute_output(input)
        
        # sliding window prediction if images do not match
        else:
            proba = self._predict_proba_sliding_window(input, overlap=overlap)
        
        if squeeze:
            proba = proba.squeeze()
        
        return proba

    def predict(self, input, thresh=0.5):
        """
        Predict label map on test samples
        """
        P = self.predict_proba(input, squeeze=False)
        
        # binary segmentation
        if P.shape[1] == 1:
            return (P > thresh).squeeze()
        
        # categorical segmentation
        else:
            return np.argmax(P, axis=1).squeeze()
        
    
    def _predict_proba_sliding_window(self, images, overlap=0.5):
        """
        Sliding window prediction for images larger than the input layer
        """
        images = images.copy()
        n_images = images.shape[0]
        h, w = images.shape[2:4]
        _, Nc, sh, sw = self.net.output_shape

        # pad images for sliding window prediction
        missing_h = int(sh * np.ceil(float(h) / sh) - h)
        missing_w = int(sw * np.ceil(float(w) / sw) - w)

        pad_top = missing_h // 2
        pad_bottom = missing_h - pad_top

        pad_left = missing_w // 2
        pad_right = missing_w - pad_left

        images = np.pad(images, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

        step_h = int(sh * (1.0 - overlap))
        row_0 = np.arange(0, images.shape[2] - sh + 1, step_h)
        row_1 = row_0 + sh

        step_w = int(sw * (1.0 - overlap))
        col_0 = np.arange(0, images.shape[3] - sw + 1, step_w)
        col_1 = col_0 + sw

        # import pdb
        # pdb.set_trace()

        # hamming window weighting
        window_h = np.hamming(sh)
        window_w = np.hamming(sw)
        ham2d = np.sqrt(np.outer(window_h, window_w))[np.newaxis, np.newaxis]

        # initialize result image
        R = np.zeros((n_images, Nc, images.shape[2], images.shape[3]))
        V = np.zeros((n_images, Nc, images.shape[2], images.shape[3]))

        for ir in xrange(len(row_0)):
            for ic in xrange(len(col_0)):
                I = images[:, :, row_0[ir]:row_1[ir], col_0[ic]:col_1[ic]]
                P = self.compute_output(I)
                R[:, :, row_0[ir]:row_1[ir], col_0[ic]:col_1[ic]] += P * ham2d
                V[:, :, row_0[ir]:row_1[ir], col_0[ic]:col_1[ic]] += ham2d

        # clip to original image size again
        R = R[:, :, pad_top:images.shape[2] - pad_bottom, pad_left:images.shape[3] - pad_right]
        V = V[:, :, pad_top:images.shape[2] - pad_bottom, pad_left:images.shape[3] - pad_right]

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(V[0, 0])
        # plt.colorbar()
        # plt.show(block=True)

        # normalize predictions
        R /= V
        return R
