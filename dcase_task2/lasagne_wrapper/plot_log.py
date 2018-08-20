
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict

import seaborn as sns
sns.set_style('ticks')
cmap = sns.color_palette()


def tsplot(data, label=None, **kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    plt.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    plt.plot(x, est, label=label, **kw)
    plt.margins(x=0)


if __name__ == '__main__':
    """
    Plot model evolution
    """

    # add argument parser
    parser = argparse.ArgumentParser(description='Show evaluation plot.')
    parser.add_argument('results', metavar='N', type=str, nargs='+', help='result.pkl files.')
    parser.add_argument('--acc', help='evaluate accuracy.', action='store_true')
    parser.add_argument('--perc', help='show percentage value.', action='store_true')
    parser.add_argument('--max_epoch', type=int, default=None, help='last epoch to plot.')
    parser.add_argument('--ymin', help='minimum y value.', type=float, default=None)
    parser.add_argument('--ymax', help='maximum y value.', type=float, default=None)
    parser.add_argument('--watch', help='refresh plot.', action='store_true')
    parser.add_argument('--key', help='key for evaluation.', type=str, default=None)
    parser.add_argument('--high_is_better', help='used for highlighting the best value.', action='store_true')
    parser.add_argument('--folds_avg', help='plot average of different folds.', action='store_true')
    args = parser.parse_args()

    if args.acc:
        args.high_is_better = True

    best_fun = np.argmax if args.high_is_better else np.argmin
    va = "bottom" if args.high_is_better else "top"

    while True:

        # load results
        all_results = OrderedDict()
        fold_results = OrderedDict()
        for result in np.sort(args.results):
            dir_name = result.split(os.sep)[-2]
            exp_name = result.split(os.sep)[-1].split('.pkl')[0]
            exp_name = '_'.join([dir_name, exp_name])
            with open(result, 'r') as fp:
                exp_res = pickle.load(fp)
                all_results[exp_name] = exp_res

                # collect results for fold averaging
                if dir_name not in fold_results:
                    fold_results[dir_name] = defaultdict(list)

                for key in exp_res.keys():
                    fold_results[dir_name][key].append(exp_res[key])

        # collect fold results
        if args.folds_avg:
            for model in fold_results.keys():
                for key in fold_results[model].keys():
                    min_samples = np.min([len(r) for r in fold_results[model][key]])
                    fold_results[model][key] = np.asarray([r[0:min_samples] for r in fold_results[model][key]])

            all_results = fold_results

        # present results
        plt.figure("Model Evolution")
        plt.clf()
        ax = plt.subplot(111)
        plt.subplots_adjust(bottom=0.15, left=0.15, right=0.9, top=0.95)

        for i, (exp_name, exp_res) in enumerate(all_results.iteritems()):

            if args.acc:

                key_tr = 'tr_accs'
                key_va = 'va_accs'

                if args.max_epoch is not None:
                    max_epoch = int(args.max_epoch)
                    exp_res['tr_accs'] = exp_res['tr_accs'][0:max_epoch]
                    exp_res['va_accs'] = exp_res['va_accs'][0:max_epoch]

                # compile label
                if args.perc:
                    acc = " (%.2f%%)" % tr_accs[-1]
                    label = exp_name + '_tr' + acc
                else:
                    label = exp_name + '_tr'

                # train accuracy
                if args.folds_avg:
                    tsplot(exp_res['tr_accs'], label=label, color=cmap[i], linewidth=2)
                else:
                    tr_accs = np.asarray(exp_res['tr_accs'], dtype=np.float)
                    tr_accs[np.equal(tr_accs, None)] = np.nan
                    indices = np.nonzero(~np.isnan(tr_accs))[0]
                    tr_accs = tr_accs[indices]

                    plt.plot(indices, tr_accs, '-', color=cmap[i], linewidth=3, alpha=0.6, label=label)

                # compile label
                if args.perc:
                    acc = " (%.2f%%)" % np.mean(va_accs[-10::])
                    label = exp_name + '_va' + acc
                else:
                    label = exp_name + '_va'

                # validation accuracy
                if args.folds_avg:
                    tsplot(exp_res['va_accs'], label=label, linewidth=1)
                else:
                    va_accs = np.asarray(exp_res['va_accs'], dtype=np.float)
                    va_accs[np.equal(va_accs, None)] = np.nan
                    indices = np.nonzero(~np.isnan(va_accs))[0]
                    va_accs = va_accs[indices]
                    plt.plot(indices, va_accs, '-', color=cmap[i], linewidth=2, label=label)
            
            else:

                if args.key is None:
                    key_tr = 'pred_tr_err'
                    key_va = 'pred_val_err'
                    label = "Loss"

                else:
                    key_tr = args.key % 'tr'
                    key_va = args.key % 'val'
                    label = args.key.replace("_%s", "")

                if args.folds_avg:
                    tsplot(exp_res[key_tr], label=exp_name + '_tr', color=cmap[i], linewidth=2)
                    tsplot(exp_res[key_va], label=exp_name + '_va', color=cmap[i], linewidth=1)
                else:
                    plt.plot(exp_res[key_tr], '-', color=cmap[i], linewidth=3, alpha=0.6, label=exp_name + '_tr')
                    plt.plot(exp_res[key_va], '-', color=cmap[i], linewidth=2, label=exp_name + '_va')

            # highlight best epoch
            if not args.folds_avg:
                best_value_idx = best_fun(exp_res[key_va])
                best_value = exp_res[key_va][best_value_idx]
            else:
                mean_vals = exp_res[key_va].mean(0)
                best_value_idx = best_fun(mean_vals)
                best_value = mean_vals[best_value_idx]
            plt.plot([0, best_value_idx], [best_value] * 2, '--', color=cmap[i], alpha=0.5)
            plt.text(best_value_idx, best_value, ('%.5f' % best_value), va=va, ha='right', color=cmap[i])
            plt.plot(best_value_idx, best_value, 'o', color=cmap[i])

        if args.acc:
            plt.ylabel("Accuracy", fontsize=20)
            plt.legend(loc="best", fontsize=18).draggable()
            plt.ylim([args.ymin, 102])
        else:
            plt.ylabel(label.upper(), fontsize=20)
            plt.legend(loc="best", fontsize=20).draggable()

        if args.ymin is not None and args.ymax is not None:
            plt.ylim([args.ymin, args.ymax])

        if args.max_epoch is not None:
            plt.xlim([0, args.max_epoch])

        plt.xlabel("Epoch", fontsize=20)
        plt.grid('on')

        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        plt.draw()

        if args.watch:
            plt.pause(10.0)
        else:
            plt.show(block=True)
            break
