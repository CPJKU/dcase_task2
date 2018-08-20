# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 08:26:40 2016

@author: matthias
"""

from __future__ import print_function

import pickle
import numpy as np
from itertools import product
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
sns.set_style('ticks')


def parallel_coordinates(data_sets, x_labels=None):
    """
    Parallel coordinates plot in python
    """
    import matplotlib.ticker as ticker

    dims = len(data_sets[0])
    x    = range(dims)
    fig, axes = plt.subplots(1, dims-1, sharey=False)
    
    # init colormap
    cmap = sns.color_palette(n_colors=len(data_sets))    
    
    # Calculate the limits on the data
    min_max_range = list()
    for m in zip(*data_sets):
        mn = min(m)
        mx = max(m)
        if mn == mx:
            mn -= 0.5
            mx = mn + 1.
        r  = float(mx - mn)
        min_max_range.append((mn, mx, r))

    # Normalize the data sets
    norm_data_sets = list()
    for ds in data_sets:
        nds = [(value - min_max_range[dimension][0]) / 
                min_max_range[dimension][2] 
                for dimension,value in enumerate(ds)]
        norm_data_sets.append(nds)
    data_sets = norm_data_sets

    # Plot the datasets on all the subplots
    for i, ax in enumerate(axes):
        for dsi, d in enumerate(data_sets):
            ax.plot(x, d, '-', color=cmap[dsi], alpha=0.7, linewidth=2.0)
        ax.set_xlim([x[i], x[i+1]])

    # Set the x axis ticks 
    for dimension, (axx,xx) in enumerate(zip(axes, x[:-1])):
        axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
        ticks = len(axx.get_yticklabels())
        labels = list()
        step = min_max_range[dimension][2] / (ticks - 1)
        mn   = min_max_range[dimension][0]
        for i in xrange(ticks):
            v = mn + i*step
            labels.append('%4.2f' % v)
        axx.set_yticklabels(labels)

    # Move the final axis' ticks to the right-hand side
    axx = plt.twinx(axes[-1])
    dimension += 1
    axx.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    ticks = len(axx.get_yticklabels())
    step = min_max_range[dimension][2] / (ticks - 1)
    mn   = min_max_range[dimension][0]
    labels = ['%4.2f' % (mn + i*step) for i in xrange(ticks)]
    axx.set_yticklabels(labels)
    
    # set x tick labels
    if x_labels is not None:
        for dimension, axx in enumerate(axes[0:-1]):
            axx.set_xticklabels([x_labels[dimension]])
        axes[-1].set_xticklabels([x_labels[-2], x_labels[-1]])
    
    # Stack the subplots 
    plt.subplots_adjust(wspace=0)


class GridSearch(object):
    """
    Class creates all combinations of grid search parameters
    """
    
    @staticmethod
    def load(pkl_file_path):
        """
        Save grid search to disc
        """
        with open(pkl_file_path, 'rb') as fp:
            gs = pickle.load(fp)
        return gs
    
    def __init__(self, small_is_better):
        """
        Constructor
        """
        self.search_space = OrderedDict()
        self.comb_idx = None
        self.small_is_better = small_is_better
    
    def add_params(self, params):
        """
        Add parameters to search space
        """

        if params.__class__ is tuple:
            params = [params]
        
        for k, vals in params:
            self.search_space[k] = vals
        
        self._init_combinations()
        
    def _init_combinations(self):
        """
        Compute possible parameter combinations
        """
        
        self.params = self.search_space.keys()
        self.n_params = len(self.params)
        self.param_list = [self.search_space[k] for k in self.params]
        self.n_comb = np.prod([len(self.search_space[k]) for k in self.params])
        self.combinations = []
        
        self.visited = np.zeros(self.n_comb, dtype=np.bool)
        self.values = np.ones(self.n_comb, dtype=np.float)
        self.all_values = self.n_comb * [np.nan]
        
        if self.small_is_better:
            self.values *= 1e9
        else:
            self.values *= -1e9
            
        
        for i_c, l in enumerate(product(*self.param_list)):
            self.combinations.append(l)
    
    def next(self):
        """
        Get next combination
        """
        not_visited = np.nonzero(self.visited == False)[0]
        for i in not_visited:
            self.comb_idx = i
            yield self._prep_params(self.combinations[i])
        
    def random(self):
        """
        Get random combination
        """
        while True:
            
            if np.alltrue(self.visited):
                break
            
            not_visited = np.nonzero(self.visited == False)[0]
            rand_idx = np.random.randint(0, len(not_visited))
            self.comb_idx = not_visited[rand_idx]
            
            yield self._prep_params(self.combinations[self.comb_idx])
    
    def set_value(self, value, values=None):
        """
        Assign score to combination
        """
        self.values[self.comb_idx] = value
        self.visited[self.comb_idx] = True
        
        if values is not None:
            self.all_values[self.comb_idx] = values
    
    def _prep_params(self, combination):
        """
        Create parameter dictionary
        """
        params = OrderedDict()
        for i, k in enumerate(self.params):
            params[k] = combination[i]
        return params
    
    def all_visited(self):
        """
        Check if all visited
        """
        return np.alltrue(self.visited)
    
    def missing_combinations(self):
        """
        Get number of missing combinations
        """
        return np.sum(self.visited == False)
    
    def get_best(self):
        """
        Get current best parameter setting
        """
        
        best_value = 1e9 if self.small_is_better else -1e9
        op = np.less if self.small_is_better else np.greater
        
        for i, v in enumerate(self.values):
            
            if not self.visited[i]:
                continue
            
            if op(self.values[i], best_value):
                best_idx = i
                best_value = self.values[i]
        
        return self.values[best_idx], self._prep_params(self.combinations[best_idx])
    
    def print_best(self):
        """
        Print current best parameter setting
        """
        best_so_far, best_comb = self.get_best()
        print("Best score so far: %.5f" % best_so_far)
        for k, v in best_comb.iteritems():
            print(" - ", k, v)

    def get_worst(self):
        """
        Get current worst parameter setting
        """
        worst_value = -1e9 if self.small_is_better else 1e9
        op = np.less if self.small_is_better else np.greater
        
        for i, v in enumerate(self.values):
            
            if not self.visited[i]:
                continue
            
            if op(worst_value, self.values[i]):
                worst_idx = i
                worst_value = self.values[i]
        
        return self.values[worst_idx], self._prep_params(self.combinations[worst_idx])
    
    def print_worst(self):
        """
        Print current worst parameter setting
        """
        worst_so_far, worst_comb = self.get_worst()
        print("Worst score so far: %.5f" % worst_so_far)
        for k, v in worst_comb.iteritems():
            print(" - ", k, v)    
    
    def get_stats(self):
        """
        Get statistics on results
        """
        values = self.values[self.visited]
        stats = dict()
        stats['mean'] = np.mean(values)
        stats['median'] = np.median(values)
        stats['std'] = np.std(values)
        stats['min'] = np.min(values)
        stats['max'] = np.max(values)
        stats['n_eval'] = len(values)
        stats['n_total'] = self.n_comb
        
        return stats
    
    def get_top_k(self, k):
        """ Get top k combinations """
        
        sorted_idx = np.argsort(self.values)
        
        if not self.small_is_better:
            sorted_idx = sorted_idx[::-1]
        
        top_k = []
        k = np.min([k, self.n_comb])
        for i in xrange(k):
            idx = sorted_idx[i]
            entry = (self.values[idx], self._prep_params(self.combinations[idx]), self.all_values[idx])
            top_k.append(entry)
        
        return top_k
    
    def visualize_grid(self):
        """ Visualize current state of grid """
        
        # prepare data
        data = []
        for i in xrange(self.n_comb):
            if self.visited[i]:            
                entry = np.asarray([self.values[i]] + list(self.combinations[i]))
                data.append(entry)
        data = np.asarray(data)
        
        # prepare data frame
        cols = ["loss"] + self.params
        parallel_coordinates(data, cols)
        plt.show()
    
    def save(self, pkl_file_path):
        """
        Save grid search to disc
        """
        with open(pkl_file_path, 'wb') as fp:
            pickle.dump(self, fp, -1)
        

if __name__ == '__main__':
    """ main """
    
    gs = GridSearch(small_is_better=True)
    gs.add_params(('N_HIDDEN', [256, 512, 1024]))
    gs.add_params(('N_LAYERS', [1, 2]))
    gs.add_params(('LEARN_RATE', [0.1, 0.01, 0.001]))
    
    for comb in gs.random():
        gs.set_value(np.random.randint(0, 100))
    
    gs.print_best()

    gs.get_top_k(5)
    
    # gs.save("/home/matthias/Desktop/tmp.pkl")
    #
    # gs = GridSearch.load("/home/matthias/Desktop/tmp.pkl")
    # gs.print_best()
    #
    # top_10 = gs.get_top_k(k=10)
    #
    # print("\nTop 10:")
    # for i in xrange(10):
    #     value, combination = top_10[i]
    #     print("\n%02d: %.5f" % (i, value))
    #
    #     for k, v in combination.iteritems():
    #         print(k, v, end="")
    #
    #     print("")
    #
    gs.visualize_grid()
