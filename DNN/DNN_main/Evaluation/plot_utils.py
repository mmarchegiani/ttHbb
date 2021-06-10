import os
import sys
import configparser

import pandas as pd
import numpy as np
import fnmatch
import pickle
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, chisquare

#create python function to create a Plot for evaluation
#we need to load data np.load(file.npz) and then plot as usual


def plot_diff_dic(self, dict,fig, pol_type = 'long'):
    plt.figure(fig)
    for key in dict:
        x = dict['truth']
        y = dict[key] 
        sc = plt.scatter(x,y,label=key)
        plt.legend()
        if pol_type == 'long':
            np.savez(self.config.get('evaluation', 'output') + '/h_' + key, scatter_long = sc)
        elif pol_type == 'trans':
            np.savez(self.config.get('evaluation', 'output') + '/h_' + key, scatter_trans = sc)
        else:
            print('wrong polarizarion')

def kolmog_smirnov(self, truth, reco):
    test_stat = ks_2samp(truth, reco)
    print('Kolmogorov-Smirnov 2-way test: {}'.format(test_stat))
    return test_stat
    
def chisqr(self, truth, reco):
    chisq, p = chisquare(reco, f_exp=truth)
    print('Chi-square test: chi-square value: {}, p-value: {}'.format(chisq, p))
    return chisq, p
