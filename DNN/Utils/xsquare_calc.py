#!/usr/bin/env python3
"""

  Michele Grossi <michele.grossi@it.ibm.com>
  Jakob Novak <janob.novak.cern.ch>
  Version 1.0, June 2020
  Create a text file with output result

  USAGE: python3  xsquare_calc.py -c JobOption/NNplot_config.cfg
  """

import os
import sys
import configparser

import fnmatch
import re
import pandas as pd
import numpy as np
import argparse
from scipy.stats import chisquare
from sklearn.metrics import mean_squared_error
from operator import itemgetter

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
args = parser.parse_args()

config = configparser.ConfigParser()
config.optionxform = str
config.read(args.config)

where_save = config.get('output','output-folder')

outfile= open(where_save+ "/xsquare.txt","w+")
#this part should be implemented if the for cicle to change the folder name according to all selection list

truth = config.get('input','truth-label')
###data reading & checking
####hdf5 reading
list_rmse = {}
K = 2

for f in config.get('input','data').split(','):  
    hdf_f = pd.read_hdf(f)
    outfile.write("\n")
    outfile.write('File: ')
    outfile.write(f)
    outfile.write("\n")
    print('looping through all files:')
    print(f)
    print(config.get('selection','type'))

    if config.get('selection','type') == 'binary':
      print('BINARY')
      s_l = hdf_f[['sol0_cos_theta','sol1_cos_theta']].values
      
      for i in fnmatch.filter(hdf_f.columns, '*' + config.get('input','model_sel') + '*_rounded_score'):
      
      #for i in fnmatch.filter(hdf_f.columns, config.get('input','model_sel') + '*'):
      #selection criterion
        print(i)
        score_rnd = np.random.randint(0,2,[s_l.shape[0],])
        score_l = hdf_f[i]
        if (config.get('selection', 'invert') == '1'):
          cos_l = [s_l[i, int(not bool(sign))] for i, sign in enumerate(score_l)]
        else:
            cos_l = [s_l[i, sign] for i, sign in enumerate(score_l)]

        cos_rnd = [s_l[i, sign] for i, sign in enumerate(score_rnd)]
        rmse_rnd = mean_squared_error(hdf_f[truth],cos_rnd, squared=False)
        chi_statistic, p_value = chisquare(cos_l, hdf_f[truth])
        rmse = mean_squared_error(hdf_f[truth],cos_l, squared=False)
        outfile.write("model: ")
        outfile.write(str(i))
        outfile.write("\n")
        #outfile.write("Xsquare = {:.3f} \n".format(round(chi_statistic, 3)))
        outfile.write("rmse = {:.3f}\n".format(round(rmse, 3)))
        outfile.write("rmse random = {:.3f}\n".format(round(rmse_rnd, 3)))
        list_rmse[str(i)] = round(rmse, 3)

    elif config.get('selection','type') == 'semi_regression':
      for i in fnmatch.filter(hdf_f.columns, '*' + config.get('input','model_sel') + '*_e100'):
        print(i)
        chi_statistic, p_value = chisquare(hdf_f[i], hdf_f[truth])
        rmse = mean_squared_error(hdf_f[truth],hdf_f[i], squared=False)
        outfile.write("model: ")
        outfile.write(str(i))
        outfile.write("\n")
        #outfile.write("Xsquare = {:.3f} \n".format(round(chi_statistic, 3)))
        outfile.write("rmse = {:.3f}\n".format(round(rmse, 3)))
        list_rmse[str(i)] = round(rmse, 3)

    elif config.get('selection','type') == 'ful_regression':
      print('FULL_REGRESSION')
      #print(hdf_f.columns)
      for i in fnmatch.filter(hdf_f.columns, '*' + config.get('input','model_sel') + '*_cos'):
        
        #cat0 electron, cat 1 muon --> *_cat1_*
        #_autoai
        # 6 variables neu60hid2bat64_cos --> '*_cos'
        print(i)
        chi_statistic, p_value = chisquare(hdf_f[i], hdf_f[truth])
        rmse = mean_squared_error(hdf_f[truth],hdf_f[i], squared=False)
        outfile.write("model: ")
        outfile.write(str(i))
        outfile.write("\n")
        #outfile.write("Xsquare = {:.3f} \n".format(round(chi_statistic, 3)))
        outfile.write("rmse = {:.3f}\n".format(round(rmse, 3)))
        list_rmse[str(i)] = round(rmse, 3)
    else:
      print('wrong selection type')
    res = dict(sorted(list_rmse.items(), key = itemgetter(1))[:K]) 
    outfile.write(str(res))
# printing result 
print("The minimum K value pairs are " + str(res)) 
print('file saved in: ' + where_save )
      
    
