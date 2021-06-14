#usage:  python3 Evaluate.py -c ../JobOption/NNconfig.cfg -p neu
#specify pattern to take only model folders
#################################
# M. Grossi - J.Novak #2019
################################
import sys
import os
import configparser

repo= os.environ['NEW_REPO']
sys.path.append(repo + '/DNN/DNN_main/Evaluation')
#sys.path.append(repo + 'neutrinoreconstruction/DeepLearning/backward_compatibility')

#import GridEvaluation_nice_legend as ge
import GridEvaluation as ge
#import TestGridEvaluation as ge 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
parser.add_argument('-p', '--pattern', type=str, required=False, default='')
args = parser.parse_args()
config = configparser.ConfigParser()
config.optionxform = str
config.read(args.config)
print(config.get('output','output-folder'))
folder_name = config.get('evaluation','output')
if os.path.exists(folder_name):
    raise ValueError('Error: folder '+folder_name+' already exists')
os.system('mkdir ' + folder_name)
print('pattern: ' +args.pattern)
grid_evaluate = ge.GridEvaluation(args.config, args.pattern)
