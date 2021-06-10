#usage:  python3 training.py -c ../JobOption/NNconfig.cfg
#################################
# M. Grossi - J.Novak #2019
################################
import sys
import os

repo= os.environ['NEW_REPO']
sys.path.append(repo + '/DNN_neutrino_reco/DNN_main/TrainingHandler')

import TrainingHandler as th

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True)
args = parser.parse_args()

trainingHandler = th.TrainingHandler(args.config)
trainingHandler.gridTrain()
