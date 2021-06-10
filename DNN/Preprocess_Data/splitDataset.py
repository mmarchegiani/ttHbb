#################################
# M. Marchegiani #2021
################################
#macro to create 3 preprocessed data (h5 format) from h5 file
#USAGE: python3 create_preprocessData.py -in '/rootfiles/gen81_mu_ewk_semilept_lsf_lhe.root'
#  -o '/rootfiles' -n 'gen81_mu_ewk_semilept_lsf_lhe' -ch 0 -s ‘0.3:0.3:0.4’
# TRAINING: unpolarized, transverse, longitudinal, mixed(transverse+longitudinal)
#evaluation: polarized(transverse and longitudinal), unpolarized (to then perform fit and deduce fraction of polarization)

import sys
import numpy as np
import pandas as pd
import os
import argparse
import tables

#this part can be improved avoiding too many line command
parser = argparse.ArgumentParser()
parser.add_argument('-in', '--input', type=str, required=False)
parser.add_argument('-o', '--output-folder', type=str, default='output_folder' , required=False)#output folder
#parser.add_argument('-n', '--name', type=str, required=True)#basename of the file
parser.add_argument('-s', '--separation', type=str, required=False, default='1')# ex. '0.2:0.4:0.4' separation of dataset between training, test, evaluation 
parser.add_argument('-nev', '--nevents', type=int, required=False, default=-1)
args = parser.parse_args()

df = pd.read_hdf(args.input)

n_events = args.nevents

if n_events > df.shape[0]:
    raise ValueError('wrong number of events selected, the maximum number available for training is:' + str(data_handler.pdarray.shape[0]))
elif n_events == -1 :
    n_events = df.shape[0]

#this is to ensure that the signal and background events are split equally in the train, val and eval samples
df = df.drop(np.where(np.isnan(df['leadAK8JetHbb'].values))[0])
n_events = df.shape[0]
df = df.sample(n_events)

if args.output_folder == parser.get_default('output_folder'):
    filename = args.input.split('/')[-1]
    args.output_folder = args.input.split('/' + filename)[0]

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

#create training and evaluation dataset from original file
fractions = [float(i) for i in args.separation.split(':')]
if np.sum(fractions) != 1:
    print('ERROR: split fraction must sum to 1, like 0.2:0.4:0.4 ')
    sys.exit(1)

fractions = [round(x * n_events) for x in fractions]
event_difference = n_events - np.sum(fractions)
fractions[0] = fractions[0] + event_difference #correction to have right number of events

curr_ind = 0
indeces = [curr_ind]
for fraction in fractions:
    curr_ind += fraction
    indeces.append(curr_ind) 

for i in range(len(indeces)-1):
    data = df[indeces[i]:indeces[i+1]]
    #name = args.output_folder + '/' + args.name + str(fractions[i]) +'_' +str(i)
    name = args.output_folder + '/' + args.input.rstrip('.h5').split('/')[-1] + '_' + str(fractions[i])
    if i == 0:
        name = name +'_train'
    elif i == 1:
        name = name +'_val'
    elif i == 2:
        name = name + '_eval'
    #add hf5
    #print(data.columns)

    #data.to_hdf(name+'.h5',name,mode='w',table=True)#to check

    data.to_hdf(name+'.h5',name,mode='w',format ='table')
    h5file = tables.open_file(name+'.h5',driver="H5FD_CORE")#this save data on disk after closure of python
    h5file.close()
     #in training I will need to open, load and work on data but not save it on disk:
    #h5file = tables.open_file("sample.h5", "a", driver="H5FD_CORE",driver_core_backing_store=0)
    
   
    print(name+'.h5')
