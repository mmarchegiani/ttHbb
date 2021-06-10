#################################
# M. Grossi - J.Novak #2019
################################
import os
import sys
import configparser
import warnings

import pandas as pd
import numpy as np
import fnmatch
import pickle

from keras.models import load_model
from sklearn.preprocessing import StandardScaler
#from sklearn.externals import joblib
from joblib import dump, load
from sklearn.metrics import auc, roc_auc_score, roc_curve

import matplotlib.pyplot as plt

repo = os.environ['NEW_REPO']
sys.path.append(repo + '/DNN_neutrino_reco/Utils/DataHandler')

import optimizeThr as ot
#import handler_kinematics as kinematics
import tables

finalconfig = True

class GridEvaluation():

	def __init__(self, config_file, selection=''):
		self.config = configparser.ConfigParser()
		self.config.optionxform = str
		self.config.read(config_file)
		
		self.trained_models = self.config.get('output','output-folder')
		training_variables = self.config.get('training', 'training-variables').split(',')
		training_labels = self.config.get('training', 'training-labels').split(',')

		print(">>> Loading datasets ...TEST ")

		self.pd_names = []
		self.pd_eval = {}
		self.data_eval = {}
		self.truth_eval = {}
		counter = 0

		for eval_samples in self.config.get('evaluation', 'data-eval').split(':'):
			pd_eval_frames = []
			for eval_sample in eval_samples.split(','):
				pd_eval_frames.append(pd.read_hdf(eval_sample))
			pd_eval = pd.concat(pd_eval_frames)
			if len(pd_eval_frames) > 1:
				if (counter == 0):
					sample_name = 'merged.calibrated.h5'
				else:
					sample_name = 'merged'+str(counter)+'.calibrated.h5'
				counter = counter + 1
			if len(pd_eval_frames) == 1:
				sample_orig = eval_samples.split('/')[-1]
				base_name = sample_orig.split('.')
				base_name.insert(1,'calibrated')
				sample_name = '.'.join(base_name)
			self.pd_names.append(sample_name)
			self.pd_eval[sample_name] = pd_eval
			self.data_eval[sample_name] = pd_eval[training_variables].values
			self.truth_eval[sample_name] = pd_eval[training_labels].values

		if self.config.get('evaluation','type') in ['binary', 'categorization']: 
			self.fig_roc = plt.figure(1)
			self.fig_roc_microav = plt.figure(2)
			self.fig_roc_macroav = plt.figure(3)

		dirs = os.listdir(self.config.get('output','output-folder'))
		self.dirs = fnmatch.filter(dirs, '*'+selection+'*')
		print(">>> Input models: " + "\n" + "\n".join(self.dirs) + "\n")

		self.evaluate_all()

		if self.config.get('evaluation','type') in ['binary', 'categorization']:
			fig_roc = plt.figure(1)
			art_r = []
			lgd_r = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2),ncol=3, fancybox=True, fontsize='small')
			art_r.append(lgd_r)
			plt.title('ROC curves')
			plt.xlabel('1 - purity')
			plt.ylabel('efficiency')
			self.fig_roc.savefig(self.config.get('evaluation', 'output') + '/roc_curves.pdf', bbox_extra_artists=art_r,bbox_inches="tight")
			# plt.figure(1)
			# plt.legend(loc='lower right', ncol=2, fancybox=True, fontsize='small')
			# #plt.xlabel('fakes')
			# plt.xlabel('1 - purity')
			# plt.ylabel('efficiency')
			# plt.title('ROC curves')
			# self.fig_roc.savefig(self.config.get('evaluation', 'output') + '/roc_curves.pdf')
			plt.figure(1)
			art_r = []
			#plt.legend(loc='upper left', ncol=2, fancybox=True, fontsize='small')
			lgd_r = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=3, fancybox=True, fontsize='small')
			art_r.append(lgd_r)
			#plt.xlabel('fakes')
			plt.xlabel('1 - purity')
			plt.ylabel('efficiency')
			plt.xlim(0.15, 0.35)
			plt.ylim(0.65, 0.85)
			plt.title('ROC curves')
			self.fig_roc.savefig(self.config.get('evaluation', 'output') + '/roc_curves_zoom.pdf', bbox_extra_artists=art_r,bbox_inches="tight")

		if self.config.get('evaluation','type') == 'categorization':
			plt.figure(2)
			plt.legend(loc='lower right', ncol=2, fancybox=True, fontsize='small')
			#plt.xlabel('fakes')
			plt.xlabel('1 - purity')
			plt.ylabel('efficiency')
			plt.title('Micro average ROC curves')
			self.fig_roc_microav.savefig(self.config.get('evaluation', 'output') + '/roc_curves_microav.pdf')
			plt.figure(3)
			plt.legend(loc='lower right', ncol=2, fancybox=True, fontsize='small')
			#plt.xlabel('fakes')
			plt.xlabel('1 - purity')
			plt.ylabel('efficiency')
			plt.title('Macro average ROC curves')
			self.fig_roc_macroav.savefig(self.config.get('evaluation', 'output') + '/roc_curves_macroav.pdf')
			plt.figure(2)
			plt.legend(loc='upper left', ncol=2, fancybox=True, fontsize='small')
			#plt.xlabel('fakes')
			plt.xlabel('1 - purity')
			plt.ylabel('efficiency')
			plt.xlim(0.15, 0.4)
			plt.ylim(0.6, 0.85)
			plt.title('Micro average ROC curves')
			self.fig_roc_microav.savefig(self.config.get('evaluation', 'output') + '/roc_curves_microav_zoom.pdf')
			plt.figure(3)
			plt.legend(loc='upper left', ncol=2, fancybox=True, fontsize='small')
			#plt.xlabel('fakes')
			plt.xlabel('1 - purity')
			plt.ylabel('efficiency')
			plt.xlim(0.15, 0.4)
			plt.ylim(0.6, 0.85)
			plt.title('Macro average ROC curves')
			self.fig_roc_macroav.savefig(self.config.get('evaluation', 'output') + '/roc_curves_macroav_zoom.pdf')

		for sample in self.pd_names:
			output_file = self.config.get('evaluation', 'output')+'/'+sample
			print(">>> Writing output "+output_file+" ...")
			#self.pd_eval[sample].to_hdf(output_file, 'evaluated_data', mode='w', table=True)
			self.pd_eval[sample].to_hdf(output_file, 'evaluated_data', mode='w', format='table')

	#########################################################################

	def roundScore(self, score, thr):
		indeces = np.argwhere(score > thr)
		for index in indeces:
			score[index] = 1
		indeces = np.argwhere(score <= thr)
		for index in indeces:
			score[index] = 0
		score = score.astype(int)

		return score

	#####################################################################

	def evaluate_all(self):
		for model_dir in self.dirs:
			models = os.listdir(self.config.get('output','output-folder') + '/' + model_dir)
			models = fnmatch.filter(models, self.config.get('evaluation','model-of-interest'))
			if len(models) == 0:
				raise ValueError('No models mathcing pattern '+self.config.get('evaluation','model-of-interest')+' found in '+model_dir)
			if len(models) > 1 and self.config.get('evaluation', 'type') == 'binary':
				warnings.warn('Only '+models[-1]+' score will be rounded')
			for model_ep in models:
				for sample in self.pd_names:
					self.evaluate( model_dir, model_ep, sample)
	
	def evaluate(self, model_dir, model_ep, sample):
		path = self.trained_models + '/' + model_dir
		
		model_name = path + '/' + model_ep
		model = load_model(model_name)

		scaler_name = path + '/scaler.pkl'
		#scaler = joblib.load(scaler_name)
		scaler = load(scaler_name)

		data_scaled = scaler.transform(self.data_eval[sample])

		pred = model.predict(data_scaled)
		
		label_sc_name = path + '/label_scaler.pkl'
		if os.path.exists(label_sc_name):
			#label_scaler = joblib.load(label_sc_name)
			label_scaler = load(label_sc_name)

			pred = label_scaler.inverse_transform(pred)

		if int(self.config.get('output','save-steps'))==1: # check this!
			epoch = model_ep[19:]
			print(">>> Evaluating model " + model_dir + " (epoch " + epoch + ") on sample " + sample.split('.')[0] + " ... ")
			if pred.shape[1]==1:
				self.pd_eval[sample][model_dir+'_e'+epoch] = pred
			else:
				for cat in range(pred.shape[1]):
					self.pd_eval[sample][model_dir+'_cat'+str(cat)+'_e'+epoch] = pred[:,cat]
			#model_label = model_dir + '_e' + epoch
			hid = model_dir.split('bat')[0].split('hid')[1]
			neu = model_dir.split('bat')[0].split('hid')[0].split('neu')[1]
			model_label = '{0} neu {1} hid. layers.'.format(neu,hid)
		else: 
			print(">>> Evaluating model " + model_dir + " on sample " + sample.split('.')[0] + " ... ")
			if pred.shape[1]==1:
				self.pd_eval[sample][model_dir+'_pred'] = pred
			else:
				for cat in range(pred.shape[1]):
					self.pd_eval[sample][model_dir+'_cat'+str(cat)+'_pred'] = pred[:,cat]
			hid = model_dir.split('bat')[0].split('hid')[1]
			neu = model_dir.split('bat')[0].split('hid')[0].split('neu')[1]
			model_label = '{0} neu {1} hid. layers.'.format(neu,hid)
			#model_label = model_dir

		if self.config.get('evaluation', 'type') == 'binary' and pred.shape[1]==1:
			plt.figure(1)
			roc_auc = roc_auc_score(self.truth_eval[sample], pred)
			print(">>> AUC: ",roc_auc)
			fp , tp, th = roc_curve(self.truth_eval[sample], pred)
			thr = ot.optimizeThr(fp,tp,th)
			plt.plot(fp, tp, label= model_label + 'AUC_'+str(round(roc_auc,2)))

			selection = self.roundScore(pred, thr)
			self.pd_eval[sample][model_dir+'_rounded_score'] = selection
			self.pd_eval[sample][model_dir+'_wp'] = thr[0]*np.ones_like(selection)
			
			nall = selection.shape[0]
			comparison = np.ones((nall,1), dtype=bool)
			np.equal(self.truth_eval[sample],selection,comparison)
			#np.equal(np.expand_dims(self.truth_eval[sample],1),selection,comparison)


			print(">>> Fraction of correct predictions: "+str(np.sum(comparison)/nall))

		if self.config.get('evaluation', 'type') == 'categorization' and pred.shape[1]>1:
			plt.figure(1)
			n_classes = pred.shape[1]
			fp = dict()
			tp = dict()
			th = dict()
			thr = dict()
			roc_auc = dict()
			pol = {0 : "LL", 1 : "LT", 2 : "TT"}
			if finalconfig == True:
				label = model_label.split('_')[0].split('bat')[0]
				#label = model_label
				print('LABEL: ',label)
				print('*'*20)

			for cat in range(n_classes):
				roc_auc[cat] = roc_auc_score(self.truth_eval[sample][:,cat], pred[:,cat])
				print(">>> AUC (class " + str(cat) + "): ",roc_auc[cat])
				fp[cat], tp[cat], th[cat] = roc_curve(self.truth_eval[sample][:,cat], pred[:,cat])
				thr[cat], var1, var2 = ot.optimizeThr(fp[cat],tp[cat],th[cat])
				#plt.plot(fp[cat], tp[cat], label=model_label + " (class " + str(cat) + ")")
				if finalconfig == True:
					plt.plot(fp[cat], tp[cat], label="class " + pol[cat])
				else:
					plt.plot(fp[cat], tp[cat], label=label)


				selection = self.roundScore(pred[:,cat], thr[cat])
				self.pd_eval[sample][model_dir+'_cat'+str(cat)+'_rounded_score'] = selection

				nall = selection.shape[0]
				comparison = np.ones((nall,1), dtype=bool)
				np.equal(np.expand_dims(self.truth_eval[sample][:,cat],1),np.expand_dims(selection,1),comparison)
				print(">>> Fraction of correct predictions: "+str(np.sum(comparison)/nall))

			fp["micro"], tp["micro"], th["micro"] = roc_curve(self.truth_eval[sample].ravel(), pred.ravel())
			roc_auc["micro"] = auc(fp["micro"], tp["micro"])

			all_fp = np.unique(np.concatenate([fp[i] for i in range(n_classes)]))
			mean_tp = np.zeros_like(all_fp)
			for i in range(n_classes):
				mean_tp += np.interp(all_fp, fp[i], tp[i])
			mean_tp /= n_classes

			fp["macro"] = all_fp
			tp["macro"] = mean_tp
			roc_auc["macro"] = auc(fp["macro"], tp["macro"])

			plt.figure(2)
			plt.plot(fp["micro"], tp["micro"], label=model_label)
			plt.figure(3)
			plt.plot(fp["macro"], tp["macro"], label=model_label)
