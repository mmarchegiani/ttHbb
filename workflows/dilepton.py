# python dilepton_analysis.py --sample 2017

import argparse
import os
import sys
import json
import h5py
from cycler import cycler
#from pdb import set_trace

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

from coffea import processor, hist
from coffea.lumi_tools import LumiMask, LumiData
from coffea.lookup_tools import extractor
from coffea.btag_tools import BTagScaleFactor
from coffea.util import save

from lib_analysis_v7 import lepton_selection, jet_nohiggs_selection, load_puhist_target, compute_lepton_weights
from lib_analysis_v7 import jet_selection_v7, get_dilepton_v7, get_diboson_v7, get_charged_leptons_v7, pnuCalculator_v7
from definitions_dilepton_analysis import histogram_settings, samples_info

class ttHbbDilepton(processor.ProcessorABC):
	def __init__(self, year='2017', parameters=None, samples_info=samples_info, hist_dir='histograms/', hist2d =False, DNN=False):
		#self.sample = sample
		self.year = year
		self.parameters = parameters
		self.samples_info = samples_info
		self.hist_dir = hist_dir
		self.hist2d = hist2d
		self.DNN = DNN
		self._nsolved = 0
		self._n2l2b = 0
		self._variables = histogram_settings['variables']
		self._variables2d = histogram_settings['variables2d']
		self._varnames = self._variables.keys()
		self._hist2dnames = self._variables2d.keys()
		self._fill_opts = histogram_settings['fill_opts']
		self._error_opts = histogram_settings['error_opts']
		self._mask_events = {
		  #'trigger'	 : None,
		  #'resolved'	 : None,
		  'basic'       : None,
		  'boost'       : None,
		  '2l1b'        : None,
		  '2l1bHbb'     : None,
		  '2l2b'        : None,
		  '2l2bsolved'  : None,
		  '2l2bnotsolved': None,
		  '2l2blowdr'  	: None,
		  '2l2bHbb'     : None,
		  '2l2bmw'      : None,
		  '2l2bHbbmw'   : None,
		  '2l2bmwmt'    : None,
		  '2l2bHbbmwmt' : None,
		  '2l2blowmt'   : None,
		  '2l2bhighmt'  : None,
		  #'joint		 : None'
		}
		self._weights_list = [
		  'ones',
		  'nominal'
		]
		self._vars_to_plot = {
		'muons_pt'					: None,
		'muons_eta'					: None,
		'goodmuons_pt'				: None,
		'goodmuons_eta'				: None,
		'electrons_pt'				: None,
		'electrons_eta'				: None,
		'goodelectrons_pt'			: None,
		'goodelectrons_eta'			: None,
		'jets_pt'					: None,
		'jets_eta'					: None,
		'goodjets_pt'				: None,
		'goodjets_eta'				: None,
		'nleps'             		: None,
		'njets'             		: None,
		'ngoodjets'         		: None,
		'btags'             		: None,
		'btags_resolved'    		: None,
		'nfatjets'          		: None,
		'charge_sum'				: None,
		'met_pt'               		: None,
		'met_phi'              		: None,
		'mll'						: None,
		'leading_jet_pt'    		: None,
		'leading_jet_eta'   		: None,
		'leading_jet_phi'   		: None,
		'leading_jet_mass'   		: None,
		'leading_bjet_pt'    		: None,
		'leading_bjet_eta'   		: None,
		'leading_bjet_phi'   		: None,
		'leading_bjet_mass'   		: None,
		'leadAK8JetMass'    		: None,
		'leadAK8JetPt'      		: None,
		'leadAK8JetEta'     		: None,
		'leadAK8JetPhi'     		: None,
		'leadAK8JetRho'     		: None,
		'leadAK8JetHbb'				: None,
		'leadAK8JetTau21'			: None,
		'lepton_plus_pt'            : None,
		'lepton_plus_eta'           : None,
		'lepton_plus_phi'           : None,
		'lepton_plus_mass'          : None,
		'lepton_minus_pt'           : None,
		'lepton_minus_eta'          : None,
		'lepton_minus_phi'          : None,
		'lepton_minus_mass'         : None,
		'leading_lepton_pt'         : None,
		'leading_lepton_eta'        : None,
		'leading_lepton_phi'        : None,
		'leading_lepton_mass'       : None,
		'ptll'                      : None,
		'mt_ww'                     : None,
		'pnu_x'						: None,
		'pnu_y'						: None,
		'pnu_z'						: None,
		'pnubar_x'					: None,
		'pnubar_y'					: None,
		'pnubar_z'					: None,
		'm_w_plus'					: None,
		'm_w_minus'					: None,
		'm_top'						: None,
		'm_topbar'					: None,
		'm_tt'						: None,
		'tt_pt'						: None,
		'top_pt'					: None,
		'topbar_pt'					: None,
		'nNuGen'					: None,
		'nNubarGen'					: None,
		'ratioNuPtGenReco'			: None,
		'ratioNubarPtGenReco'		: None,
		'deltaRNuNuGen'				: None,
		'deltaRNubarNubarGen'		: None,
		'deltaRLeptonPlusHiggs'		: None,
		'deltaRLeptonMinusHiggs'	: None,
		'deltaRBBbar'				: None,
		'deltaRHiggsTop'			: None,
		'deltaRHiggsTopbar'			: None,
		'deltaRHiggsTT'				: None,
		'deltaRTopTopbar'			: None,
		'deltaPhiBBbar'				: None,
		'deltaPhiHiggsTop'			: None,
		'deltaPhiHiggsTopbar'		: None,
		'deltaPhiHiggsTT'			: None,
		'deltaPhiTopTopbar'			: None,
		'ttHbb_label'				: None,
		'weights_nominal'			: None,
		}
		#self._vars_to_plot = processor.dict_accumulator({
		#	'leading_lepton_pt'         : None,
		#	})

		self._accumulator = processor.dict_accumulator({
			"sumw": processor.defaultdict_accumulator(float),
			"nevts": processor.defaultdict_accumulator(int),
			"nevts_solved": processor.defaultdict_accumulator(int),
			#"leading_lepton_pt": processor.column_accumulator(ak.Array([])),
		})

		for var in self._vars_to_plot.keys():
			self._accumulator.add(processor.dict_accumulator({var : processor.column_accumulator(np.array([]))}))

		for wn in self._weights_list:
			for mask_name in self._mask_events.keys():
				#self._accumulator.add(processor.dict_accumulator({f'sumw_SR_{mask_name}_weights_{wn}' : processor.defaultdict_accumulator(float),})
				for var_name in self._varnames:
					self._accumulator.add(processor.dict_accumulator({f'hist_{var_name}_{mask_name}_weights_{wn}' : hist.Hist("$N_{events}$",
																	  hist.Cat("dataset", "Dataset"),
																	  hist.Bin("values", self._variables[var_name]['xlabel'], np.linspace( *self._variables[var_name]['binning'] ) ) ) } ))
				for hist2d_name in self._hist2dnames:
					varname_x = list(self._variables2d[hist2d_name].keys())[0]
					varname_y = list(self._variables2d[hist2d_name].keys())[1]
					self._accumulator.add(processor.dict_accumulator({f'hist2d_{hist2d_name}_{mask_name}_weights_{wn}' : hist.Hist("$N_{events}$",
																	  hist.Cat("dataset", "Dataset"),
																	  hist.Bin("x", self._variables2d[hist2d_name][varname_x]['xlabel'], np.linspace( *self._variables2d[hist2d_name][varname_x]['binning'] ) ),
																	  hist.Bin("y", self._variables2d[hist2d_name][varname_y]['ylabel'], np.linspace( *self._variables2d[hist2d_name][varname_y]['binning'] ) )
																	   ) } ))

			"""
		vars_split = ['leadAK8JetMass', 'leadAK8JetRho']
		ptbins = np.append( np.arange(250,600,50), [600, 1000, 5000] )
		for var_name in vars_split:
			#var = vars_to_plot[var_name]
			for ipt in range( len(ptbins)-1 ):
				for m in ['2J2WdeltaR']:#, '2J2WdeltaRTau21']:#, '2J2WdeltaRTau21DDT']:
					for r in ['Pass','Fail']:
						for o in ['','_orthogonal']:
							mask_name = f'{m}_{r}{o}'
							if not mask_name in self._mask_events_list: continue
							hist_name = f'hist_{var_name}_{mask_name}_pt{ptbins[ipt]}to{ptbins[ipt+1]}'
							self._accumulator.add(processor.dict_accumulator({hist_name : hist.Hist("entries",
																			  hist.Cat("dataset", "Dataset"),
																			  hist.Bin("values", var_name, np.linspace( *self._variables[var_name] ) ) ) } ) )

		#for wn,w in weights.items():
		for wn in ['nominal']:
			if wn != 'nominal': continue
			#ret[f'nevts_overlap{weight_name}'] = Histogram( [sum(weights[w]), sum(weights[w][mask_events['2J2WdeltaR']]), sum(weights[w][mask_events['resolved']]), sum(weights[w][mask_events['overlap']])], 0,0 )
			for mask_name in self._mask_events_list:
				if not 'deltaR' in mask_name: continue
				for var_name in self._varnames:
					#if (not is_mc) and ('Pass' in mask_name) and (var_name=='leadAK8JetMass') : continue
					try:
						hist_name = f'hist_{var_name}_{mask_name}_weights_{wn}'
						self._accumulator.add(processor.dict_accumulator({hist_name : hist.Hist("entries",
																		  hist.Cat("dataset", "Dataset"),
																		  hist.Bin("values", var_name, np.linspace( *self._variables[var_name if not var_name.startswith('weights') else 'weights'] ) ) ) } ) )
					except KeyError:
						print(f'!!!!!!!!!!!!!!!!!!!!!!!! Please add variable {var_name} to the histogram settings')
			"""

	@property
	def accumulator(self):
		return self._accumulator

	def process(self, events):
		output = self.accumulator.identity()
		dataset = events.metadata["dataset"]
		#nEvents = events.event.size
		nEvents = ak.count(events.event)
		output['nevts'][dataset] += nEvents
		is_mc = 'genWeight' in events.fields
		if is_mc:
			output['sumw'][dataset] += sum(events.genWeight)

		muons = events.Muon
		electrons = events.Electron
		#scalars = events.eventvars
		jets = events.Jet
		fatjets = events.FatJet
		PuppiMET = events.PuppiMET
		PV = events.PV
		Flag = events.Flag
		run = events.run
		luminosityBlock = events.luminosityBlock
		HLT = events.HLT
		genWeight = events.genWeight
		#puWeight = events.puWeight
		if is_mc:
			genparts = events.GenPart

		if self.year =='2017':
			metstruct = 'METFixEE2017'
			MET = events.METFixEE2017
		else:
			metstruct = 'MET'
			MET = events.MET

		#print("MET choice: %s" % metstruct)

		mask_events = np.ones(nEvents, dtype=np.bool)

		# apply event cleaning and  PV selection
		flags = [
			"goodVertices", "globalSuperTightHalo2016Filter", "HBHENoiseFilter", "HBHENoiseIsoFilter", "EcalDeadCellTriggerPrimitiveFilter", "BadPFMuonFilter"]#, "BadChargedCandidateFilter", "ecalBadCalibFilter"]
		if not is_mc:
			flags.append("eeBadScFilter")
		for flag in flags:
			mask_events = mask_events & getattr(Flag, flag)
		mask_events = mask_events & (PV.npvsGood > 0)

		#in case of data: check if event is in golden lumi file
		if not is_mc and not (lumimask is None):
			mask_lumi = lumimask(run, luminosityBlock)
			mask_events = mask_events & mask_lumi

		# apply object selection for muons, electrons, jets
		good_muons, veto_muons = lepton_selection(muons, self.parameters["muons"], self.year)
		good_electrons, veto_electrons = lepton_selection(electrons, self.parameters["electrons"], self.year)
		good_jets = jet_selection_v7(jets, muons, (good_muons|veto_muons), self.parameters["jets"]) & jet_selection_v7(jets, electrons, (good_electrons|veto_electrons), self.parameters["jets"])
		bjets_resolved = good_jets & (getattr(jets, self.parameters["btagging_algorithm"]) > self.parameters["btagging_WP"])
		good_fatjets = jet_selection_v7(fatjets, muons, good_muons, self.parameters["fatjets"]) & jet_selection_v7(fatjets, electrons, good_electrons, self.parameters["fatjets"])

		#leading_fatjets = ak.pad_none(fatjets[good_fatjets], 1)[:,0]
		#print(leading_fatjets)
		pfake = {'pt' : [-9999.9], 'eta' : [-9999.9], 'phi' : [-9999.9], 'mass' : [-9999.9]}
		fakeFatJet = ak.zip(pfake, with_name="PtEtaPhiMCandidate")
		#print("fatjets[good_fatjets]", fatjets[good_fatjets])
		#leading_fatjets = ak.fill_none(ak.pad_none(fatjets[good_fatjets], 1), fakeFatJet)[:,0]
		leading_fatjets = ak.fill_none(ak.pad_none(fatjets[good_fatjets], 1), fakeFatJet, axis=None)[:,0]

		good_jets_nohiggs = ( good_jets & (jets.delta_r(leading_fatjets) > 1.2) )
		bjets = good_jets_nohiggs & (getattr(jets, self.parameters["btagging_algorithm"]) > self.parameters["btagging_WP"])
		nonbjets = good_jets_nohiggs & (getattr(jets, self.parameters["btagging_algorithm"]) < self.parameters["btagging_WP"])

		# apply basic event selection -> individual categories cut later
		ngoodmuons     = ak.num(muons[good_muons])
		ngoodelectrons = ak.num(electrons[good_electrons])
		nmuons         = ak.num(muons[good_muons | veto_muons])
		nelectrons     = ak.num(electrons[good_electrons | veto_electrons])
		ngoodleps      = ngoodmuons + ngoodelectrons
		nleps          = nmuons + nelectrons
		#lepton_veto    = muons[veto_muons].counts[mask_events] + electrons[veto_electrons].counts[mask_events]
		njets          = ak.num(jets[nonbjets])
		ngoodjets      = ak.num(jets[good_jets])
		btags          = ak.num(jets[bjets])
		btags_resolved = ak.num(jets[bjets_resolved])
		nfatjets       = ak.num(fatjets[good_fatjets])
		#nhiggs         = fatjets[higgs_candidates].counts[mask_events]

		# trigger logic
		trigger_ee = (nleps==2) & (nelectrons==2)
		trigger_emu = (nleps==2) & (nelectrons==1) & (nmuons==1)
		trigger_mumu = (nleps==2) & (nmuons==2)
		#print("nleps", len(nleps))
		for trigger in self.parameters["triggers"]["ee"]:	trigger_ee   = trigger_ee   & HLT[trigger.lstrip("HLT_")]
		for trigger in self.parameters["triggers"]["emu"]:	trigger_emu  = trigger_emu  & HLT[trigger.lstrip("HLT_")]
		for trigger in self.parameters["triggers"]["mumu"]:	trigger_mumu = trigger_mumu & HLT[trigger.lstrip("HLT_")]
		
		#if "DoubleMuon" in self.sample: trigger_ee = np.zeros(nEvents, dtype=np.bool)
		#if "DoubleElectron" in self.sample: trigger_emu = np.zeros(nEvents, dtype=np.bool)
		mask_events = mask_events & (trigger_ee | trigger_emu | trigger_mumu)
		self._mask_events["trigger"] = mask_events

		# select good objects
		
		for key, n in zip(['ngoodmuons', 'ngoodelectrons', 'nmuons', 'nelectrons', 'ngoodleps', 'nleps',
							   'njets', 'ngoodjets', 'btags', 'btags_resolved', 'nfatjets'],
							  [ngoodmuons, ngoodelectrons, nmuons, nelectrons, ngoodleps, nleps,
							   njets, ngoodjets, btags, btags_resolved, nfatjets]):
			events[key] = n
		#events["GoodMuon"]    = muons[good_muons]
		#events["GoodElectron"]= electrons[good_electrons]
		events["GoodMuon"]     = muons[good_muons | veto_muons]
		events["GoodElectron"] = electrons[good_electrons | veto_electrons]
		events["GoodJet"]      = jets[nonbjets]
		events["GoodBJet"]     = jets[bjets]
		events["GoodFatJet"]   = fatjets[good_fatjets]
		selev = events[mask_events]
		ll = get_dilepton_v7(selev.GoodElectron, selev.GoodMuon)
		ww = get_diboson_v7(ll, selev.MET)
		ww_t = get_diboson_v7(ll, selev.MET, transverse=True)
		SFOS = ( ((selev.nmuons == 2) & (selev.nelectrons == 0)) | ((selev.nmuons == 0) & (selev.nelectrons == 2)) ) & (ll.charge == 0)
		not_SFOS = ( (selev.nmuons == 1) & (selev.nelectrons == 1) ) & (ll.charge == 0)

		# for reference, this is the selection for the resolved analysis
		mask_events_res   = ((selev.nleps == 2) & (selev.ngoodleps >= 1) & (ll.charge == 0) &
							(selev.ngoodjets >= 2) & (selev.btags_resolved > 1) & (selev.MET.pt > 40) &
							(ll.mass > 20) & ((SFOS & ((ll.mass < 76) | (ll.mass > 106))) | not_SFOS) )
		# apply basic event selection
		mask_events_basic = ((selev.nleps == 2) & (selev.ngoodleps >= 1) & (ll.charge == 0) &
							(selev.MET.pt > self.parameters['met']) & (selev.nfatjets > 0) & (selev.btags >= self.parameters['btags']) &
							(ll.mass > 20) & ((SFOS & ((ll.mass < 76) | (ll.mass > 106))) | not_SFOS) ) # & (selev.btags_resolved < 3)# & (selev.njets > 1)  # & np.invert( (selev.njets >= 4)  )
		mask_events_OS    = (mask_events_res | mask_events_basic)

		#print("fatjets[good_fatjets]", good_fatjets )
		#print("mask_events_basic", (selev.nfatjets > 0) )
		#print("mask_events_basic", ((selev.MET.pt > self.parameters['met']) & (selev.nfatjets > 0) & (selev.btags >= self.parameters['btags'])) )

		# calculate basic variables
		leading_jet_pt         = ak.pad_none(selev.GoodJet, 1)[:,0].pt
		leading_jet_eta        = ak.pad_none(selev.GoodJet, 1)[:,0].eta
		leading_jet_phi        = ak.pad_none(selev.GoodJet, 1)[:,0].phi
		leading_jet_mass       = ak.pad_none(selev.GoodJet, 1)[:,0].mass
		leading_bjet_pt        = ak.pad_none(selev.GoodBJet, 1)[:,0].pt
		leading_bjet_eta       = ak.pad_none(selev.GoodBJet, 1)[:,0].eta
		leading_bjet_phi       = ak.pad_none(selev.GoodBJet, 1)[:,0].phi
		leading_bjet_mass      = ak.pad_none(selev.GoodBJet, 1)[:,0].mass
		leading_fatjet_SDmass  = ak.pad_none(selev.GoodFatJet, 1)[:,0].msoftdrop
		leading_fatjet_pt      = ak.pad_none(selev.GoodFatJet, 1)[:,0].pt
		leading_fatjet_eta     = ak.pad_none(selev.GoodFatJet, 1)[:,0].eta
		leading_fatjet_phi     = ak.pad_none(selev.GoodFatJet, 1)[:,0].phi
		leading_fatjet_mass    = ak.pad_none(selev.GoodFatJet, 1)[:,0].mass
		leading_fatjet_rho     = ak.Array( np.log(leading_fatjet_SDmass**2 / leading_fatjet_pt**2) )
		leading_fatjet_Hbb     = ak.pad_none(selev.GoodFatJet, 1)[:,0][self.parameters['bbtagging_algorithm']]
		leading_fatjet_tau1    = ak.pad_none(selev.GoodFatJet, 1)[:,0].tau1
		leading_fatjet_tau2    = ak.pad_none(selev.GoodFatJet, 1)[:,0].tau2
		leading_fatjet_tau21   = ak.Array( leading_fatjet_tau2/leading_fatjet_tau1 )
		lepton_plus 		   = get_charged_leptons_v7(selev.GoodElectron, selev.GoodMuon, +1, SFOS | not_SFOS)
		lepton_minus 		   = get_charged_leptons_v7(selev.GoodElectron, selev.GoodMuon, -1, SFOS | not_SFOS)

		antilepton_is_leading  = (lepton_plus.pt > lepton_minus.pt)
		leading_lepton_pt      = ak.where(antilepton_is_leading, lepton_plus.pt, lepton_minus.pt)
		leading_lepton_eta     = ak.where(antilepton_is_leading, lepton_plus.eta, lepton_minus.eta)
		leading_lepton_phi     = ak.where(antilepton_is_leading, lepton_plus.phi, lepton_minus.phi)
		leading_lepton_mass    = ak.where(antilepton_is_leading, lepton_plus.mass, lepton_minus.mass)

		mask_events_boost	   = mask_events_basic & (leading_fatjet_Hbb > self.parameters['bbtagging_WP'])
		mask_events_2l2b	   = mask_events_basic & (selev.btags >= 2)
		mask_events_2l2bHbb    = mask_events_boost & (selev.btags >= 2)
		mask_events_2l1b	   = mask_events_basic & (selev.btags >= 1)
		mask_events_2l1bHbb    = mask_events_boost & (selev.btags >= 1)

		lepton_plus_2l2b 	   = get_charged_leptons_v7(selev.GoodElectron, selev.GoodMuon, +1, mask_events_2l2b)
		lepton_minus_2l2b	   = get_charged_leptons_v7(selev.GoodElectron, selev.GoodMuon, -1, mask_events_2l2b)
		pnu, pnubar, pb, pbbar, mask_events_2l2bsolved = pnuCalculator_v7(lepton_minus_2l2b, lepton_plus_2l2b, selev.GoodBJet, selev.MET, ak.pad_none(selev.GoodFatJet, 1)[:,0])
		mask_events_2l2bnotsolved = np.invert(mask_events_2l2bsolved) & mask_events_2l2b

		nEvents_solved = ak.count(events.event[mask_events_2l2bsolved])
		output['nevts_solved'][dataset] += nEvents_solved

		print("lepton_minus_2l2b:", len(lepton_minus_2l2b))
		print("pnubar:", len(pnubar))
		pwm = lepton_minus_2l2b + pnubar
		pwp = lepton_plus_2l2b + pnu
		ptop = pwp + pb
		ptopbar = pwm + pbbar
		ptt = ptop + ptopbar

		NeutrinoIDs = [12,14,16]
		#AntiNeutrinoIDs = [-id for id in NeutrinoIDs]
		neutrinos = ak.zeros_like(selev.LHEPart.pdgId == 12, dtype=bool)
		antineutrinos = ak.zeros_like(selev.LHEPart.pdgId == 12, dtype=bool)
		for id in NeutrinoIDs:
			neutrinos = neutrinos | (selev.LHEPart.pdgId == id)
			antineutrinos = antineutrinos | (selev.LHEPart.pdgId == -id)
		pnuGen = selev.LHEPart[neutrinos]
		pnubarGen = selev.LHEPart[antineutrinos]

		nNuGen 				   = ak.count(pnuGen.pt[pnuGen.pt>0], axis=1)
		nNubarGen			   = ak.count(pnubarGen.pt[pnubarGen.pt>0], axis=1)

		#for i in range(len(pnuGen.pt)):
		#	if len(pnuGen.pt[i]) == 0:
		#		print(selev.LHEPart.pdgId[i])

		# Since not all events at generator level contain a neutrino and an anti-neutrino, we pad pnuGen.pt with a nan negative value
		ratioNuPtGenReco	   = ak.fill_none(ak.pad_none(pnuGen.pt, 1), -999.9)[:,0]/pnu.pt
		ratioNubarPtGenReco	   = ak.fill_none(ak.pad_none(pnubarGen.pt, 1), -999.9)[:,0]/pnubar.pt

		ratioNuPtRecoGen	   = pnu.pt/ak.fill_none(ak.pad_none(pnuGen.pt, 1), -999.9)[:,0]
		ratioNubarPtRecoGen	   = pnubar.pt/ak.fill_none(ak.pad_none(pnubarGen.pt, 1), -999.9)[:,0]

		deltaRNuNuGen		   = pnu.delta_r(ak.pad_none(pnuGen, 1)[:,0])
		deltaRNubarNubarGen	   = pnubar.delta_r(ak.pad_none(pnubarGen, 1)[:,0])

		print("deltaRNuNuGen", deltaRNuNuGen)
		
		#mask_events_withGoodFatJet = selev.GoodFatJet.counts > 0
		#higgs 				   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_withGoodFatJet, dtype=int), pt=leading_fatjet_pt[mask_events_withGoodFatJet], eta=leading_fatjet_eta[mask_events_withGoodFatJet], phi=leading_fatjet_phi[mask_events_withGoodFatJet], mass=leading_fatjet_mass[mask_events_withGoodFatJet])
		#tops				   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=ptop['pt'][mask_events_2l2b], eta=ptop['eta'][mask_events_2l2b], phi=ptop['phi'][mask_events_2l2b], mass=ptop['mass'][mask_events_2l2b])
		#topbars				   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=ptopbar['pt'][mask_events_2l2b], eta=ptopbar['eta'][mask_events_2l2b], phi=ptopbar['phi'][mask_events_2l2b], mass=ptopbar['mass'][mask_events_2l2b])
		#tts					   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=ptt['pt'][mask_events_2l2b], eta=ptt['eta'][mask_events_2l2b], phi=ptt['phi'][mask_events_2l2b], mass=ptt['mass'][mask_events_2l2b])

		deltaRLeptonPlusHiggs  = lepton_plus.delta_r(ak.pad_none(selev.GoodFatJet, 1)[:,0])
		deltaRLeptonMinusHiggs = lepton_plus.delta_r(ak.pad_none(selev.GoodFatJet, 1)[:,0])

		deltaRBBbar			   = pb.delta_r(pbbar)
		deltaRHiggsTop		   = ptop.delta_r(ak.pad_none(selev.GoodFatJet, 1)[:,0])
		deltaRHiggsTopbar	   = ptopbar.delta_r(ak.pad_none(selev.GoodFatJet, 1)[:,0])
		deltaRHiggsTT		   = ptt.delta_r(ak.pad_none(selev.GoodFatJet, 1)[:,0])
		deltaRTopTopbar		   = ptop.delta_r(ptopbar)
		deltaPhiBBbar		   = pb.delta_phi(pbbar)
		deltaPhiHiggsTop	   = ptop.delta_phi(ak.pad_none(selev.GoodFatJet, 1)[:,0])
		deltaPhiHiggsTopbar	   = ptopbar.delta_phi(ak.pad_none(selev.GoodFatJet, 1)[:,0])
		deltaPhiHiggsTT		   = ptt.delta_phi(ak.pad_none(selev.GoodFatJet, 1)[:,0])
		deltaPhiTopTopbar	   = ptop.delta_phi(ptopbar)

		print("deltaRHiggsTop", deltaRHiggsTop)

		if dataset == 'ttHTobb':
			signal_label = ak.ones_like(lepton_plus.pt)
		else:
			signal_label = ak.zeros_like(lepton_plus.pt)

		mask_events_2l2blowdr	= mask_events_2l2b & (deltaRBBbar < 0.2)
		mask_events_2l2blowmt   = mask_events_2l2b & (ptop.mass < 200)
		mask_events_2l2bhighmt  = mask_events_2l2b & (ptop.mass > 200)
		mask_events_2l2bmw      = mask_events_2l2b & (pwp.mass < 200) & (pwm.mass < 200)
		mask_events_2l2bHbbmw   = mask_events_2l2bHbb & (pwp.mass < 200) & (pwm.mass < 200)
		mask_events_2l2bmwmt    = mask_events_2l2bmw & (ptop.mass < 200) & (ptopbar.mass < 200)
		mask_events_2l2bHbbmwmt = mask_events_2l2bHbbmw & (ptop.mass < 200) & (ptopbar.mass < 200)

		"""
		#good_events           = events[mask_events]
		mask_events_withGoodFatJet = selev.GoodFatJet.counts > 0
		mask_events_withLepton = nleps > 0
		leading_leptons = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_withLepton, dtype=int), pt=leading_lepton_pt[mask_events_withLepton], eta=leading_lepton_eta[mask_events_withLepton], phi=leading_lepton_phi[mask_events_withLepton], mass=leading_lepton_mass[mask_events_withLepton])
		higgs = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_withGoodFatJet, dtype=int), pt=leading_fatjet_pt[mask_events_withGoodFatJet], eta=leading_fatjet_eta[mask_events_withGoodFatJet], phi=leading_fatjet_phi[mask_events_withGoodFatJet], mass=leading_fatjet_mass[mask_events_withGoodFatJet])
		deltaRHiggsLepton      = calc_dr(leading_leptons, higgs)
		selev.LeadingFatJet["deltaRHiggsLepton"] = deltaRHiggsLepton

		self._nsolved		   += np.array(pnu['x'] > -1000).sum()
		self._n2l2b			   += np.array(mask_events_2l2b).sum()		
		"""
		# calculate weights for MC samples
		weights = {}
		weights["ones"] = np.ones(nEvents, dtype=np.float32)
		weights["nominal"] = np.ones(nEvents, dtype=np.float32)

		if is_mc:
			weights["nominal"] = weights["nominal"] * genWeight * self.parameters["lumi"] * samples_info[dataset]["XS"] / output["sumw"][dataset]
			#weights["nominal"] = weights["nominal"] * genWeight * self.parameters["lumi"] * samples_info[dataset]["XS"] / samples_info[dataset]["ngen_weight"][self.year]

			# pu corrections
			#if puWeight is not None:
			#	weights['pu'] = puWeight
				#if not uncertaintyName.startswith('puWeight'):
				#	weights['pu'] = puWeight
				#else:
				#	weights['pu'] = uncertaintyName
			#else:
		#        weights['pu'] = compute_pu_weights(self.parameters["pu_corrections_target"], weights["nominal"], scalars["Pileup_nTrueInt"], scalars["PV_npvsGood"])
				#weights['pu'] = compute_pu_weights(self.parameters["pu_corrections_target"], weights["nominal"], scalars["Pileup_nTrueInt"], scalars["Pileup_nTrueInt"])
			#weights["nominal"] = weights["nominal"] * weights['pu']

			# lepton SF corrections
			#electron_weights = compute_lepton_weights(events.GoodElectron, evaluator, ["el_triggerSF", "el_recoSF", "el_idSF"], lepton_eta=(events.GoodElectron.deltaEtaSC + events.GoodElectron.eta))
			#muon_weights = compute_lepton_weights(events.GoodMuon, evaluator, ["mu_triggerSF", "mu_isoSF", "mu_idSF"], year=self.year)
			#weights['lepton']  = muon_weights * electron_weights
			#weights["nominal"] = weights["nominal"] * weights['lepton']

		self._mask_events = {
		  #'trigger'  : mask_events_trigger,
		  #'resolved' : mask_events_res,
		  'basic'    	: mask_events_basic,
		  'boost'    	: mask_events_boost,
		  '2l1b'     	: mask_events_2l1b,
		  '2l1bHbb'    	: mask_events_2l1bHbb,
		  '2l2b'     	: mask_events_2l2b,
		  '2l2bsolved'  : mask_events_2l2bsolved,
		  '2l2bnotsolved': mask_events_2l2bnotsolved,
		  '2l2blowdr'  	: mask_events_2l2blowdr,
		  '2l2bHbb'     : mask_events_2l2bHbb,
		  '2l2bmw'      : mask_events_2l2bmw,
		  '2l2bHbbmw'   : mask_events_2l2bHbbmw,
		  '2l2bmwmt'    : mask_events_2l2bmwmt,
		  '2l2bHbbmwmt' : mask_events_2l2bHbbmwmt,
		  '2l2blowmt'   : mask_events_2l2blowmt,
		  '2l2bhighmt'  : mask_events_2l2bhighmt,
		  #'joint'    : mask_events_OS
		}

		############# histograms
		vars_to_plot = {
		'muons_pt'					: muons.pt,
		'muons_eta'					: muons.eta,
		'goodmuons_pt'				: events.GoodMuon.pt,
		'goodmuons_eta'				: events.GoodMuon.eta,
		'electrons_pt'				: electrons.pt,
		'electrons_eta'				: electrons.eta,
		'goodelectrons_pt'			: events.GoodElectron.pt,
		'goodelectrons_eta'			: events.GoodElectron.eta,
		'jets_pt'					: jets.pt,
		'jets_eta'					: jets.eta,
		'goodjets_pt'				: events.GoodJet.pt,
		'goodjets_eta'				: events.GoodJet.eta,
		'nleps'             		: selev.nleps,
		'njets'             		: selev.njets,
		'ngoodjets'         		: selev.ngoodjets,
		'btags'             		: selev.btags,
		'btags_resolved'    		: selev.btags_resolved,
		'nfatjets'          		: selev.nfatjets,
		'charge_sum'				: ll.charge,
		'met_pt'               		: MET.pt,
		'met_phi'              		: MET.phi,
		'mll'						: ll.mass,
		'leading_jet_pt'    		: leading_jet_pt,
		'leading_jet_eta'   		: leading_jet_eta,
		'leading_jet_phi'   		: leading_jet_phi,
		'leading_jet_mass'   		: leading_jet_mass,
		'leading_bjet_pt'    		: leading_bjet_pt,
		'leading_bjet_eta'   		: leading_bjet_eta,
		'leading_bjet_phi'   		: leading_bjet_phi,
		'leading_bjet_mass'   		: leading_bjet_mass,
		'leadAK8JetMass'    		: leading_fatjet_SDmass,
		'leadAK8JetPt'      		: leading_fatjet_pt,
		'leadAK8JetEta'     		: leading_fatjet_eta,
		'leadAK8JetPhi'     		: leading_fatjet_phi,
		'leadAK8JetRho'     		: leading_fatjet_rho,
		'leadAK8JetHbb'				: leading_fatjet_Hbb,
		'leadAK8JetTau21'			: leading_fatjet_tau21,
		'lepton_plus_pt'            : lepton_plus.pt,
		'lepton_plus_eta'           : lepton_plus.eta,
		'lepton_plus_phi'           : lepton_plus.phi,
		'lepton_plus_mass'          : lepton_plus.mass,
		'lepton_minus_pt'           : lepton_minus.pt,
		'lepton_minus_eta'          : lepton_minus.eta,
		'lepton_minus_phi'          : lepton_minus.phi,
		'lepton_minus_mass'         : lepton_minus.mass,
		'leading_lepton_pt'         : leading_lepton_pt,
		'leading_lepton_eta'        : leading_lepton_eta,
		'leading_lepton_phi'        : leading_lepton_phi,
		'leading_lepton_mass'       : leading_lepton_mass,
		'ptll'                      : ll.pt,
		'mt_ww'                     : ww_t.mass,
		'pnu_x'						: pnu.x,
		'pnu_y'						: pnu.y,
		'pnu_z'						: pnu.z,
		'pnubar_x'					: pnubar.x,
		'pnubar_y'					: pnubar.y,
		'pnubar_z'					: pnubar.z,
		'm_w_plus'					: pwp.mass,
		'm_w_minus'					: pwm.mass,
		'm_top'						: ptop.mass,
		'm_topbar'					: ptopbar.mass,
		'm_tt'						: ptt.mass,
		'tt_pt'						: ptt.pt,
		'top_pt'					: ptop.pt,
		'topbar_pt'					: ptopbar.pt,
		'nNuGen'					: nNuGen,
		'nNubarGen'					: nNubarGen,
		'ratioNuPtGenReco'			: ratioNuPtGenReco,
		'ratioNubarPtGenReco'		: ratioNubarPtGenReco,
		'deltaRNuNuGen'				: deltaRNuNuGen,
		'deltaRNubarNubarGen'		: deltaRNubarNubarGen,
		'deltaRLeptonPlusHiggs'		: deltaRLeptonPlusHiggs,
		'deltaRLeptonMinusHiggs'	: deltaRLeptonMinusHiggs,
		'deltaRBBbar'				: deltaRBBbar,
		'deltaRHiggsTop'			: deltaRHiggsTop,
		'deltaRHiggsTopbar'			: deltaRHiggsTopbar,
		'deltaRHiggsTT'				: deltaRHiggsTT,
		'deltaRTopTopbar'			: deltaRTopTopbar,
		'deltaPhiBBbar'				: deltaPhiBBbar,
		'deltaPhiHiggsTop'			: deltaPhiHiggsTop,
		'deltaPhiHiggsTopbar'		: deltaPhiHiggsTopbar,
		'deltaPhiHiggsTT'			: deltaPhiHiggsTT,
		'deltaPhiTopTopbar'			: deltaPhiTopTopbar,
		'ttHbb_label'				: signal_label,
		'weights_nominal'			: weights["nominal"],
		}
		self._vars_to_plot = vars_to_plot.copy()
		for var in self._vars_to_plot.keys():
			if var.split("_")[0] in ["muons", "goodmuons", "electrons", "goodelectrons", "jets", "goodjets"]:
				continue
			#print(self._vars_to_plot[var])
			mask = self._mask_events['basic']

			# 0-padding for the AK4 jet variables and angular variables
			if ('leading_jet' in var) | ('deltaR' in var) | ('nNu' in var) | ('ratioNu' in var):
				column = ak.fill_none(self._vars_to_plot[var][mask], 0.)
				column = ak.where(column != float('inf'), column, ak.zeros_like(column))
				output[var] = output[var] + processor.column_accumulator(ak.to_numpy(column))
			else:
				output[var] = output[var] + processor.column_accumulator(ak.to_numpy(self._vars_to_plot[var][mask]))
		#output['leading_lepton_pt'] = output['leading_lepton_pt'] + processor.column_accumulator(ak.to_numpy(leading_lepton_pt))

		vars2d_to_plot = {
			'm_top_vs_pnu_x' : {
				'pnu_x' : abs(pnu.x),
				'm_top' : ptop.mass,
			},
			'm_top_vs_met' : {
				'met'   : MET.pt,
				'm_top' : ptop.mass,
			},
			'm_top_vs_leading_lepton_pt' : {
				'leading_lepton_pt' : leading_lepton_pt,
				'm_top'             : ptop.mass,
			},
			'm_top_vs_leadAK8JetHbb' : {
				'leadAK8JetHbb' : leading_fatjet_Hbb,
				'm_top'         : ptop.mass,
			},
			'm_top_vs_btags' : {
				'btags' : btags,
				'm_top' : ptop.mass,
			},
			#'m_top_vs_leading_bjet_pt' : {
			#	'leading_bjet_pt' : leading_bjet_pt,
			#	'm_top'			  : ptop.mass,
			#},
			#'m_top_vs_leading_bjet_eta' : {
			#	'leading_bjet_eta' : abs(leading_bjet_eta),
			#	'm_top'			   : ptop.mass,
			#},
			'm_top_vs_m_w_plus' : {
				'm_w_plus' 		   : pwp.mass,
				'm_top'			   : ptop.mass,
			},
			'm_top_vs_m_w_minus' : {
				'm_w_minus' 	   : pwm.mass,
				'm_top'			   : ptop.mass,
			},
			'm_topbar_vs_m_w_plus' : {
				'm_w_plus' 		   : pwp.mass,
				'm_topbar'		   : ptopbar.mass,
			},
			'm_topbar_vs_m_w_minus' : {
				'm_w_minus'		   : pwm.mass,
				'm_topbar'		   : ptopbar.mass,
			},
			'deltaRBBbar_vs_nleps' : {
				'nleps'			   : nleps,
				'deltaRBBbar'	   : deltaRBBbar,
			},
			'deltaRBBbar_vs_met' : {
				'met'			   : MET.pt,
				'deltaRBBbar'	   : deltaRBBbar,
			},
			'deltaRBBbar_vs_leadAK8JetPt' : {
				'leadAK8JetPt'	   : leading_fatjet_pt,
				'deltaRBBbar'	   : deltaRBBbar,
			},
			'm_w_minus_vs_m_w_plus' : {
				'm_w_plus' 	   	   : pwp.mass,
				'm_w_minus'		   : pwm.mass,
			},
			'm_topbar_vs_m_top' : {
				'm_top' 	   	   : ptop.mass,
				'm_topbar'		   : ptopbar.mass,
			},
		}

		def flatten(ar): # flatten awkward into a 1d array to hist
			return ak.flatten(ak.fill_none(ar, -999.9), axis=None)

		for wn,w in weights.items():
			if not wn in ['ones', 'nominal']: continue
			for mask_name, mask in self._mask_events.items():
				for var_name, var in vars_to_plot.items():
					try:
						if var_name.split("_")[0] in ["muons", "goodmuons", "electrons", "goodelectrons", "jets", "goodjets"]:
							continue
						else:
							if wn == 'ones':
								output[f'hist_{var_name}_{mask_name}_weights_{wn}'].fill(dataset=dataset, values=flatten(var[mask]))
							else:
								output[f'hist_{var_name}_{mask_name}_weights_{wn}'].fill(dataset=dataset, values=flatten(var[mask]), weight=flatten(w[mask]))
					except KeyError:
						print(f'!!!!!!!!!!!!!!!!!!!!!!!! Please add variable {var_name} to the histogram settings ({mask_name})')
				for hist2d_name, vars2d in vars2d_to_plot.items():
					#try:
					varname_x = list(vars2d.keys())[0]
					varname_y = list(vars2d.keys())[1]
					var_x = vars2d[varname_x]
					var_y = vars2d[varname_y]
					if wn == 'ones':
						output[f'hist2d_{hist2d_name}_{mask_name}_weights_{wn}'].fill(dataset=dataset, x=flatten(var_x[mask]), y=flatten(var_y[mask]))
					else:
						output[f'hist2d_{hist2d_name}_{mask_name}_weights_{wn}'].fill(dataset=dataset, x=flatten(var_x[mask]), y=flatten(var_y[mask]), weight=flatten(w[mask]))
					#except KeyError:
					#	print(f'!!!!!!!!!!!!!!!!!!!!!!!! Please add variables {hist2d_name} to the histogram settings ({mask_name})')
		return output

	def postprocess(self, accumulator):

		#print(accumulator['leading_lepton_pt'])
		if self.DNN:
			filepath = self.DNN
			hdf_dir = '/'.join(filepath.split('/')[:-1])
			if not os.path.exists(hdf_dir):
				os.makedirs(hdf_dir)
			# Minimal input set, only leading jet, lepton and fatjet observables
			# No top-related variables used
			input_vars = ['ngoodjets', 'njets', 'btags', 'nfatjets', 'met_pt', 'met_phi',
						  'leading_jet_pt', 'leading_jet_eta', 'leading_jet_phi', 'leading_jet_mass',
						  'leadAK8JetPt', 'leadAK8JetEta', 'leadAK8JetPhi', 'leadAK8JetMass', 'leadAK8JetHbb', 'leadAK8JetTau21',
						  'lepton_plus_pt', 'lepton_plus_eta', 'lepton_plus_phi', 'lepton_plus_mass',
						  'lepton_minus_pt', 'lepton_minus_eta', 'lepton_minus_phi', 'lepton_minus_mass',
						  'deltaRHiggsTop', 'deltaRHiggsTopbar', 'deltaRHiggsTT', 'deltaRLeptonPlusHiggs', 'deltaRLeptonMinusHiggs',
						  'ttHbb_label',
						  'weights_nominal']
			for key, value in accumulator.items():
				if key in input_vars:
					print(key, value.value)
			vars_to_DNN = {key : ak.Array(value.value) for key, value in accumulator.items() if key in input_vars}
			print("vars_to_DNN = ", vars_to_DNN)
			inputs = ak.zip(vars_to_DNN)
			print("inputs = ", inputs)
			print(f"Saving DNN inputs to {filepath}")
			df = ak.to_pandas(inputs)
			df.to_hdf(filepath, key='df', mode='w')
			"""
			h5f = h5py.File(filepath, 'w')
			for k in inputs.fields:
				print("Create", k)
				h5f.create_dataset(k, data=inputs[k])
			h5f.close()
			"""

		return accumulator
