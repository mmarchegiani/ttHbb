# python dilepton_analysis.py --sample 2017

import argparse
import os
import sys
import json
from cycler import cycler
#from pdb import set_trace

import awkward1
import matplotlib.pyplot as plt
import numpy as np

from coffea import processor, hist
from coffea.analysis_objects import JaggedCandidateArray
from coffea.lumi_tools import LumiMask, LumiData
from coffea.lookup_tools import extractor
from coffea.btag_tools import BTagScaleFactor
from coffea.util import save
from uproot_methods import TLorentzVectorArray

from lib_analysis import lepton_selection, jet_selection, jet_nohiggs_selection, get_charge_sum, get_dilepton_vars, get_transverse_mass, get_charged_var, get_leading_value, load_puhist_target, compute_lepton_weights, calc_dr, calc_dphi, pnuCalculator, obj_reco
from definitions_dilepton_analysis import parameters, histogram_settings, samples_info

class ttHbb(processor.ProcessorABC):
	def __init__(self):
		#self.sample = sample
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

		self._accumulator = processor.dict_accumulator({
			"sumw": processor.defaultdict_accumulator(float),
			"nevts": processor.defaultdict_accumulator(int),
			"nevts_solved": processor.defaultdict_accumulator(int),
		})

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

	def process(self, events, parameters=parameters, samples_info=samples_info):
		output = self.accumulator.identity()
		dataset = events.metadata["dataset"]
		#nEvents = events.event.size
		nEvents = awkward1.count(events.event)
		output['nevts'][dataset] += nEvents
		is_mc = 'genWeight' in events.columns
		#is_mc = 'genWeight' in events.fields
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

		if args.year =='2017':
			metstruct = 'METFixEE2017'
			MET = events.METFixEE2017
		else:
			metstruct = 'MET'
			MET = events.MET

		#print("MET choice: %s" % metstruct)

		muons.p4 = JaggedCandidateArray.candidatesfromcounts(muons.counts, pt=muons.pt.content, eta=muons.eta.content, phi=muons.phi.content, mass=muons.mass.content)
		electrons.p4 = JaggedCandidateArray.candidatesfromcounts(electrons.counts, pt=electrons.pt.content, eta=electrons.eta.content, phi=electrons.phi.content, mass=electrons.mass.content)
		jets.p4 = JaggedCandidateArray.candidatesfromcounts(jets.counts, pt=jets.pt.content, eta=jets.eta.content, phi=jets.phi.content, mass=jets.mass.content)
		fatjets.p4 = JaggedCandidateArray.candidatesfromcounts(fatjets.counts, pt=fatjets.pt.content, eta=fatjets.eta.content, phi=fatjets.phi.content, mass=fatjets.mass.content)

		"""
		for obj in [muons, electrons, jets, fatjets, PuppiMET, MET]:
			obj.masks = {}
			obj.masks['all'] = np.ones_like(obj.flatten(), dtype=np.bool)
		"""

		mask_events = np.ones(nEvents, dtype=np.bool)
		#print("mask_events", len(mask_events))

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
		good_muons, veto_muons = lepton_selection(muons, parameters["muons"], args.year)
		good_electrons, veto_electrons = lepton_selection(electrons, parameters["electrons"], args.year)
		good_jets = jet_selection(jets, muons, (good_muons|veto_muons), parameters["jets"]) & jet_selection(jets, electrons, (good_electrons|veto_electrons), parameters["jets"])
		bjets_resolved = good_jets & (getattr(jets, parameters["btagging_algorithm"]) > parameters["btagging_WP"])
		good_fatjets = jet_selection(fatjets, muons, good_muons, parameters["fatjets"]) & jet_selection(fatjets, electrons, good_electrons, parameters["fatjets"])

		mask_events_withFatJet = fatjets[good_fatjets].counts > 0
		leading_fatjets = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_withFatJet, dtype=int), pt=get_leading_value(fatjets[good_fatjets].pt[mask_events_withFatJet]), eta=get_leading_value(fatjets[good_fatjets].eta[mask_events_withFatJet]), phi=get_leading_value(fatjets[good_fatjets].phi[mask_events_withFatJet]), mass=get_leading_value(fatjets[good_fatjets].mass[mask_events_withFatJet]))

		good_jets_nohiggs = good_jets & jet_nohiggs_selection(jets.p4, good_jets, leading_fatjets, 1.2)
		bjets = good_jets_nohiggs & (getattr(jets, parameters["btagging_algorithm"]) > parameters["btagging_WP"])
		nonbjets = good_jets_nohiggs & (getattr(jets, parameters["btagging_algorithm"]) < parameters["btagging_WP"])

		# apply basic event selection -> individual categories cut later
		#nmuons         = muons[good_muons].counts[mask_events]
		#nelectrons     = electrons[good_electrons].counts[mask_events]
		ngoodmuons     = muons[good_muons].counts
		ngoodelectrons = electrons[good_electrons].counts
		nmuons         = muons[good_muons | veto_muons].counts
		nelectrons     = electrons[good_electrons | veto_electrons].counts
		ngoodleps      = ngoodmuons + ngoodelectrons
		nleps          = nmuons + nelectrons
		#lepton_veto    = muons[veto_muons].counts[mask_events] + electrons[veto_electrons].counts[mask_events]
		njets          = jets[nonbjets].counts
		ngoodjets      = jets[good_jets].counts
		btags          = jets[bjets].counts
		btags_resolved = jets[bjets_resolved].counts
		nfatjets       = fatjets[good_fatjets].counts
		#nhiggs         = fatjets[higgs_candidates].counts[mask_events]

		# trigger logic
		trigger_el = (nleps==2) & (nelectrons==2)
		trigger_el_mu = (nleps==2) & (nelectrons==1) & (nmuons==1)
		trigger_mu = (nleps==2) & (nmuons==2)
		#print("nleps", len(nleps))
		if args.year.startswith('2016'):
			trigger_el = trigger_el & (HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ | HLT.Ele27_WPTight_Gsf)
			trigger_el_mu = trigger_el_mu & (HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL | HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ |
											 HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL | HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ |
											 HLT.Ele27_WPTight_Gsf | HLT.IsoMu24 | HLT.IsoTkMu24)
			trigger_mu = trigger_mu & (HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL | HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ |
									   HLT.Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL | HLT.Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ)
		elif args.year.startswith('2017'):
			#if args.sample.endswith(('2017B','2017C')):
			#	trigger_tmp = HLT.Ele32_WPTight_Gsf_L1DoubleEG & any([getattr(HLT, 'L1_SingleEG{n}er2p5') for n in (10,15,26,34,36,38,40,42,45,8)])
			#else:
			#	trigger_tmp = HLT.Ele32_WPTight_Gsf
			#trigger_el = trigger_el & (trigger_tmp | HLT.Ele28_eta2p1_WPTight_Gsf_HT150)
			#print("trigger_el", len(trigger_el))
			#print("Ele23_Ele12_CaloIdL_TrackIdL_IsoVL", len(HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL))
			#print("Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ", len(HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ))
			#print("Ele32_WPTight_Gsf", len(HLT.Ele32_WPTight_Gsf))
			trigger_el = trigger_el & (HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL | HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ | HLT.Ele32_WPTight_Gsf)
			trigger_el_mu = trigger_el_mu & (HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL | HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ |
											 HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ | HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ |
											 HLT.Ele32_WPTight_Gsf | HLT.IsoMu24_eta2p1 | HLT.IsoMu27)
			trigger_mu = trigger_mu & (HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ | HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 |
									   HLT.IsoMu24_eta2p1 | HLT.IsoMu27)
		elif args.year.startswith('2018'):
			trigger_el = trigger_el & (HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL | HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ | HLT.Ele32_WPTight_Gsf)
			trigger_el_mu = trigger_el_mu & (HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL | HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ |
											 HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ | HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ |
											 HLT.Ele32_WPTight_Gsf | HLT.IsoMu24)
			trigger_mu = trigger_mu & (HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8 | HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 | HLT.IsoMu24)
		#if "DoubleMuon" in args.sample: trigger_el = np.zeros(nEvents, dtype=np.bool)
		#if "DoubleElectron" in args.sample: trigger_mu = np.zeros(nEvents, dtype=np.bool)
		mask_events = mask_events & (trigger_el | trigger_el_mu | trigger_mu)
		mask_events_trigger = mask_events

		# select good objects
		#events["GoodMuon"]    = muons[good_muons]
		#events["GoodElectron"]= electrons[good_electrons]
		events["GoodMuon"]     = muons[good_muons | veto_muons]
		events["GoodElectron"] = electrons[good_electrons | veto_electrons]
		events["GoodJet"]      = jets[nonbjets]
		events["GoodBJet"]     = jets[bjets]
		events["GoodFatJet"]   = fatjets[good_fatjets]
		charge_sum = get_charge_sum(events.GoodElectron, events.GoodMuon)
		goodmuons = JaggedCandidateArray.candidatesfromcounts(events.GoodMuon.counts, pt=events.GoodMuon.pt.flatten(), eta=events.GoodMuon.eta.flatten(), phi=events.GoodMuon.phi.flatten(), mass=events.GoodMuon.mass.flatten())
		goodelectrons = JaggedCandidateArray.candidatesfromcounts(events.GoodElectron.counts, pt=events.GoodElectron.pt.flatten(), eta=events.GoodElectron.eta.flatten(), phi=events.GoodElectron.phi.flatten(), mass=events.GoodElectron.mass.flatten())
		ptll, etall, phill, mll = get_dilepton_vars(goodelectrons, goodmuons)
		mask_events_2l = (ptll >= 0)
		dileptons = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l, dtype=int), pt=ptll[mask_events_2l], eta=etall[mask_events_2l], phi=phill[mask_events_2l], mass=mll[mask_events_2l])
		METs = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l, dtype=int), pt=MET.pt[mask_events_2l], eta=np.zeros_like(MET.pt[mask_events_2l]), phi=MET.phi[mask_events_2l], mass=np.zeros_like(MET.pt[mask_events_2l]))
		mt_ww = get_transverse_mass(dileptons, METs)
		#lepWW = dileptons.p4 + METs.p4
		SFOS = ( ((nmuons == 2) & (nelectrons == 0)) | ((nmuons == 0) & (nelectrons == 2)) ) & (charge_sum == 0)
		not_SFOS = ( (nmuons == 1) & (nelectrons == 1) ) & (charge_sum == 0)

		# for reference, this is the selection for the resolved analysis
		mask_events_res   = (mask_events & (nleps == 2) & (ngoodleps >= 1) & (charge_sum == 0) &
							(ngoodjets >= 2) & (btags_resolved > 1) & (MET.pt > 40) &
							(mll > 20) & ((SFOS & ((mll < 76) | (mll > 106))) | not_SFOS) )
		# apply basic event selection
		mask_events_basic = (mask_events & (nleps == 2) & (ngoodleps >= 1) & (charge_sum == 0) &
							(MET.pt > parameters['met']) & (nfatjets > 0) & (btags >= parameters['btags']) &
							(mll > 20) & ((SFOS & ((mll < 76) | (mll > 106))) | not_SFOS) ) # & (btags_resolved < 3)# & (njets > 1)  # & np.invert( (njets >= 4)  )
		mask_events_OS    = (mask_events_res | mask_events_basic)

		# calculate basic variables
		leading_jet_pt         = get_leading_value(events.GoodJet.pt)
		leading_jet_eta        = get_leading_value(events.GoodJet.eta, default=-9.)
		leading_jet_phi        = get_leading_value(events.GoodJet.phi)
		leading_bjet_pt        = get_leading_value(events.GoodBJet.pt)
		leading_bjet_eta       = get_leading_value(events.GoodBJet.eta, default=-9.)
		leading_bjet_phi       = get_leading_value(events.GoodBJet.phi)
		leading_fatjet_SDmass  = get_leading_value(events.GoodFatJet.msoftdrop)
		leading_fatjet_pt      = get_leading_value(events.GoodFatJet.pt)
		leading_fatjet_eta     = get_leading_value(events.GoodFatJet.eta, default=-9.)
		leading_fatjet_phi     = get_leading_value(events.GoodFatJet.phi)
		leading_fatjet_mass    = get_leading_value(events.GoodFatJet.mass)
		leading_fatjet_rho     = awkward1.from_iter( np.log(leading_fatjet_SDmass**2 / leading_fatjet_pt**2) )
		leading_fatjet_Hbb     = get_leading_value(events.GoodFatJet[parameters['bbtagging_algorithm']])
		leading_fatjet_tau1    = get_leading_value(events.GoodFatJet.tau1)
		leading_fatjet_tau2    = get_leading_value(events.GoodFatJet.tau2)
		leading_fatjet_tau21   = awkward1.from_iter( leading_fatjet_tau2/leading_fatjet_tau1 )
		lepton_plus_pt         = get_charged_var("pt", events.GoodElectron, events.GoodMuon, +1, SFOS | not_SFOS)
		lepton_plus_eta        = get_charged_var("eta", events.GoodElectron, events.GoodMuon, +1, SFOS | not_SFOS, default=-9.)
		lepton_plus_phi        = get_charged_var("phi", events.GoodElectron, events.GoodMuon, +1, SFOS | not_SFOS)
		lepton_plus_mass       = get_charged_var("mass", events.GoodElectron, events.GoodMuon, +1, SFOS | not_SFOS)
		lepton_minus_pt        = get_charged_var("pt", events.GoodElectron, events.GoodMuon, -1, SFOS | not_SFOS)
		lepton_minus_eta       = get_charged_var("eta", events.GoodElectron, events.GoodMuon, -1, SFOS | not_SFOS, default=-9.)
		lepton_minus_phi       = get_charged_var("phi", events.GoodElectron, events.GoodMuon, -1, SFOS | not_SFOS)
		lepton_minus_mass      = get_charged_var("mass", events.GoodElectron, events.GoodMuon, -1, SFOS | not_SFOS)
		antilepton_is_leading  = (lepton_plus_pt > lepton_minus_pt)
		leading_lepton_pt      = awkward1.where(antilepton_is_leading, lepton_plus_pt, lepton_minus_pt)
		leading_lepton_eta     = awkward1.where(antilepton_is_leading, lepton_plus_eta, lepton_minus_eta)
		leading_lepton_phi     = awkward1.where(antilepton_is_leading, lepton_plus_phi, lepton_minus_phi)
		leading_lepton_mass    = awkward1.where(antilepton_is_leading, lepton_plus_mass, lepton_minus_mass)

		mask_events_boost	   = mask_events_basic & (leading_fatjet_Hbb > parameters['bbtagging_WP'])
		mask_events_2l2b	   = mask_events_basic & (btags >= 2)
		mask_events_2l2bHbb    = mask_events_boost & (btags >= 2)
		mask_events_2l1b	   = mask_events_basic & (btags >= 1)
		mask_events_2l1bHbb    = mask_events_boost & (btags >= 1)

		leptons_plus		   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=lepton_plus_pt[mask_events_2l2b], eta=lepton_plus_eta[mask_events_2l2b], phi=lepton_plus_phi[mask_events_2l2b], mass=lepton_plus_mass[mask_events_2l2b])
		leptons_minus		   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=lepton_minus_pt[mask_events_2l2b], eta=lepton_minus_eta[mask_events_2l2b], phi=lepton_minus_phi[mask_events_2l2b], mass=lepton_minus_mass[mask_events_2l2b])
		goodbjets			   = JaggedCandidateArray.candidatesfromcounts(events.GoodBJet.counts, pt=events.GoodBJet.pt.flatten(), eta=events.GoodBJet.eta.flatten(), phi=events.GoodBJet.phi.flatten(), mass=events.GoodBJet.mass.flatten())
		METs_2b			   	   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=MET.pt[mask_events_2l2b], eta=np.zeros_like(MET.pt[mask_events_2l2b]), phi=MET.phi[mask_events_2l2b], mass=np.zeros_like(MET.pt[mask_events_2l2b]))
		pnu, pnubar, pb, pbbar, mask_events_2l2bsolved = pnuCalculator(leptons_minus, leptons_plus, goodbjets, METs_2b)
		mask_events_2l2bnotsolved = np.invert(mask_events_2l2bsolved) & mask_events_2l2b
		
		nEvents_solved = awkward1.count(events.event[mask_events_2l2bsolved])
		output['nevts_solved'][dataset] += nEvents_solved

		neutrinos			   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), px=pnu['x'][mask_events_2l2b], py=pnu['y'][mask_events_2l2b], pz=pnu['z'][mask_events_2l2b], mass=np.zeros_like(pnu['x'][mask_events_2l2b]))
		antineutrinos		   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), px=pnubar['x'][mask_events_2l2b], py=pnubar['y'][mask_events_2l2b], pz=pnubar['z'][mask_events_2l2b], mass=np.zeros_like(pnubar['x'][mask_events_2l2b]))
		bs					   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), px=pb['x'][mask_events_2l2b], py=pb['y'][mask_events_2l2b], pz=pb['z'][mask_events_2l2b], mass=pb['mass'][mask_events_2l2b])
		bbars				   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), px=pbbar['x'][mask_events_2l2b], py=pbbar['y'][mask_events_2l2b], pz=pbbar['z'][mask_events_2l2b], mass=pbbar['mass'][mask_events_2l2b])
		pwm, pwp, ptop, ptopbar, ptt = obj_reco(leptons_minus, leptons_plus, neutrinos, antineutrinos, bs, bbars, mask_events_2l2b)
		
		mask_events_withGoodFatJet = events.GoodFatJet.counts > 0
		higgs 				   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_withGoodFatJet, dtype=int), pt=leading_fatjet_pt[mask_events_withGoodFatJet], eta=leading_fatjet_eta[mask_events_withGoodFatJet], phi=leading_fatjet_phi[mask_events_withGoodFatJet], mass=leading_fatjet_mass[mask_events_withGoodFatJet])
		tops				   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=ptop['pt'][mask_events_2l2b], eta=ptop['eta'][mask_events_2l2b], phi=ptop['phi'][mask_events_2l2b], mass=ptop['mass'][mask_events_2l2b])
		topbars				   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=ptopbar['pt'][mask_events_2l2b], eta=ptopbar['eta'][mask_events_2l2b], phi=ptopbar['phi'][mask_events_2l2b], mass=ptopbar['mass'][mask_events_2l2b])
		tts					   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=ptt['pt'][mask_events_2l2b], eta=ptt['eta'][mask_events_2l2b], phi=ptt['phi'][mask_events_2l2b], mass=ptt['mass'][mask_events_2l2b])

		deltaRBBbar			   = calc_dr(bs, bbars)
		deltaRHiggsTop		   = calc_dr(higgs, tops)
		deltaRHiggsTopbar	   = calc_dr(higgs, topbars)
		deltaRHiggsTT		   = calc_dr(higgs, tts)
		deltaRTopTopbar		   = calc_dr(tops, topbars)
		deltaPhiBBbar		   = calc_dphi(bs, bbars)
		deltaPhiHiggsTop	   = calc_dphi(higgs, tops)
		deltaPhiHiggsTopbar	   = calc_dphi(higgs, topbars)
		deltaPhiHiggsTT		   = calc_dphi(higgs, tts)
		deltaPhiTopTopbar	   = calc_dphi(tops, topbars)

		mask_events_2l2blowdr	= mask_events_2l2b & (deltaRBBbar < 0.2)
		mask_events_2l2blowmt   = mask_events_2l2b & (ptop['mass'] < 200)
		mask_events_2l2bhighmt  = mask_events_2l2b & (ptop['mass'] > 200)
		mask_events_2l2bmw      = mask_events_2l2b & (pwp['mass'] < 200) & (pwm['mass'] < 200)
		mask_events_2l2bHbbmw   = mask_events_2l2bHbb & (pwp['mass'] < 200) & (pwm['mass'] < 200)
		mask_events_2l2bmwmt    = mask_events_2l2bmw & (ptop['mass'] < 200) & (ptopbar['mass'] < 200)
		mask_events_2l2bHbbmwmt = mask_events_2l2bHbbmw & (ptop['mass'] < 200) & (ptopbar['mass'] < 200)

		"""
		#good_events           = events[mask_events]
		mask_events_withGoodFatJet = events.GoodFatJet.counts > 0
		mask_events_withLepton = nleps > 0
		leading_leptons = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_withLepton, dtype=int), pt=leading_lepton_pt[mask_events_withLepton], eta=leading_lepton_eta[mask_events_withLepton], phi=leading_lepton_phi[mask_events_withLepton], mass=leading_lepton_mass[mask_events_withLepton])
		higgs = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_withGoodFatJet, dtype=int), pt=leading_fatjet_pt[mask_events_withGoodFatJet], eta=leading_fatjet_eta[mask_events_withGoodFatJet], phi=leading_fatjet_phi[mask_events_withGoodFatJet], mass=leading_fatjet_mass[mask_events_withGoodFatJet])
		deltaRHiggsLepton      = calc_dr(leading_leptons, higgs)
		events.LeadingFatJet["deltaRHiggsLepton"] = deltaRHiggsLepton

		self._nsolved		   += np.array(pnu['x'] > -1000).sum()
		self._n2l2b			   += np.array(mask_events_2l2b).sum()		
		"""
		# calculate weights for MC samples
		weights = {}
		weights["ones"] = np.ones(nEvents, dtype=np.float32)
		weights["nominal"] = np.ones(nEvents, dtype=np.float32)

		if is_mc:
			weights["nominal"] = weights["nominal"] * genWeight * parameters["lumi"] * samples_info[dataset]["XS"] / output["sumw"][dataset]
			#weights["nominal"] = weights["nominal"] * genWeight * parameters["lumi"] * samples_info[dataset]["XS"] / samples_info[dataset]["ngen_weight"][args.year]

			# pu corrections
			#if puWeight is not None:
			#	weights['pu'] = puWeight
				#if not uncertaintyName.startswith('puWeight'):
				#	weights['pu'] = puWeight
				#else:
				#	weights['pu'] = uncertaintyName
			#else:
		#        weights['pu'] = compute_pu_weights(parameters["pu_corrections_target"], weights["nominal"], scalars["Pileup_nTrueInt"], scalars["PV_npvsGood"])
				#weights['pu'] = compute_pu_weights(parameters["pu_corrections_target"], weights["nominal"], scalars["Pileup_nTrueInt"], scalars["Pileup_nTrueInt"])
			#weights["nominal"] = weights["nominal"] * weights['pu']

			# lepton SF corrections
			#electron_weights = compute_lepton_weights(events.GoodElectron, evaluator, ["el_triggerSF", "el_recoSF", "el_idSF"], lepton_eta=(events.GoodElectron.deltaEtaSC + events.GoodElectron.eta))
			#muon_weights = compute_lepton_weights(events.GoodMuon, evaluator, ["mu_triggerSF", "mu_isoSF", "mu_idSF"], year=args.year)
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
		'nleps'             		: nleps,
		'njets'             		: njets,
		'ngoodjets'         		: ngoodjets,
		'btags'             		: btags,
		'btags_resolved'    		: btags_resolved,
		'nfatjets'          		: nfatjets,
		'charge_sum'				: charge_sum,
		'met'               		: MET.pt,
		'mll'						: mll,
		'leading_jet_pt'    		: leading_jet_pt,
		'leading_jet_eta'   		: leading_jet_eta,
		'leading_bjet_pt'    		: leading_bjet_pt,
		'leading_bjet_eta'   		: leading_bjet_eta,
		'leadAK8JetMass'    		: leading_fatjet_SDmass,
		'leadAK8JetPt'      		: leading_fatjet_pt,
		'leadAK8JetEta'     		: leading_fatjet_eta,
		'leadAK8JetRho'     		: leading_fatjet_rho,
		'leadAK8JetHbb'				: leading_fatjet_Hbb,
		'leadAK8JetTau21'			: leading_fatjet_tau21,
		'lepton_plus_pt'            : lepton_plus_pt,
		'lepton_plus_eta'           : lepton_plus_eta,
		'lepton_minus_pt'           : lepton_minus_pt,
		'lepton_minus_eta'          : lepton_minus_eta,
		'leading_lepton_pt'         : leading_lepton_pt,
		'leading_lepton_eta'        : leading_lepton_eta,
		'ptll'                      : ptll,
		'mt_ww'                     : mt_ww,
		'pnu_x'						: pnu['x'],
		'pnu_y'						: pnu['y'],
		'pnu_z'						: pnu['z'],
		'pnubar_x'					: pnubar['x'],
		'pnubar_y'					: pnubar['y'],
		'pnubar_z'					: pnubar['z'],
		'm_w_plus'					: pwp['mass'],
		'm_w_minus'					: pwm['mass'],
		'm_top'						: ptop['mass'],
		'm_topbar'					: ptopbar['mass'],
		'm_tt'						: ptt['mass'],
		'tt_pt'						: ptt['pt'],
		'top_pt'					: ptop['pt'],
		'topbar_pt'					: ptopbar['pt'],
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
		}

		vars2d_to_plot = {
			'm_top_vs_pnu_x' : {
				'pnu_x' : abs(pnu['x']),
				'm_top' : ptop['mass'],
			},
			'm_top_vs_met' : {
				'met'   : MET.pt,
				'm_top' : ptop['mass'],
			},
			'm_top_vs_leading_lepton_pt' : {
				'leading_lepton_pt' : leading_lepton_pt,
				'm_top'             : ptop['mass'],
			},
			'm_top_vs_leadAK8JetHbb' : {
				'leadAK8JetHbb' : leading_fatjet_Hbb,
				'm_top'         : ptop['mass'],
			},
			'm_top_vs_btags' : {
				'btags' : btags,
				'm_top' : ptop['mass'],
			},
			'm_top_vs_leading_bjet_pt' : {
				'leading_bjet_pt' : leading_bjet_pt,
				'm_top'			  : ptop['mass'],
			},
			'm_top_vs_leading_bjet_eta' : {
				'leading_bjet_eta' : abs(leading_bjet_eta),
				'm_top'			   : ptop['mass'],
			},
			'm_top_vs_m_w_plus' : {
				'm_w_plus' 		   : pwp['mass'],
				'm_top'			   : ptop['mass'],
			},
			'm_top_vs_m_w_minus' : {
				'm_w_minus' 	   : pwm['mass'],
				'm_top'			   : ptop['mass'],
			},
			'm_topbar_vs_m_w_plus' : {
				'm_w_plus' 		   : pwp['mass'],
				'm_topbar'		   : ptopbar['mass'],
			},
			'm_topbar_vs_m_w_minus' : {
				'm_w_minus'		   : pwm['mass'],
				'm_topbar'		   : ptopbar['mass'],
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
				'm_w_plus' 	   	   : pwp['mass'],
				'm_w_minus'		   : pwm['mass'],
			},
			'm_topbar_vs_m_top' : {
				'm_top' 	   	   : ptop['mass'],
				'm_topbar'		   : ptopbar['mass'],
			},
		}

		for wn,w in weights.items():
			if not wn in ['ones', 'nominal']: continue
			for mask_name, mask in self._mask_events.items():
				for var_name, var in vars_to_plot.items():
					try:
						if var_name.split("_")[0] in ["muons", "goodmuons", "electrons", "goodelectrons", "jets", "goodjets"]:
							continue
						else:
							if wn == 'ones':
								output[f'hist_{var_name}_{mask_name}_weights_{wn}'].fill(dataset=dataset, values=var[mask])
							else:
								output[f'hist_{var_name}_{mask_name}_weights_{wn}'].fill(dataset=dataset, values=var[mask], weight=w[mask])

					except KeyError:
						print(f'!!!!!!!!!!!!!!!!!!!!!!!! Please add variable {var_name} to the histogram settings ({mask_name})')
				for hist2d_name, vars2d in vars2d_to_plot.items():
					#try:
					varname_x = list(vars2d.keys())[0]
					varname_y = list(vars2d.keys())[1]
					var_x = vars2d[varname_x]
					var_y = vars2d[varname_y]
					if wn == 'ones':
						output[f'hist2d_{hist2d_name}_{mask_name}_weights_{wn}'].fill(dataset=dataset, x=var_x[mask], y=var_y[mask])
					else:
						output[f'hist2d_{hist2d_name}_{mask_name}_weights_{wn}'].fill(dataset=dataset, x=var_x[mask], y=var_y[mask], weight=w[mask])
					#except KeyError:
					#	print(f'!!!!!!!!!!!!!!!!!!!!!!!! Please add variables {hist2d_name} to the histogram settings ({mask_name})')
		return output

	def postprocess(self, accumulator):

		#print("Neutrino momenta efficiency = ", self._nsolved/self._n2l2b)

		"""
		for wn in self._weights_list:
			if not wn in ['ones', 'nominal']: continue
			for mask_name in self._mask_events.keys():
				#if not mask_name in ['basic', '2l2b', '2l2bmw', '2l2bmwmt']: continue
				for var_name in self._varnames:
					if var_name.split("_")[0] in ["muons", "goodmuons", "electrons", "goodelectrons", "jets", "goodjets"]:
						continue
					fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
					hist.plot1d(accumulator[f'hist_{var_name}_{mask_name}_weights_{wn}'], overlay='dataset', fill_opts=self._fill_opts, error_opts=self._error_opts, density=True)
					plt.xlim(*self._variables[var_name]['xlim'])
					fig.savefig(plot_dir + f'hist_{var_name}_{mask_name}_weights_{wn}' + ".png", dpi=300, format="png")
					plt.close(ax.figure)
		"""

		return accumulator

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Run analysis on baconbits files using processor coffea files')
	# Inputs
	#parser.add_argument( '--wf', '--workflow', dest='workflow', choices=['dilepton'], help='Which processor to run', required=True)
	parser.add_argument('-o', '--output', default=r'hists.coffea', help='Output histogram filename (default: %(default)s)')
	parser.add_argument('--samples', '--json', dest='samplejson', help='JSON file containing dataset and file locations (default: %(default)s)', required=True)
	parser.add_argument('--year', type=str, choices=['2016', '2017', '2018'], default='2017', help='Year of data/MC samples')

	# Scale out
	parser.add_argument('--executor', choices=['iterative', 'futures', 'parsl/slurm', 'dask/condor', 'dask/slurm'], default='futures', help='The type of executor to use (default: %(default)s)')
	parser.add_argument('-j', '--workers', type=int, default=12, help='Number of workers (cores/threads) to use for multi-worker executors (e.g. futures or condor) (default: %(default)s)')
	parser.add_argument('-s', '--scaleout', type=int, default=6, help='Number of nodes to scale out to if using slurm/condor. Total number of concurrent threads is ``workers x scaleout`` (default: %(default)s)')
	parser.add_argument('--voms', default=None, type=str, help='Path to voms proxy, accessible to worker nodes. By default a copy will be made to $HOME.')

	# Debugging
	#parser.add_argument('--validate', action='store_true', help='Do not process, just check all files are accessible')
	parser.add_argument('--skipbadfiles', action='store_true', help='Skip bad files.')
	parser.add_argument('--only', type=str, default=None, help='Only process specific dataset or file')
	parser.add_argument('--limit', type=int, default=None, metavar='N', help='Limit to the first N files of each dataset in sample JSON')
	parser.add_argument('--chunk', type=int, default=500000, metavar='N', help='Number of events per process chunk')
	parser.add_argument('--max', type=int, default=None, metavar='N', help='Max number of chunks to run in total')

	parser.add_argument('--parameters', nargs='+', help='change default parameters, syntax: name value, eg --parameters met 40 bbtagging_algorithm btagDDBvL', default=None)
	#parser.add_argument('--machine', action='store', choices=['lxplus', 't3', 'local'], help="Machine: 'lxplus' or 't3'", default='lxplus', required=True)
	args = parser.parse_args()

	if args.output == parser.get_default('output'):
		label = args.samplejson.strip('.json')
		args.output = f'hists_dilepton_{label}_{args.year}.coffea'

	from definitions_dilepton_analysis import parameters, eraDependentParameters, samples_info
	parameters.update(eraDependentParameters[args.year])
	if args.parameters is not None:
		if len(args.parameters)%2 is not 0:
			raise Exception('incomplete parameters specified, quitting.')
		for p,v in zip(args.parameters[::2], args.parameters[1::2]):
			try: parameters[p] = type(parameters[p])(v) #convert the string v to the type of the parameter already in the dictionary
			except: print(f'invalid parameter specified: {p} {v}')

	if "Single" in args.samplejson:
		is_mc = False
		lumimask = LumiMask(parameters["lumimask"])
	else:
		is_mc = True
		lumimask = None

	if is_mc:
		# add information needed for MC corrections
		parameters["pu_corrections_target"] = load_puhist_target(parameters["pu_corrections_file"])
		parameters["btag_SF_target"] = BTagScaleFactor(parameters["btag_SF_{}".format(parameters["btagging_algorithm"])], BTagScaleFactor.MEDIUM)
		### this computes the lepton weights
		ext = extractor()
		print(parameters["corrections"])
		for corr in parameters["corrections"]:
			ext.add_weight_sets([corr])
		ext.finalize()
		evaluator = ext.make_evaluator()

	# load dataset
	with open(args.samplejson) as f:
		sample_dict = json.load(f)
	
	for key in sample_dict.keys():
		sample_dict[key] = sample_dict[key][:args.limit]

	# For debugging
	if args.only is not None:
		if args.only in sample_dict.keys():  # is dataset
			sample_dict = dict([(args.only, sample_dict[args.only])])
		if "*" in args.only: # wildcard for datasets
			_new_dict = {}
			print("Will only proces the following datasets:")
			for k, v in sample_dict.items():
				if k.lstrip("/").startswith(args.only.rstrip("*")):
					print("    ", k)
					_new_dict[k] = v
			sample_dict = _new_dict
		else:  # is file
			for key in sample_dict.keys():
				if args.only in sample_dict[key]:
					sample_dict = dict([(key, [args.only])])

	processor_instance = ttHbb()

	if args.executor not in ['futures', 'iterative']:
		# dask/parsl needs to export x509 to read over xrootd
		if args.voms is not None:
			_x509_path = args.voms
		else:
			_x509_localpath = [l for l in os.popen('voms-proxy-info').read().split("\n") if l.startswith('path')][0].split(":")[-1].strip()
			_x509_path = os.environ['HOME'] + f'/.{_x509_localpath.split("/")[-1]}'
			os.system(f'cp {_x509_localpath} {_x509_path}')

		env_extra = [
			'export XRD_RUNFORKHANDLER=1',
			f'export X509_USER_PROXY={_x509_path}',
			#f'export X509_CERT_DIR={os.environ["X509_CERT_DIR"]}',
			f'export X509_VOMS_DIR={os.environ["X509_VOMS_DIR"]}',
			'ulimit -u 32768',
		]

	#########
	# Execute
	if args.executor in ['futures', 'iterative']:
		import uproot4 as uproot
		uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
		if args.executor == 'iterative':
			_exec = processor.iterative_executor
		else:
			_exec = processor.futures_executor
		output = processor.run_uproot_job(sample_dict,
									treename='Events',
									processor_instance=processor_instance,
									executor=_exec,
									executor_args={
										'nano' : True,
										'skipbadfiles':args.skipbadfiles,
										'schema': processor.NanoAODSchema, 
										'workers': args.workers},
									chunksize=args.chunk, maxchunks=args.max
									)
	elif args.executor == 'parsl/slurm':
		import parsl
		from parsl.providers import LocalProvider, CondorProvider, SlurmProvider
		from parsl.channels import LocalChannel
		from parsl.config import Config
		from parsl.executors import HighThroughputExecutor
		from parsl.launchers import SrunLauncher
		from parsl.addresses import address_by_hostname

		slurm_htex = Config(
			executors=[
				HighThroughputExecutor(
					label="coffea_parsl_slurm",
					address=address_by_hostname(),
					prefetch_capacity=0,
					provider=SlurmProvider(
						channel=LocalChannel(script_dir='logs_parsl'),
						launcher=SrunLauncher(),
						max_blocks=(args.scaleout)+10,
						init_blocks=args.scaleout, 
						partition='wn',
						worker_init="\n".join(env_extra) + "\nexport PYTHONPATH=$PYTHONPATH:$PWD", 
						walltime='02:00:00'
					),
					#cores_per_worker=1,
					#mem_per_worker=2, #GB
				)
			],
			retries=20,
		)
		dfk = parsl.load(slurm_htex)

		output = processor.run_uproot_job(sample_dict,
									treename='Events',
									processor_instance=processor_instance,
									executor=processor.parsl_executor,
									executor_args={
										'skipbadfiles':True,
										'schema': processor.NanoAODSchema, 
										'config': None,
									},
									chunksize=args.chunk, maxchunks=args.max
									)
		
	elif 'dask' in args.executor:
		from dask_jobqueue import SLURMCluster, HTCondorCluster
		from distributed import Client
		from dask.distributed import performance_report

		if 'slurm' in args.executor:
			cluster = SLURMCluster(
				queue='all',
				cores=args.workers,
				processes=args.workers,
				memory="200 GB",
				retries=10,
				walltime='00:30:00',
				env_extra=env_extra,
			)
		elif 'condor' in args.executor:
			cluster = HTCondorCluster(
				 cores=args.workers, 
				 memory='2GB', 
				 disk='2GB', 
				 env_extra=env_extra,
			)
		cluster.scale(jobs=args.scaleout)

		client = Client(cluster)
		with performance_report(filename="dask-report.html"):
			output = processor.run_uproot_job(sample_dict,
										treename='Events',
										processor_instance=processor_instance,
										executor=processor.dask_executor,
										executor_args={
											'client': client,
											'skipbadfiles':args.skipbadfiles,
											#'schema': processor.NanoAODSchema, 
										},
										chunksize=args.chunk, maxchunks=args.max
							)

	hist_dir = os.getcwd() + "/histograms/"
	if not os.path.exists(hist_dir):
		os.makedirs(hist_dir)
	save(output, hist_dir + args.output)

	print(f"Saving output to {hist_dir + args.output}")
