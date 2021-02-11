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

class NanoProcessor(processor.ProcessorABC):
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
		  '2l2b'        : None,
		  '2l2bsolved'  : None,
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
		})

		for wn in self._weights_list:
			for mask_name in self._mask_events.keys():
				for var_name in self._varnames:
					self._accumulator.add(processor.dict_accumulator({f'hist_{var_name}_{mask_name}_weights_{wn}' : hist.Hist("a.u.",
																  	  hist.Cat("dataset", "Dataset"),
																	  hist.Bin("values", self._variables[var_name]['xlabel'], np.linspace( *self._variables[var_name]['binning'] ) ) ) } ))
				for hist2d_name in self._hist2dnames:
					varname_x = list(self._variables2d[hist2d_name].keys())[0]
					varname_y = list(self._variables2d[hist2d_name].keys())[1]
					self._accumulator.add(processor.dict_accumulator({f'hist2d_{hist2d_name}_{mask_name}_weights_{wn}' : hist.Hist("a.u.",
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
		nEvents = events.event.size
		sample = dataset
		is_mc = 'genWeight' in events.columns
		#print("Processing %d %s events" % (nEvents, dataset))

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

		leptons_plus		   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=lepton_plus_pt[mask_events_2l2b], eta=lepton_plus_eta[mask_events_2l2b], phi=lepton_plus_phi[mask_events_2l2b], mass=lepton_plus_mass[mask_events_2l2b])
		leptons_minus		   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=lepton_minus_pt[mask_events_2l2b], eta=lepton_minus_eta[mask_events_2l2b], phi=lepton_minus_phi[mask_events_2l2b], mass=lepton_minus_mass[mask_events_2l2b])
		goodbjets			   = JaggedCandidateArray.candidatesfromcounts(events.GoodBJet.counts, pt=events.GoodBJet.pt.flatten(), eta=events.GoodBJet.eta.flatten(), phi=events.GoodBJet.phi.flatten(), mass=events.GoodBJet.mass.flatten())
		METs_2b			   	   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=MET.pt[mask_events_2l2b], eta=np.zeros_like(MET.pt[mask_events_2l2b]), phi=MET.phi[mask_events_2l2b], mass=np.zeros_like(MET.pt[mask_events_2l2b]))
		pnu, pnubar, pb, pbbar, mask_events_2l2bsolved = pnuCalculator(leptons_minus, leptons_plus, goodbjets, METs_2b)
		#efficiency			   = np.array(pnu['x'] > -1000).sum()/len(pnu['x'])
		#print(efficiency)
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

		#m_w_plus			   = w_mass(leptons_plus, neutrinos, mask_events_2l2b)
		#m_w_minus			   = w_mass(leptons_minus, antineutrinos, mask_events_2l2b)
		#m_top				   = t_mass(leptons_plus, neutrinos, bs, mask_events_2l2b)
		#m_top_bar			   = t_mass(leptons_minus, antineutrinos, bbars, mask_events_2l2b)
		#m_tt 				   = top_reco(leptons_minus, leptons_plus, neutrinos, antineutrinos, bs, bbars, mask_events_2l2b)

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
			weights["nominal"] = weights["nominal"] * genWeight * parameters["lumi"] * samples_info[sample]["XS"] / samples_info[sample]["ngen_weight"][args.year]

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
		  '2l2b'     	: mask_events_2l2b,
		  '2l2bsolved'  : mask_events_2l2bsolved,
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

		if not args.test:
			hist_dir = os.getcwd() + "/histograms/"
			print("Saving histograms in " + hist_dir)
			if not os.path.exists(hist_dir):
				os.makedirs(hist_dir)
			save(accumulator, hist_dir + args.output)

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
	parser = argparse.ArgumentParser(description='Runs a simple array-based analysis')
	parser.add_argument('--outdir', action='store', help='directory to store outputs', type=str, default=os.getcwd())
	parser.add_argument('-o', '--output', action='store', help='Output histogram filename (default: %(default)s)', type=str, default=r'hists.coffea')
	parser.add_argument('--sample', action='store', help='sample name', choices=['mc'], type=str, default=None, required=True)
	parser.add_argument('--year', action='store', choices=['2016', '2017', '2018'], help='Year of data/MC samples', default='2017')
	parser.add_argument('--parameters', nargs='+', help='change default parameters, syntax: name value, eg --parameters met 40 bbtagging_algorithm btagDDBvL', default=None)
	parser.add_argument('--machine', action='store', choices=['lxplus', 't3', 'local'], help="Machine: 'lxplus' or 't3'", default='lxplus', required=True)
	parser.add_argument('--workers', action='store', help='Number of workers (CPUs) to use', type=int, default=10)
	parser.add_argument('--chunksize', action='store', help='Number of events in a single chunk', type=int, default=30000)
	parser.add_argument('--maxchunks', action='store', help='Maximum number of chunks', type=int, default=25)
	parser.add_argument('--test', action='store_true', help='Test without plots', default=False)
	args = parser.parse_args()

	if args.output == parser.get_default('output'):
		args.output = f'hists_dilepton_{args.sample}{args.year}.coffea'

	from definitions_dilepton_analysis import parameters, eraDependentParameters, samples_info
	parameters.update(eraDependentParameters[args.year])
	if args.parameters is not None:
		if len(args.parameters)%2 is not 0:
			raise Exception('incomplete parameters specified, quitting.')
		for p,v in zip(args.parameters[::2], args.parameters[1::2]):
			try: parameters[p] = type(parameters[p])(v) #convert the string v to the type of the parameter already in the dictionary
			except: print(f'invalid parameter specified: {p} {v}')

	if "Single" in args.sample:
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

	# PostProcessed NanoAOD
	#f1 = open("datasets/RunIIFall17NanoAODv7PostProc/ttHTobb_2017.txt", 'r')
	#f2 = open("datasets/RunIIFall17NanoAODv7PostProc/TTTo2L2Nu_2017.txt", 'r')
	# Central NanoAOD
	if args.sample == 'mc' and args.year == '2017':
		f1 = open("datasets/RunIIFall17NanoAODv7/ttHTobb_2017.txt", 'r')
		f2 = open("datasets/RunIIFall17NanoAODv7/TTTo2L2Nu_2017_localfiles.txt", 'r')
		#f3 = open("datasets/RunIIFall17NanoAODv7/TTToSemiLeptonic_2017.txt", 'r')
		#samples = { "TTToSemiLeptonic": f3.read().splitlines(), "TTTo2L2Nu": f2.read().splitlines(), "ttHTobb": f1.read().splitlines() }
		samples = { "TTTo2L2Nu": f2.read().splitlines(), "ttHTobb": f1.read().splitlines() }
		f1.close()
		f2.close()
		#f3.close()
	if args.machine == 't3':
		for sample in samples:
			for (i, file) in enumerate(samples[sample]):
				samples[sample][i] = file.replace('root://xrootd-cms.infn.it/', '/pnfs/psi.ch/cms/trivcat')
	if args.machine == 'local':
		print("Reading local test files...")
		samples = {
			"ttHTobb": [
				"/afs/cern.ch/work/m/mmarcheg/Coffea/test/nano_postprocessed_24_ttHbb.root",
			],
			"TTTo2L2Nu": [
				"/afs/cern.ch/work/m/mmarcheg/Coffea/test/nano_postprocessed_45_tt_semileptonic.root"
			]
		}

	MyProcessor = ttHbb()
	#MyProcessor = ttHbb(sample=args.sample)

	print("Running uproot job...")
	result = processor.run_uproot_job(
		samples,
		"Events",
		MyProcessor,
		processor.futures_executor,
		{"nano": True, "workers": args.workers},
		chunksize=args.chunksize,
		maxchunks=args.maxchunks,
	)
