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
from uproot_methods import TLorentzVectorArray

from lib_analysis import lepton_selection, jet_selection, jet_nohiggs_selection, get_charge_sum, get_dilepton_vars, get_transverse_mass, get_charged_var, get_leading_value, load_puhist_target, compute_lepton_weights, calc_dr, pnuCalculator
from definitions_dilepton_analysis import parameters, histogram_settings, samples_info

class ttHbb(processor.ProcessorABC):
	def __init__(self):
		#self.sample = sample
		self._variables = histogram_settings['variables']
		self._varnames = self._variables.keys()
		self._fill_opts = histogram_settings['fill_opts']
		self._error_opts = histogram_settings['error_opts']
		self._mask_events_list = [
		  'resolved',
		  'basic',
		  '2J',
		  '2J2W',
		  '2J2WdeltaR',
		  '2J2WdeltaR_Pass',
		  '2J2WdeltaR_Fail',
		  'basic_orthogonal',
		  '2J_orthogonal',
		  '2J2W_orthogonal',
		  '2J2WdeltaR_orthogonal',
		  '2J2WdeltaR_Pass_orthogonal',
		  '2J2WdeltaR_Fail_orthogonal',
		  #'basic_overlap',
		  #'2J_overlap',
		  #'2J2W_overlap',
		  #'2J2WdeltaR_overlap',
		  #'2J2WdeltaR_Pass_overlap',
		  #'2J2WdeltaR_Fail_overlap',
		]

		self._accumulator = processor.dict_accumulator({
			"sumw": processor.defaultdict_accumulator(float),
		})

		for var_name in self._varnames:
			self._accumulator.add(processor.dict_accumulator({var_name : hist.Hist("entries",
														  	  hist.Cat("dataset", "Dataset"),
														  	  hist.Bin("values", self._variables[var_name]['xlabel'], np.linspace( *self._variables[var_name]['binning'] ) ) ) } ))

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
		puWeight = events.puWeight
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
		ngoodmuons     = muons[good_muons].counts[mask_events]
		ngoodelectrons = electrons[good_electrons].counts[mask_events]
		nmuons         = muons[good_muons | veto_muons].counts[mask_events]
		nelectrons     = electrons[good_electrons | veto_electrons].counts[mask_events]
		ngoodleps      = ngoodmuons + ngoodelectrons
		nleps          = nmuons + nelectrons
		#lepton_veto    = muons[veto_muons].counts[mask_events] + electrons[veto_electrons].counts[mask_events]
		njets          = jets[nonbjets].counts[mask_events]
		ngoodjets      = jets[good_jets].counts[mask_events]
		btags          = jets[bjets].counts[mask_events]
		btags_resolved = jets[bjets_resolved].counts[mask_events]
		nfatjets       = fatjets[good_fatjets].counts[mask_events]
		#nhiggs         = fatjets[higgs_candidates].counts[mask_events]

		# trigger logic
		trigger_el = (nleps==2) & (nelectrons==2)
		trigger_el_mu = (nleps==2) & (nelectrons==1) & (nmuons==1)
		trigger_mu = (nleps==2) & (nmuons==2)
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
		#mask_events_higgs = mask_events & (nleps == 1) & (MET.pt > 20) & (nhiggs > 0) & (njets > 1)  # & np.invert( (njets >= 4) & (btags >=2) ) & (lepton_veto == 0)
		mask_events_boost = (mask_events & (nleps == 2) & (ngoodleps >= 1) & (charge_sum == 0) &
							(MET.pt > parameters['met']) & (nfatjets > 0) & (btags >= parameters['btags']) &
							(mll > 20) & ((SFOS & ((mll < 76) | (mll > 106))) | not_SFOS) ) # & (btags_resolved < 3)# & (njets > 1)  # & np.invert( (njets >= 4)  )
		mask_events_OS    = (mask_events_res | mask_events_boost)

		# calculate basic variables
		leading_jet_pt         = get_leading_value(events.GoodJet.pt)
		leading_jet_eta        = get_leading_value(events.GoodJet.eta, default=-9.)
		leading_jet_phi        = get_leading_value(events.GoodJet.phi)
		leading_fatjet_SDmass  = get_leading_value(events.GoodFatJet.msoftdrop)
		leading_fatjet_pt      = get_leading_value(events.GoodFatJet.pt)
		leading_fatjet_eta     = get_leading_value(events.GoodFatJet.eta, default=-9.)
		leading_fatjet_phi     = get_leading_value(events.GoodFatJet.phi)
		leading_fatjet_mass    = get_leading_value(events.GoodFatJet.mass)
		leading_fatjet_rho     = awkward1.from_iter( np.log(leading_fatjet_SDmass**2 / leading_fatjet_pt**2) )
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
		mask_events_2l2b	   = mask_events_boost & (btags >= 2)
		leptons_plus		   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=lepton_plus_pt[mask_events_2l2b], eta=lepton_plus_eta[mask_events_2l2b], phi=lepton_plus_phi[mask_events_2l2b], mass=lepton_plus_mass[mask_events_2l2b])
		leptons_minus		   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=lepton_minus_pt[mask_events_2l2b], eta=lepton_minus_eta[mask_events_2l2b], phi=lepton_minus_phi[mask_events_2l2b], mass=lepton_minus_mass[mask_events_2l2b])
		goodbjets			   = JaggedCandidateArray.candidatesfromcounts(events.GoodBJet.counts, pt=events.GoodBJet.pt.flatten(), eta=events.GoodBJet.eta.flatten(), phi=events.GoodBJet.phi.flatten(), mass=events.GoodBJet.mass.flatten())
		#goodbjets			   = JaggedCandidateArray.candidatesfromcounts(np.where(mask_events_2l2b, events.GoodBJet.counts, 0), pt=events.GoodBJet.pt[mask_events_2l2b].flatten(), eta=events.GoodBJet.eta[mask_events_2l2b].flatten(), phi=events.GoodBJet.phi[mask_events_2l2b].flatten(), mass=events.GoodBJet.mass[mask_events_2l2b].flatten())
		METs_2b			   	   = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_2l2b, dtype=int), pt=MET.pt[mask_events_2l2b], eta=np.zeros_like(MET.pt[mask_events_2l2b]), phi=MET.phi[mask_events_2l2b], mass=np.zeros_like(MET.pt[mask_events_2l2b]))
		pnu, pnubar			   = pnuCalculator(leptons_minus, leptons_plus, goodbjets, METs_2b)

		"""
		#good_events           = events[mask_events]
		mask_events_withGoodFatJet = events.GoodFatJet.counts > 0
		mask_events_withLepton = nleps > 0
		leading_leptons = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_withLepton, dtype=int), pt=leading_lepton_pt[mask_events_withLepton], eta=leading_lepton_eta[mask_events_withLepton], phi=leading_lepton_phi[mask_events_withLepton], mass=leading_lepton_mass[mask_events_withLepton])
		higgs = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_withGoodFatJet, dtype=int), pt=leading_fatjet_pt[mask_events_withGoodFatJet], eta=leading_fatjet_eta[mask_events_withGoodFatJet], phi=leading_fatjet_phi[mask_events_withGoodFatJet], mass=leading_fatjet_mass[mask_events_withGoodFatJet])
		deltaRHiggsLepton      = calc_dr(leading_leptons, higgs)
		events.LeadingFatJet["deltaRHiggsLepton"] = deltaRHiggsLepton
		"""

		# calculate weights for MC samples
		weights = {}
		weights["ones"] = np.ones(nEvents, dtype=np.float32)
		weights["nominal"] = np.ones(nEvents, dtype=np.float32)

		if is_mc:
			weights["nominal"] = weights["nominal"] * genWeight * parameters["lumi"] * samples_info[sample]["XS"] / samples_info[sample]["ngen_weight"][args.year]

			# pu corrections
			if puWeight is not None:
				weights['pu'] = puWeight
				#if not uncertaintyName.startswith('puWeight'):
				#	weights['pu'] = puWeight
				#else:
				#	weights['pu'] = uncertaintyName
			#else:
		#        weights['pu'] = compute_pu_weights(parameters["pu_corrections_target"], weights["nominal"], scalars["Pileup_nTrueInt"], scalars["PV_npvsGood"])
				#weights['pu'] = compute_pu_weights(parameters["pu_corrections_target"], weights["nominal"], scalars["Pileup_nTrueInt"], scalars["Pileup_nTrueInt"])
			weights["nominal"] = weights["nominal"] * weights['pu']

			# lepton SF corrections
			#electron_weights = compute_lepton_weights(events.GoodElectron, evaluator, ["el_triggerSF", "el_recoSF", "el_idSF"], lepton_eta=(events.GoodElectron.deltaEtaSC + events.GoodElectron.eta))
			#muon_weights = compute_lepton_weights(events.GoodMuon, evaluator, ["mu_triggerSF", "mu_isoSF", "mu_idSF"], year=args.year)
			#weights['lepton']  = muon_weights * electron_weights
			#weights["nominal"] = weights["nominal"] * weights['lepton']

		mask_events = {
		  'trigger'  : mask_events_trigger,
		  'resolved' : mask_events_res,
		  'basic'    : mask_events_boost,
		  'joint'    : mask_events_OS
		}
		#mask_events['2J']   = mask_events['basic'] & (njets>1)

		"""
		#Ws reconstruction
		leading_leptons = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events['2J'], dtype=int), pt=leading_lepton_pt[mask_events['2J']], eta=leading_lepton_eta[mask_events['2J']], phi=leading_lepton_phi[mask_events['2J']], mass=leading_lepton_mass[mask_events['2J']])
		METs = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events['2J'], dtype=int), pt=MET.pt[mask_events['2J']], eta=np.zeros_like(MET.pt[mask_events['2J']]), phi=MET.phi[mask_events['2J']], mass=np.zeros_like(MET.pt[mask_events['2J']]))
		pznu = METzCalculator(leading_leptons.p4, METs.p4)
		neutrinos = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events['2J'], dtype=int), px=METs.p4.x.content, py=METs.p4.y.content, pz=pznu, mass=np.zeros_like(METs.p4.x.content))
		lepW = leading_leptons.p4 + neutrinos.p4

		good_jets_p4 = JaggedCandidateArray.candidatesfromcounts(np.where(mask_events['2J'], events.GoodJet.counts, np.zeros_like(events.GoodJet.counts)), pt=events.GoodJet.pt[mask_events['2J']].flatten(), eta=events.GoodJet.eta[mask_events['2J']].flatten(), phi=events.GoodJet.phi[mask_events['2J']].flatten(), mass=events.GoodJet.mass[mask_events['2J']].flatten())

		hadW, n_hadW = hadronic_W(good_jets_p4)

		#print(awkward1.any(lepW.mass>parameters['W']['min_mass'], axis=1))
		#mask_events['2J2W'] = mask_events['2J'] & (hadW.mass>parameters['W']['min_mass']) & (hadW.mass<parameters['W']['max_mass']) & (lepW.mass>parameters['W']['min_mass']) & (lepW.mass<parameters['W']['max_mass'])
		mask_events['2J2W'] = mask_events['2J'] & awkward1.any(hadW.mass>parameters['W']['min_mass'], axis=1) & awkward1.any(hadW.mass<parameters['W']['max_mass'], axis=1) & awkward1.any(lepW.mass>parameters['W']['min_mass'], axis=1) & awkward1.any(lepW.mass<parameters['W']['max_mass'], axis=1)

		#deltaR between objects
		lepWs = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events['2J'], dtype=int), pt=lepW.pt.flatten(), eta=lepW.eta.flatten(), phi=lepW.phi.flatten(), mass=lepW.mass.flatten())
		hadWs = JaggedCandidateArray.candidatesfromcounts(n_hadW, pt=hadW.pt.flatten(), eta=hadW.eta.flatten(), phi=hadW.phi.flatten(), mass=hadW.mass.flatten())
		#higgs = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events['2J'], dtype=int), pt=leading_fatjet_pt[mask_events['2J']], eta=leading_fatjet_eta[mask_events['2J']], phi=leading_fatjet_phi[mask_events['2J']], mass=leading_fatjet_mass[mask_events['2J']])
		deltaRlepWHiggs = calc_dr(lepWs, higgs)
		deltaRhadWHiggs = calc_dr(hadWs, higgs)

#		mask_events['2J2WdeltaR'] = mask_events['2J2W'] & (deltaRlepWHiggs>1.5) & (deltaRhadWHiggs>1.5) & (deltaRlepWHiggs<4) & (deltaRhadWHiggs<4)
		mask_events['2J2WdeltaR'] = mask_events['2J2W'] & (deltaRlepWHiggs>1) & (deltaRhadWHiggs>1) # & (deltaRlepWHiggs<4) & (deltaRhadWHiggs<4)

		#boosted Higgs
		leading_fatjet_tau1     = get_leading_value(events.GoodFatJet.tau1, default=999.9)
		leading_fatjet_tau2     = get_leading_value(events.GoodFatJet.tau2)
		leading_fatjet_tau21    = np.divide(leading_fatjet_tau2, leading_fatjet_tau1)

		leading_fatjet_Hbb = get_leading_value(getattr(fatjets, parameters["bbtagging_algorithm"]))
		for m in ['2J2WdeltaR']:#, '2J2WdeltaRTau21']:#, '2J2WdeltaRTau21DDT']:
			mask_events[f'{m}_Pass'] = mask_events[m] & (leading_fatjet_Hbb>parameters['bbtagging_WP'])
			mask_events[f'{m}_Fail'] = mask_events[m] & (leading_fatjet_Hbb<=parameters['bbtagging_WP'])
		"""
		#mask_events['overlap'] = mask_events['2J2WdeltaR'] & mask_events['resolved']
		#mask_events['overlap'] = mask_events['2J2WdeltaR_Pass'] & mask_events['resolved']

		############# overlap study
		for m in mask_events.copy():
			if m=='resolved': continue
			mask_events[m+'_orthogonal'] = mask_events[m] & (btags_resolved < 3)
			#mask_events[m+'_overlap']    = mask_events[m] & mask_events['resolved']
		for mn,m in mask_events.items():
			output["sumw"]['nevts_'+mn] += sum(weights['nominal'][m])

		vars2d = {
			'ngoodjets' : ngoodjets,
			'njets'     : njets
			}

		"""
		for mn,m in mask_events.items():
			if 'overlap' in mn:
				for vn,v in vars2d.items():
					hist, binsx, binsy = np.histogram2d(v[m], btags_resolved[m],\
							bins=(\
							np.linspace(*self._variables[vn]),\
							np.linspace(*self._variables['btags_resolved']),\
							),\
							weights=weights["nominal"][m]\
							)
					ret[f'hist2d_{vn}VSbtags_{mn}'] = Histogram( hist, hist, (*self._variables[vn],*self._variables['btags_resolved']) )
		"""
		############# histograms
		vars_to_plot = {
		'muons_pt'					: muons.pt.flatten(),
		'muons_eta'					: muons.eta.flatten(),
		'goodmuons_pt'				: events.GoodMuon.pt[mask_events['trigger']].flatten(),
		'goodmuons_eta'				: events.GoodMuon.eta[mask_events['trigger']].flatten(),
		'goodmuons_res_pt'			: events.GoodMuon.pt[mask_events['resolved']].flatten(),
		'goodmuons_res_eta'			: events.GoodMuon.eta[mask_events['resolved']].flatten(),
		'goodmuons_boost_pt'		: events.GoodMuon.pt[mask_events['basic']].flatten(),
		'goodmuons_boost_eta'		: events.GoodMuon.eta[mask_events['basic']].flatten(),
		'goodmuons_cuts_pt'			: events.GoodMuon.pt[mask_events['joint']].flatten(),
		'goodmuons_cuts_eta'		: events.GoodMuon.eta[mask_events['joint']].flatten(),
		'electrons_pt'				: electrons.pt.flatten(),
		'electrons_eta'				: electrons.eta.flatten(),
		'goodelectrons_pt'			: events.GoodElectron.pt[mask_events['trigger']].flatten(),
		'goodelectrons_eta'			: events.GoodElectron.eta[mask_events['trigger']].flatten(),
		'goodelectrons_res_pt'		: events.GoodElectron.pt[mask_events['resolved']].flatten(),
		'goodelectrons_res_eta'		: events.GoodElectron.eta[mask_events['resolved']].flatten(),
		'goodelectrons_boost_pt'	: events.GoodElectron.pt[mask_events['basic']].flatten(),
		'goodelectrons_boost_eta'	: events.GoodElectron.eta[mask_events['basic']].flatten(),
		'goodelectrons_cuts_pt'		: events.GoodElectron.pt[mask_events['joint']].flatten(),
		'goodelectrons_cuts_eta'	: events.GoodElectron.eta[mask_events['joint']].flatten(),
		'jets_pt'					: jets.pt.flatten(),
		'jets_eta'					: jets.eta.flatten(),
		'goodjets_pt'				: events.GoodJet.pt[mask_events['trigger']].flatten(),
		'goodjets_eta'				: events.GoodJet.eta[mask_events['trigger']].flatten(),
		'goodjets_cuts_pt'			: events.GoodJet.pt[mask_events['joint']].flatten(),
		'goodjets_cuts_eta'			: events.GoodJet.eta[mask_events['joint']].flatten(),
		'nleps'             		: nleps,
		'nleps_cuts'             	: nleps[mask_events['joint']],
		'njets'             		: njets,
		'njets_cuts'             	: njets[mask_events['joint']],
		'ngoodjets'         		: ngoodjets,
		'ngoodjets_cuts'      		: ngoodjets[mask_events['joint']],
		'btags'             		: btags,
		'btags_cuts'           		: btags[mask_events['joint']],
		'btags_resolved'    		: btags_resolved,
		'btags_resolved_cuts'  		: btags_resolved[mask_events['joint']],
		'nfatjets'          		: nfatjets,
		'nfatjets_cuts'        		: nfatjets[mask_events['joint']],
		'charge_sum'				: charge_sum,
		'charge_sum_cuts'			: charge_sum[mask_events['joint']],
		'met'               		: MET.pt.flatten(),
		'met_cuts'             		: MET.pt[mask_events['joint']].flatten(),
		'mll'						: mll,
		'mll_cuts'					: mll[mask_events['joint']],
		'leading_jet_pt'    		: leading_jet_pt[mask_events['joint']],
		'leading_jet_eta'   		: leading_jet_eta[mask_events['joint']],
		'leadAK8JetMass'    		: leading_fatjet_SDmass[mask_events['joint']],
		'leadAK8JetMass_boost' 		: leading_fatjet_SDmass[mask_events['basic']],
		'leadAK8JetPt'      		: leading_fatjet_pt[mask_events['joint']],
		'leadAK8JetPt_boost'   		: leading_fatjet_pt[mask_events['basic']],
		'leadAK8JetEta'     		: leading_fatjet_eta[mask_events['joint']],
		'leadAK8JetEta_boost'  		: leading_fatjet_eta[mask_events['basic']],
		#'leadAK8JetHbb'     		: leading_fatjet_Hbb[mask_events['joint']],
		#'leadAK8JetHbb_boost' 		: leading_fatjet_Hbb[mask_events['basic']],
		#'leadAK8JetTau21'   		: leading_fatjet_tau21[mask_events['joint']],
		#'leadAK8JetTau21_boost'	: leading_fatjet_tau21[mask_events['basic']],
		'leadAK8JetRho'     		: leading_fatjet_rho[mask_events['joint']],
		'leadAK8JetRho_boost'   	: leading_fatjet_rho[mask_events['basic']],
		'lepton_plus_pt'            : lepton_plus_pt[mask_events['joint']],
		'lepton_plus_eta'           : lepton_plus_eta[mask_events['joint']],
		'lepton_minus_pt'           : lepton_minus_pt[mask_events['joint']],
		'lepton_minus_eta'          : lepton_minus_eta[mask_events['joint']],
		'leading_lepton_pt'         : leading_lepton_pt[mask_events['joint']],
		'leading_lepton_eta'        : leading_lepton_eta[mask_events['joint']],
		'ptll'                      : ptll,
		'ptll_cuts'                 : ptll[mask_events['joint']],
		'mt_ww'                     : mt_ww,
		'mt_ww_cuts'                : mt_ww[mask_events['joint']],
		'pnu_x'						: pnu['x'][pnu['x'] > -999],
		'pnu_y'						: pnu['y'][pnu['y'] > -999],
		'pnu_z'						: pnu['z'][pnu['z'] > -999],
		'pnubar_x'					: pnubar['x'][pnubar['x'] > -999],
		'pnubar_y'					: pnubar['y'][pnubar['y'] > -999],
		'pnubar_z'					: pnubar['z'][pnubar['z'] > -999],
		#'hadWPt'            : get_leading_value(hadW.pt),
		#'hadWEta'           : get_leading_value(hadW.eta),
		#'hadWMass'          : get_leading_value(hadW.mass),
		#'lepWPt'            : get_leading_value(lepW.pt),
		#'lepWEta'           : get_leading_value(lepW.eta),
		#'lepWMass'          : get_leading_value(lepW.mass),
		#'deltaRlepWHiggs'   : deltaRlepWHiggs,
		#'deltaRhadWHiggs'   : deltaRhadWHiggs,
		#'deltaRHiggsLepton' : deltaRHiggsLepton,
		#'PV_npvsGood'       : scalars['PV_npvsGood'],
		}

		"""
		if is_mc:
			for wn,w in weights.items():
				vars_to_plot[f'weights_{wn}'] = w

		#var_name, var = 'leadAK8JetMass', leading_fatjet_SDmass
		vars_split = ['leadAK8JetMass', 'leadAK8JetRho']
		ptbins = np.append( np.arange(250,600,50), [600, 1000, 5000] )
		for var_name in vars_split:
			var = vars_to_plot[var_name]
			for ipt in range( len(ptbins)-1 ):
				for m in ['2J2WdeltaR']:#, '2J2WdeltaRTau21']:#, '2J2WdeltaRTau21DDT']:
					for r in ['Pass','Fail']:
						for o in ['','_orthogonal']:
							mask_name = f'{m}_{r}{o}'
							if not mask_name in mask_events: continue
							mask = mask_events[mask_name] & (leading_fatjet_pt>ptbins[ipt]) & (leading_fatjet_pt<ptbins[ipt+1])
							output[f'hist_{var_name}_{mask_name}_pt{ptbins[ipt]}to{ptbins[ipt+1]}'].fill(dataset=dataset, values=var[mask], weight=weights['nominal'][mask])

		for wn,w in weights.items():
			if wn != 'nominal': continue
			#ret[f'nevts_overlap{weight_name}'] = Histogram( [sum(weights[w]), sum(weights[w][mask_events['2J2WdeltaR']]), sum(weights[w][mask_events['resolved']]), sum(weights[w][mask_events['overlap']])], 0,0 )
			for mask_name, mask in mask_events.items():
				if not 'deltaR' in mask_name: continue
				for var_name, var in vars_to_plot.items():
					#if (not is_mc) and ('Pass' in mask_name) and (var_name=='leadAK8JetMass') : continue
					try:
						#print(var_name, mask_name)
						output[f'hist_{var_name}_{mask_name}_weights_{wn}'].fill(dataset=dataset, values=var[mask], weight=w[mask])
					except KeyError:
						print(f'!!!!!!!!!!!!!!!!!!!!!!!! Please add variable {var_name} to the histogram settings')
		"""

		for var_name, var in vars_to_plot.items():
			try:
				output[var_name].fill(dataset=dataset, values=var)
			except KeyError:
				print(f'!!!!!!!!!!!!!!!!!!!!!!!! Please add variable {var_name} to the histogram settings')

		return output

	def postprocess(self, accumulator):
		plot_dir = "plots/dilepton/"
		print("Saving plots in " + plot_dir)

		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)

		for var_name in self._varnames:
			fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
			hist.plot1d(accumulator[var_name], overlay='dataset', fill_opts=self._fill_opts, error_opts=self._error_opts)
			plt.xlim(*self._variables[var_name]['xlim'])
			fig.savefig(plot_dir + var_name + ".png", dpi=300, format="png")
			plt.close(ax.figure)

		"""
		plot_dir = "plots/comparison/"
		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)
		print("Saving histograms in " + plot_dir)
		datasets = []
		for item in accumulator.keys():
			if "hist" in item:
				datasets = accumulator[item].values().keys()
				break
		hist_dir = plot_dir + "nominal/"
		if not os.path.exists(hist_dir):
			os.makedirs(hist_dir)
		for dataset in datasets:
			data = {}
			for histo in [item for item in accumulator.keys() if "hist" in item]:
				dataset_label = str(dataset).strip("'(),")
				d = {}
				d['contents'] = accumulator[histo].values()[dataset].tolist()
				identifiers = accumulator[histo].identifiers('values')
				d['edges'] = [item.lo for item in identifiers] + [identifiers[-1].hi]
				data[histo] = d
			with open(hist_dir + 'out_' + dataset_label + '_nominal_merged.json', 'w') as outfile:
				json.dump(data, outfile, sort_keys=True, indent=4)
		"""
		return accumulator

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Runs a simple array-based analysis')
	#parser.add_argument('--use-cuda', action='store_true', help='Use the CUDA backend')
	#parser.add_argument('--from-cache', action='store_true', help='Load from cache (otherwise create it)')
	#parser.add_argument('--nthreads', action='store', help='Number of CPU threads to use', type=int, default=4, required=False)
	#parser.add_argument('--files-per-batch', action='store', help='Number of files to process per batch', type=int, default=1, required=False)
	#parser.add_argument('--cache-location', action='store', help='Path prefix for the cache, must be writable', type=str, default=os.path.join(os.getcwd(), 'cache'))
	#parser.add_argument('--outdir', action='store', help='directory to store outputs', type=str, default=os.getcwd())
	#parser.add_argument('--outtag', action='store', help='outtag added to output file', type=str, default="")
	#parser.add_argument('--version', action='store', help='tag added to the output directory', type=str, default='')
	#parser.add_argument('--filelist', action='store', help='List of files to load', type=str, default=None, required=False)
	parser.add_argument('--sample', action='store', help='sample name', type=str, default=None, required=True)
	#parser.add_argument('--categories', nargs='+', help='categories to be processed (default: sl_jge4_tge2)', default="sl_jge4_tge2")
	#parser.add_argument('--boosted', action='store_true', help='Flag to include boosted objects', default=False)
	parser.add_argument('--year', action='store', choices=['2016', '2017', '2018'], help='Year of data/MC samples', default='2017')
	parser.add_argument('--parameters', nargs='+', help='change default parameters, syntax: name value, eg --parameters met 40 bbtagging_algorithm btagDDBvL', default=None)
	#parser.add_argument('--corrections', action='store_true', help='Flag to include corrections')
	#parser.add_argument('filenames', nargs=argparse.REMAINDER)
	parser.add_argument('--machine', action='store', choices=['lxplus', 't3'], help="Machine: 'lxplus' or 't3'", default='lxplus', required=True)
	parser.add_argument('--workers', action='store', help='Number of workers (CPUs) to use', type=int, default=10)
	parser.add_argument('--chunksize', action='store', help='Number of events in a single chunk', type=int, default=30000)
	parser.add_argument('--maxchunks', action='store', help='Maximum number of chunks', type=int, default=25)
	args = parser.parse_args()

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

	f1 = open("datasets/RunIIFall17NanoAODv7PostProc/ttHTobb_2017.txt", 'r')
	f2 = open("datasets/RunIIFall17NanoAODv7PostProc/TTTo2L2Nu_2017.txt", 'r')
	samples = { "ttHTobb": f1.read().splitlines(), "TTTo2L2Nu": f2.read().splitlines() }
	f1.close()
	f2.close()
	if args.machine == 't3':
		for sample in samples:
			for (i, file) in enumerate(samples[sample]):
				samples[sample][i] = file.replace('root://xrootd-cms.infn.it/', '/pnfs/psi.ch/cms/trivcat')

	"""
	samples = {
		"ttHTobb": [
			"/afs/cern.ch/work/m/mmarcheg/Coffea/test/nano_postprocessed_24_ttHbb.root",
		],
		"TTToSemiLeptonic": [
			"/afs/cern.ch/work/m/mmarcheg/Coffea/test/nano_postprocessed_45_tt_semileptonic.root"
		]
	}
	"""

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
