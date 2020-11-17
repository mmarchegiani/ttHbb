import argparse
import os
import sys

#import awkward1 as ak
import awkward1
import matplotlib.pyplot as plt
import numpy as np

from coffea import processor, hist
from coffea.analysis_objects import JaggedCandidateArray
from coffea.lumi_tools import LumiMask, LumiData
from coffea.lookup_tools import extractor
from coffea.btag_tools import BTagScaleFactor
from uproot_methods import TLorentzVectorArray

from lib_analysis import lepton_selection, jet_selection, jet_nohiggs_selection, get_leading_value, load_puhist_target, compute_lepton_weights, METzCalculator, hadronic_W, calc_dr
from definitions_analysis import parameters, histogram_settings, samples_info

class ttHbb(processor.ProcessorABC):
	def __init__(self):
		#self.sample = sample
		self.var_names = [
		  'nleps'            ,
		  'njets'            ,
		  'ngoodjets'        ,
		  'btags'            ,
		  'btags_resolved'   ,
		  'nfatjets'         ,
		  'met'              ,
		  'leading_jet_pt'   ,
		  'leading_jet_eta'  ,
		  'leadAK8JetMass'   ,
		  'leadAK8JetPt'     ,
		  'leadAK8JetEta'    ,
		  'leadAK8JetHbb'    ,
		  'leadAK8JetTau21'  ,
		  'leadAK8JetRho'    ,
		  'lepton_pt'        ,
		  'lepton_eta'       ,
		  'hadWPt'           ,
		  'hadWEta'          ,
		  'hadWMass'         ,
		  'lepWPt'           ,
		  'lepWEta'          ,
		  'lepWMass'         ,
		  'deltaRlepWHiggs'  ,
		  'deltaRhadWHiggs'  ,
		  'deltaRHiggsLepton',
		  #'PV_npvsGood',
		  'weights_ones',
		  'weights_nominal',
		  'weights_pu',
		  'weights_lepton',
		]
		self.mask_events_list = [
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
			"muons": hist.Hist(
				"entries",
				hist.Cat("dataset", "Dataset"),
				hist.Bin("pt", "$p^{T}_{\mu}$ [GeV]", np.linspace(*histogram_settings['lepton_pt'])),
				hist.Bin("eta", "$\eta_{\mu}$", np.linspace(*histogram_settings['lepton_eta'])),
			),
			"good_muons": hist.Hist(
				"entries",
				hist.Cat("dataset", "Dataset"),
				hist.Bin("pt", "$p^{T}_{\mu}$ [GeV]", np.linspace(*histogram_settings['lepton_pt'])),
				hist.Bin("eta", "$\eta_{\mu}$", np.linspace(*histogram_settings['lepton_eta'])),
			),
			"jets": hist.Hist(
				"entries",
				hist.Cat("dataset", "Dataset"),
				hist.Bin("pt", "$p^{T}_{\mu}$ [GeV]", np.linspace(*histogram_settings['leading_jet_pt'])),
				hist.Bin("eta", "$\eta_{\mu}$", np.linspace(*histogram_settings['leading_jet_eta'])),
			),
			"good_jets": hist.Hist(
				"entries",
				hist.Cat("dataset", "Dataset"),
				hist.Bin("pt", "$p^{T}_{\mu}$ [GeV]", np.linspace(*histogram_settings['leading_jet_pt'])),
				hist.Bin("eta", "$\eta_{\mu}$", np.linspace(*histogram_settings['leading_jet_eta'])),
			),
			"njets": hist.Hist(
				"entries",
				hist.Cat("dataset", "Dataset"),
				hist.Bin("njets", "$N_{jets}$", np.linspace(*histogram_settings['njets'])),
				hist.Bin("ngoodjets", "$N_{good_jets}$", np.linspace(*histogram_settings['ngoodjets'])),
				#hist.Bin("ngoodjets_nohiggs", "$N_{nohiggs}$", np.linspace(*histogram_settings['ngoodjets'])),
				hist.Bin("nnonbjets", "$N_{nonbjets}$", np.linspace(*histogram_settings['ngoodjets'])),
			),
			"leptons": hist.Hist(
				"entries",
				hist.Cat("dataset", "Dataset"),
				hist.Bin("pt", "$p^{T}_{\ell}$ [GeV]", np.linspace(*histogram_settings['lepton_pt'])),
				hist.Bin("eta", "$\eta_{\ell}$", np.linspace(*histogram_settings['lepton_eta'])),
			),
			"higgs": hist.Hist(
				"entries",
				hist.Cat("dataset", "Dataset"),
				#hist.Bin("pt", "$p^{T}_{H}$ [GeV]", np.linspace(*histogram_settings['leadAK8JetPt'])),
				hist.Bin("deltaR", "$\Delta R_{H,\ell}$", np.linspace(0,15,91)),
			),
			"higgs_mass": hist.Hist(
				"entries",
				hist.Cat("dataset", "Dataset"),
				hist.Bin("mass", "$M_{H}$ [GeV]", np.linspace(*histogram_settings['leadAK8JetMass'])),
				hist.Bin("rho", "${\{rho}}_{H} $", np.linspace(*histogram_settings['leadAK8JetRho'])),
			),
		})

		vars_split = ['leadAK8JetMass', 'leadAK8JetRho']
		ptbins = np.append( np.arange(250,600,50), [600, 1000, 5000] )
		for var_name in vars_split:
			#var = vars_to_plot[var_name]
			for ipt in range( len(ptbins)-1 ):
				for m in ['2J2WdeltaR']:#, '2J2WdeltaRTau21']:#, '2J2WdeltaRTau21DDT']:
					for r in ['Pass','Fail']:
						for o in ['','_orthogonal']:
							mask_name = f'{m}_{r}{o}'
							if not mask_name in self.mask_events_list: continue
							hist_name = f'hist_{var_name}_{mask_name}_pt{ptbins[ipt]}to{ptbins[ipt+1]}'
							self._accumulator.add(processor.dict_accumulator({hist_name : hist.Hist("entries",
														   					  hist.Cat("dataset", "Dataset"),
														   					  hist.Bin("values", var_name, np.linspace( *histogram_settings[var_name] ) ) ) } ) )

		#for wn,w in weights.items():
		for wn in ['nominal']:
			if wn != 'nominal': continue
			#ret[f'nevts_overlap{weight_name}'] = Histogram( [sum(weights[w]), sum(weights[w][mask_events['2J2WdeltaR']]), sum(weights[w][mask_events['resolved']]), sum(weights[w][mask_events['overlap']])], 0,0 )
			for mask_name in self.mask_events_list:
				if not 'deltaR' in mask_name: continue
				for var_name in self.var_names:
					#if (not is_mc) and ('Pass' in mask_name) and (var_name=='leadAK8JetMass') : continue
					try:
						hist_name = f'hist_{var_name}_{mask_name}_weights_{wn}'
						self._accumulator.add(processor.dict_accumulator({hist_name : hist.Hist("entries",
																 		  hist.Cat("dataset", "Dataset"),
																 		  hist.Bin("values", var_name, np.linspace( *histogram_settings[var_name if not var_name.startswith('weights') else 'weights'] ) ) ) } ) )
					except KeyError:
						print(f'!!!!!!!!!!!!!!!!!!!!!!!! Please add variable {var_name} to the histogram settings')

	@property
	def accumulator(self):
		return self._accumulator

	def process(self, events, parameters=parameters, samples_info=samples_info, is_mc=True, lumimask=None, cat=False, boosted=False, uncertainty=None, uncertaintyName=None, parametersName=None, extraCorrection=None):
		output = self.accumulator.identity()
		dataset = events.metadata["dataset"]
		nEvents = events.event.size
		sample = dataset
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
		MET.p4 = JaggedCandidateArray.candidatesfromcounts(np.ones_like(MET.pt), pt=MET.pt, eta=np.zeros_like(MET.pt), phi=MET.phi, mass=np.zeros_like(MET.pt))
		
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
#	    good_jets = jet_selection(jets, muons, (veto_muons | good_muons), parameters["jets"]) & jet_selection(jets, electrons, (veto_electrons | good_electrons) , parameters["jets"])
		bjets_resolved = good_jets & (getattr(jets, parameters["btagging_algorithm"]) > parameters["btagging_WP"])
		good_fatjets = jet_selection(fatjets, muons, good_muons, parameters["fatjets"]) & jet_selection(fatjets, electrons, good_electrons, parameters["fatjets"])
#	    good_fatjets = jet_selection(fatjets, muons, (veto_muons | good_muons), parameters["fatjets"]) & jet_selection(fatjets, electrons, (veto_electrons | good_electrons), parameters["fatjets"]) #FIXME remove vet_leptons

		good_jets_nohiggs = good_jets & jet_nohiggs_selection(jets, fatjets, good_fatjets, 1.2)
		bjets = good_jets_nohiggs & (getattr(jets, parameters["btagging_algorithm"]) > parameters["btagging_WP"])
		nonbjets = good_jets_nohiggs & (getattr(jets, parameters["btagging_algorithm"]) < parameters["btagging_WP"])

		# apply basic event selection -> individual categories cut later
		nmuons         = muons[mask_events,:][good_muons].counts
		nelectrons     = electrons[mask_events,:][good_electrons].counts
		nleps          = nmuons + nelectrons
		lepton_veto    = muons[mask_events,:][veto_muons].counts + electrons[mask_events,:][veto_electrons].counts
		njets_raw      = jets[mask_events,:].counts
		njets          = jets[mask_events,:][nonbjets].counts
		ngoodjets      = jets[mask_events,:][good_jets].counts
		btags          = jets[mask_events,:][bjets].counts
		btags_resolved = jets[mask_events,:][bjets_resolved].counts
		nfatjets       = fatjets[mask_events,:][good_fatjets].counts
		#nhiggs         = fatjets[mask_events,:][higgs_candidates].counts

		# trigger logic
		trigger_el = (nleps==1) & (nelectrons==1)
		trigger_mu = (nleps==1) & (nmuons==1)
		if args.year.startswith('2016'):
			trigger_el = trigger_el & HLT.Ele27_WPTight_Gsf
			trigger_mu = trigger_mu & (HLT.IsoMu24 | HLT.IsoTkMu24)
		elif args.year.startswith('2017'):
			#trigger = (HLT.Ele35_WPTight_Gsf | HLT.Ele28_eta2p1_WPTight_Gsf_HT150 | HLT.IsoMu27 | HLT.IsoMu24_eta2p1) #FIXME for different runs
			if args.sample.endswith(('2017B','2017C')):
				trigger_tmp = HLT.Ele32_WPTight_Gsf_L1DoubleEG & any([getattr(HLT, 'L1_SingleEG{n}er2p5') for n in (10,15,26,34,36,38,40,42,45,8)])
			else:
				trigger_tmp = HLT.Ele32_WPTight_Gsf
			trigger_el = trigger_el & (trigger_tmp | HLT.Ele28_eta2p1_WPTight_Gsf_HT150)
			trigger_mu = trigger_mu & HLT.IsoMu27
		elif args.year.startswith('2018'):
			trigger = (HLT.Ele32_WPTight_Gsf | HLT.Ele28_eta2p1_WPTight_Gsf_HT150 | HLT.IsoMu24 )
			trigger_el = trigger_el & (HLT.Ele32_WPTight_Gsf | HLT.Ele28_eta2p1_WPTight_Gsf_HT150)
			trigger_mu = trigger_mu & HLT.IsoMu24
		if "SingleMuon" in args.sample: trigger_el = np.zeros(nEvents, dtype=np.bool)
		if "SingleElectron" in args.sample: trigger_mu = np.zeros(nEvents, dtype=np.bool)
		mask_events = mask_events & (trigger_el | trigger_mu)

		# for reference, this is the selection for the resolved analysis
		mask_events_res = mask_events & (nleps == 1) & (lepton_veto == 0) & (ngoodjets >= 4) & (btags_resolved > 2) & (MET.pt > 20)
		# apply basic event selection
		#mask_events_higgs = mask_events & (nleps == 1) & (MET.pt > 20) & (nhiggs > 0) & (njets > 1)  # & np.invert( (njets >= 4) & (btags >=2) ) & (lepton_veto == 0)
		mask_events_boost = mask_events & (nleps == 1) & (lepton_veto == 0) & (MET.pt > parameters['met']) & (nfatjets > 0) & (btags >= parameters['btags']) # & (btags_resolved < 3)# & (njets > 1)  # & np.invert( (njets >= 4)  )

		# select good objects
		mask_events           = mask_events_res | mask_events_boost
		events["GoodMuon"]    = muons[good_muons]
		events["GoodElectron"]= electrons[good_electrons]
		events["GoodJet"]     = jets[nonbjets]
		events["GoodFatJet"]  = fatjets[good_fatjets]

		# calculate basic variables
		leading_jet_pt         = get_leading_value(events.GoodJet.pt)
		leading_jet_eta        = get_leading_value(events.GoodJet.eta, default=-9.)
		leading_fatjet_SDmass  = get_leading_value(events.GoodFatJet.msoftdrop)
		leading_fatjet_pt      = get_leading_value(events.GoodFatJet.pt)
		leading_fatjet_eta     = get_leading_value(events.GoodFatJet.eta, default=-9.)
		leading_fatjet_phi     = get_leading_value(events.GoodFatJet.phi)
		leading_fatjet_mass    = get_leading_value(events.GoodFatJet.mass)
		leading_lepton_pt      = get_leading_value(events.GoodMuon.pt, events.GoodElectron.pt)
		leading_lepton_eta     = get_leading_value(events.GoodMuon.eta, events.GoodElectron.eta, default=-9.)
		leading_lepton_phi     = get_leading_value(events.GoodMuon.phi, events.GoodElectron.phi)
		leading_lepton_mass    = get_leading_value(events.GoodMuon.mass, events.GoodElectron.mass)
		#leading_lepton_pt      = get_leading_value(good_events.GoodMuon.pt, good_events.GoodElectron.pt)
		#leading_lepton_eta     = get_leading_value(good_events.GoodMuon.eta, good_events.GoodElectron.eta)
		#leading_lepton_phi     = get_leading_value(good_events.GoodMuon.phi, good_events.GoodElectron.phi)
		#leading_lepton_mass    = get_leading_value(good_events.GoodMuon.mass,good_events.GoodElectron.mass)
		leading_fatjet_rho     = awkward1.from_iter( np.log(leading_fatjet_SDmass**2 / leading_fatjet_pt**2) )

		import awkward
		events["LeadingLepton"] = awkward.Table(pt=leading_lepton_pt, eta=leading_lepton_eta, phi=leading_lepton_phi, mass=leading_lepton_mass)
		events["LeadingFatJet"] = awkward.Table(pt=leading_fatjet_pt, eta=leading_fatjet_eta, phi=leading_fatjet_phi, mass=leading_fatjet_mass, SDmass=leading_fatjet_SDmass, rho=leading_fatjet_rho)

		#good_events           = events[mask_events]
		mask_events_withFatJet = events.GoodFatJet.counts > 0
		mask_events_withLepton = nleps > 0
		leading_leptons = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_withLepton, dtype=int), pt=leading_lepton_pt[mask_events_withLepton], eta=leading_lepton_eta[mask_events_withLepton], phi=leading_lepton_phi[mask_events_withLepton], mass=leading_lepton_mass[mask_events_withLepton])
		higgs = JaggedCandidateArray.candidatesfromcounts(np.array(mask_events_withFatJet, dtype=int), pt=leading_fatjet_pt[mask_events_withFatJet], eta=leading_fatjet_eta[mask_events_withFatJet], phi=leading_fatjet_phi[mask_events_withFatJet], mass=leading_fatjet_mass[mask_events_withFatJet])
		deltaRHiggsLepton      = calc_dr(leading_leptons, higgs)
		#deltaRHiggsLepton      = awkward1.firsts(events.GoodFatJet[mask_events_withFatJet].delta_r(events.LeadingLepton[mask_events_withFatJet]))
		events.LeadingFatJet["deltaRHiggsLepton"] = deltaRHiggsLepton

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
			electron_weights = compute_lepton_weights(events.GoodElectron, evaluator, ["el_triggerSF", "el_recoSF", "el_idSF"], lepton_eta=(events.GoodElectron.deltaEtaSC + events.GoodElectron.eta))
			muon_weights = compute_lepton_weights(events.GoodMuon, evaluator, ["mu_triggerSF", "mu_isoSF", "mu_idSF"], year=args.year)
			weights['lepton']  = muon_weights * electron_weights
			weights["nominal"] = weights["nominal"] * weights['lepton']

		mask_events = {
		  'resolved' : mask_events_res,
		  'basic'    : mask_events_boost
		}
		mask_events['2J']   = mask_events['basic'] & (njets>1)

		#Ws reconstruction
		events.LeadingLepton["p4"] = JaggedCandidateArray.candidatesfromcounts(np.ones(nEvents), pt=events.LeadingLepton.pt, eta=events.LeadingLepton.eta, phi=events.LeadingLepton.phi, mass=events.LeadingLepton.mass)
		pznu = METzCalculator(events.LeadingLepton.p4.p4, MET.p4.p4, mask_events['2J'])
		#neutrino_p4 = JaggedCandidateArray.candidatesfromcounts(np.ones(nEvents), px=MET.p4.p4.x, py=MET.p4.p4.y, pz=pznu, energy=np.sqrt( MET.p4.p4.x**2 + MET.p4.p4.y**2 + pznu**2 ))
		neutrino_p4 = TLorentzVectorArray.from_cartesian(MET.p4.p4.x, MET.p4.p4.y, pznu, np.sqrt( MET.p4.p4.x**2 + MET.p4.p4.y**2 + pznu**2 ))
		leading_lepton_p4 = TLorentzVectorArray.from_ptetaphim(events.LeadingLepton.pt, events.LeadingLepton.eta, events.LeadingLepton.phi, events.LeadingLepton.mass)
		#lepW = neutrino_p4.cross(events.LeadingLepton["p4"])
		lepW = leading_lepton_p4 + neutrino_p4
		#lepW = JaggedCandidateArray.candidatesfromcounts(np.ones(nEvents), pt=lepW_vector.pt, eta=lepW_vector.eta, phi=lepW_vector.phi, mass=lepW_vector.mass)

		good_jets_p4 = JaggedCandidateArray.candidatesfromcounts(events.GoodJet.counts, pt=events.GoodJet.pt.content, eta=events.GoodJet.eta.content, phi=events.GoodJet.phi.content, mass=events.GoodJet.mass.content)

		hadW, n_hadW = hadronic_W(good_jets_p4, lepW)

		#print(awkward1.any(lepW.mass>parameters['W']['min_mass'], axis=1))
		#mask_events['2J2W'] = mask_events['2J'] & (hadW.mass>parameters['W']['min_mass']) & (hadW.mass<parameters['W']['max_mass']) & (lepW.mass>parameters['W']['min_mass']) & (lepW.mass<parameters['W']['max_mass'])
		mask_events['2J2W'] = mask_events['2J'] & awkward1.any(hadW.mass>parameters['W']['min_mass'], axis=1) & awkward1.any(hadW.mass<parameters['W']['max_mass'], axis=1) & awkward1.any(lepW.mass>parameters['W']['min_mass'], axis=1) & awkward1.any(lepW.mass<parameters['W']['max_mass'], axis=1)

		#deltaR between objects
		#deltaRHiggsLepton      = awkward1.firsts(events.GoodFatJet[mask_events_withFatJet].delta_r(events.LeadingLepton[mask_events_withFatJet]))
		lepWs = JaggedCandidateArray.candidatesfromcounts(np.array(nleps > 0, dtype=int), pt=lepW.pt.flatten(), eta=lepW.eta.flatten(), phi=lepW.phi.flatten(), mass=lepW.mass.flatten())
		hadWs = JaggedCandidateArray.candidatesfromcounts(n_hadW, pt=hadW.pt.flatten(), eta=hadW.eta.flatten(), phi=hadW.phi.flatten(), mass=hadW.mass.flatten())
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
							np.linspace(*histogram_settings[vn]),\
							np.linspace(*histogram_settings['btags_resolved']),\
							),\
							weights=weights["nominal"][m]\
							)
					ret[f'hist2d_{vn}VSbtags_{mn}'] = Histogram( hist, hist, (*histogram_settings[vn],*histogram_settings['btags_resolved']) )
		"""
		############# histograms
		vars_to_plot = {
		'nleps'             : nleps,
		'njets'             : njets,
		'ngoodjets'         : ngoodjets,
		'btags'             : btags,
		'btags_resolved'    : btags_resolved,
		'nfatjets'          : nfatjets,
		'met'               : MET.pt,
		'leading_jet_pt'    : leading_jet_pt,
		'leading_jet_eta'   : leading_jet_eta,
		'leadAK8JetMass'    : leading_fatjet_SDmass,
		'leadAK8JetPt'      : leading_fatjet_pt,
		'leadAK8JetEta'     : leading_fatjet_eta,
		'leadAK8JetHbb'     : leading_fatjet_Hbb,
		'leadAK8JetTau21'   : leading_fatjet_tau21,
		'leadAK8JetRho'     : leading_fatjet_rho,
		'lepton_pt'         : leading_lepton_pt,
		'lepton_eta'        : leading_lepton_eta,
		'hadWPt'            : get_leading_value(hadW.pt),
		'hadWEta'           : get_leading_value(hadW.eta),
		'hadWMass'          : get_leading_value(hadW.mass),
		'lepWPt'            : get_leading_value(lepW.pt),
		'lepWEta'           : get_leading_value(lepW.eta),
		'lepWMass'          : get_leading_value(lepW.mass),
		'deltaRlepWHiggs'   : deltaRlepWHiggs,
		'deltaRhadWHiggs'   : deltaRhadWHiggs,
		'deltaRHiggsLepton' : deltaRHiggsLepton,
		#'PV_npvsGood'       : scalars['PV_npvsGood'],
		}

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
		from coffea.util import _ensure_flat
		for wn,w in weights.items():
			for mask_name, mask in mask_events.items():
				if not 'deltaR' in mask_name: continue
				print(wn, "[", mask_name, "]: ", w[mask])
				array = _ensure_flat(w[mask])
		"""

######################################################

						
		output["muons"].fill(
			dataset=dataset,
			pt=muons.pt.flatten(),
			eta=muons.eta.flatten(),
		)
		output["good_muons"].fill(
			dataset=dataset,
			pt=events.GoodMuon.pt.flatten(),
			eta=events.GoodMuon.eta.flatten(),
		)
		output["jets"].fill(
			dataset=dataset,
			pt=jets.pt.flatten(),
			eta=jets.eta.flatten(),
		)
		output["good_jets"].fill(
			dataset=dataset,
			pt=events.GoodJet.pt.flatten(),
			eta=events.GoodJet.eta.flatten(),
		)
		output["njets"].fill(
			dataset=dataset,
			njets=njets_raw,
			ngoodjets=ngoodjets,
			#ngoodjets_nohiggs=jets[mask_events,:][good_jets_nohiggs].counts,
			nnonbjets=njets,
		)
		output["leptons"].fill(
			dataset=dataset,
			pt=events.LeadingLepton.pt.flatten(),
			eta=events.LeadingLepton.eta.flatten(),
		)
		output["higgs"].fill(
			dataset=dataset,
			#pt=events.LeadingFatJet.pt.flatten(),
			deltaR=events.LeadingFatJet.deltaRHiggsLepton.flatten(),
		)
		output["higgs_mass"].fill(
			dataset=dataset,
			mass=events.LeadingFatJet.SDmass.flatten(),
			rho=events.LeadingFatJet.rho.flatten()
		)

		return output

	def postprocess(self, accumulator):
		plot_dir = "plots/"
		histos = ["muons_pt.png", "muons_eta.png", "goodmuons_pt.png", "goodmuons_eta.png", "jets_pt.png", "jets_eta.png", "goodjets_pt.png", "goodjets_eta.png",
					 "njets.png", "ngoodjets.png", "nnonbjets.png", "leptons_pt.png","leptons_eta.png", "higgs_rho.png", "higgs_mass.png"]
		histo_names = ['muons', 'muons', 'good_muons', 'good_muons', 'jets', 'jets', 'good_jets', 'good_jets', 'njets', 'njets', 'njets', 'leptons', 'leptons', 'higgs_mass', 'higgs_mass']
		integrateover = ['eta', 'pt', 'eta', 'pt', 'eta', 'pt', 'eta', 'pt', ['ngoodjets', 'nnonbjets'], ['njets', 'nnonbjets'], ['njets', 'ngoodjets'], 'eta', 'pt', 'mass', 'rho']
		#integrateover = ['eta', 'pt', 'eta', 'pt', 'eta', 'pt', 'eta', 'pt', ('ngoodjets', 'ngoodjets_nohiggs')]
		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)
		for (i, histo) in enumerate(histos):
			if histo in ["njets.png", "ngoodjets.png", "nnonbjets.png"]:
				ax = hist.plot1d(accumulator[histo_names[i]].sum(*integrateover[i]), overlay='dataset')
			else:
				ax = hist.plot1d(accumulator[histo_names[i]].sum(integrateover[i]), overlay='dataset')
			ax.figure.savefig(plot_dir + histo, dpi=300, format="png")
			plt.close(ax.figure)

		ax = hist.plot1d(accumulator['higgs'], overlay='dataset')
		ax.figure.savefig(plot_dir + "deltaRHiggsLepton.png", dpi=300, format="png")
		plt.close(ax.figure)

		plot_dir = "plots/comparison/"
		if not os.path.exists(plot_dir):
			os.makedirs(plot_dir)
		print("Saving plots in " + plot_dir)
		#for histo in accumulator["hist_list_split"] + accumulator["hist_list"]:
		for histo in [item for item in accumulator.keys() if "hist" in item]:
			ax = hist.plot1d(accumulator[histo], overlay='dataset')
			ax.figure.savefig(plot_dir + histo + ".png", dpi=300, format="png")
			plt.close(ax.figure)
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
	args = parser.parse_args()

	from definitions_analysis import parameters, eraDependentParameters, samples_info
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
	f2 = open("datasets/RunIIFall17NanoAODv7PostProc/TTToSemiLeptonic_2017.txt", 'r')
	samples = { "ttHTobb": f1.read().splitlines(), "TTToSemiLeptonic": f2.read().splitlines() }
	f1.close()
	f2.close()

	"""
	samples = {
		"ttHTobb": [
			"/afs/cern.ch/work/m/mmarcheg/Coffea/test/nano_postprocessed_18_ttHbb.root",
		],
		"TTToSemiLeptonic": [
			"/afs/cern.ch/work/m/mmarcheg/Coffea/test/nano_postprocessed_97_tt_semileptonic.root"
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
		{"nano": True, "workers": 10},
		chunksize=30000,
		maxchunks=6,
	)
