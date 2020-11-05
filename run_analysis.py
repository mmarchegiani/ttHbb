import argparse
import os
import sys

#import awkward1 as ak
import awkward1
from coffea import processor, hist
from coffea.analysis_objects import JaggedCandidateArray
import matplotlib.pyplot as plt
import numpy as np

from lib_analysis import lepton_selection, jet_selection, jet_nohiggs_selection, get_leading_value, calc_dr
from definitions_analysis import parameters, histogram_settings

class ttHbb(processor.ProcessorABC):
	def __init__(self):
		#self.sample = sample
		self._accumulator = processor.dict_accumulator({
			"sumw": processor.defaultdict_accumulator(float),
			"mass": hist.Hist(
				"entries",
				hist.Cat("dataset", "Dataset"),
				hist.Bin("mw_vis", "$M^{vis}_W$ [GeV]", np.linspace(*histogram_settings['lepWMass'])),
			),
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
				hist.Bin("pt", "$p^{T}_{H}$ [GeV]", np.linspace(*histogram_settings['leadAK8JetPt'])),
				hist.Bin("deltaR", "$\Delta R_{H,\ell}$", np.linspace(*histogram_settings['deltaRHiggsLepton'])),
			),
		})

	@property
	def accumulator(self):
		return self._accumulator

	def process(self, events, parameters=parameters, samples_info={}, is_mc=True, lumimask=None, cat=False, boosted=False, uncertainty=None, uncertaintyName=None, parametersName=None, extraCorrection=None):
		output = self.accumulator.identity()
		dataset=events.metadata["dataset"]
		nEvents = events.event.size
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
		good_events           = events[mask_events]
		
		#print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		#print(muons[good_muons].pt)
		#print(good_events.GoodMuon.pt)
		#print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

		# calculate basic variables
		leading_jet_pt        = awkward1.firsts(good_events.GoodJet.pt)
		leading_jet_eta       = awkward1.firsts(good_events.GoodJet.eta)
		leading_fatjet_SDmass = awkward1.firsts(good_events.GoodFatJet.msoftdrop)
		leading_fatjet_pt     = awkward1.firsts(good_events.GoodFatJet.pt)
		leading_fatjet_eta    = awkward1.firsts(good_events.GoodFatJet.eta)
		leading_fatjet_phi    = awkward1.firsts(good_events.GoodFatJet.phi)
		leading_fatjet_mass   = awkward1.firsts(good_events.GoodFatJet.mass)
		leading_lepton_pt     = get_leading_value(good_events.GoodMuon.pt, good_events.GoodElectron.pt)
		leading_lepton_eta    = get_leading_value(good_events.GoodMuon.eta, good_events.GoodElectron.eta)
		leading_lepton_phi    = get_leading_value(good_events.GoodMuon.phi, good_events.GoodElectron.phi)
		leading_lepton_mass   = get_leading_value(good_events.GoodMuon.mass,good_events.GoodElectron.mass)
		leading_fatjet_rho    = awkward1.from_iter( np.log(leading_fatjet_SDmass**2 / leading_fatjet_pt**2) )

		mask_events_withFatJet= good_events.GoodFatJet.counts > 0
		good_events_withFatJet= good_events[mask_events_withFatJet]
		leading_lepton_p4     = JaggedCandidateArray.candidatesfromcounts(good_events_withFatJet.GoodElectron.counts + good_events_withFatJet.GoodMuon.counts, pt=leading_lepton_pt[mask_events_withFatJet], eta=leading_lepton_eta[mask_events_withFatJet], phi=leading_lepton_phi[mask_events_withFatJet], mass=leading_lepton_mass[mask_events_withFatJet])
		leading_fatjet_p4     = JaggedCandidateArray.candidatesfromcounts(good_events_withFatJet.GoodFatJet.counts, pt=leading_fatjet_pt[mask_events_withFatJet], eta=leading_fatjet_eta[mask_events_withFatJet], phi=leading_fatjet_phi[mask_events_withFatJet], mass=leading_fatjet_mass[mask_events_withFatJet])

		#deltaRHiggsLepton     = good_events.GoodElectron.delta_r(good_events.GoodFatJet)
		#deltaRHiggsMuon       = good_events_withFatjet.GoodFatJet.delta_r(good_events.Muon)
		#deltaRHiggsElectron   = good_events_withFatjet.GoodFatJet.delta_r(good_events.Electron)
		deltaRHiggsLepton     = calc_dr(leading_lepton_p4, leading_fatjet_p4)

		#print(deltaRHiggsMuon)
		#print(deltaRHiggsElectron)
		print("dr: ", deltaRHiggsLepton)

		######################################################

		cut = (muons.counts == 1) & (jets.counts >= 2)
		cut_goodmuons = cut & good_muons
		cut_goodjets = cut & good_jets
		#selected_events = events[cut]
		#candidate_w = muons.p4[cut].cross(METp4[cut])
		candidate_w = (muons.p4[cut]).cross(METp4[cut])
				
		output["sumw"][dataset] += nEvents
		output["mass"].fill(
			dataset=dataset,
			mw_vis=candidate_w.mass.flatten(),
		)
		output["muons"].fill(
			dataset=dataset,
			pt=muons[cut].pt.flatten(),
			eta=muons[cut].eta.flatten(),
		)
		output["good_muons"].fill(
			dataset=dataset,
			pt=muons[cut_goodmuons].pt.flatten(),
			eta=muons[cut_goodmuons].eta.flatten(),
		)
		output["jets"].fill(
			dataset=dataset,
			pt=jets[cut].pt.flatten(),
			eta=jets[cut].eta.flatten(),
		)
		output["good_jets"].fill(
			dataset=dataset,
			pt=jets[cut_goodjets].pt.flatten(),
			eta=jets[cut_goodjets].eta.flatten(),
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
			pt=leading_lepton_pt,
			eta=leading_lepton_eta,
		)
		output["higgs"].fill(
			dataset=dataset,
			pt=leading_fatjet_pt,
			deltaR=deltaRHiggsLepton,
		)

		return output

	def postprocess(self, accumulator):
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

	"""
	if "Single" in args.sample:
		is_mc = False
		lumimask = LumiMask(parameters["lumimask"])
	else:
		is_mc = True
		lumimask = None
	"""

	"""
	samples = {
		'ttHbb': [
			"root://xrootd-cms.infn.it//store/user/algomez/tmpFiles/ttH/ttHTobb_M125_TuneCP5_PSweights_13TeV-powheg-pythia8/ttHTobb_nanoAODPostProcessor_2017_v03/201009_121211/0000/nano_postprocessed_18.root",
		],
		'tt semileptonic': [
			"root://xrootd-cms.infn.it//store/user/algomez/tmpFiles/ttH/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic_nanoAODPostProcessor_2017_v03/200903_113849/0000/nano_postprocessed_97.root"
		]
	}
	"""

	samples = {
		'ttHbb': [
			"/afs/cern.ch/work/m/mmarcheg/Coffea/test/nano_postprocessed_18_ttHbb.root",
		],
		'tt semileptonic': [
			"/afs/cern.ch/work/m/mmarcheg/Coffea/test/nano_postprocessed_97_tt_semileptonic.root"
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
		{"nano": True, "workers": 10},
		chunksize=30000,
		maxchunks=6,
	)

	plot_dir = "plots/"
	histos = ["muons_pt.png", "muons_eta.png", "goodmuons_pt.png", "goodmuons_eta.png", "jets_pt.png", "jets_eta.png", "goodjets_pt.png", "goodjets_eta.png",
				 "njets.png", "ngoodjets.png", "nnonbjets.png", "leptons_pt.png","leptons_eta.png"]
	histo_names = ['muons', 'muons', 'good_muons', 'good_muons', 'jets', 'jets', 'good_jets', 'good_jets', 'njets', 'njets', 'njets', 'leptons', 'leptons', 'higgs', 'higgs']
	integrateover = ['eta', 'pt', 'eta', 'pt', 'eta', 'pt', 'eta', 'pt', ['ngoodjets', 'nnonbjets'], ['njets', 'nnonbjets'], ['njets', 'ngoodjets'], 'eta', 'pt', 'pt', 'deltaRHiggsLepton']
	#integrateover = ['eta', 'pt', 'eta', 'pt', 'eta', 'pt', 'eta', 'pt', ('ngoodjets', 'ngoodjets_nohiggs')]
	if not os.path.exists(plot_dir):
		os.makedirs(plot_dir)
	for (i, histo) in enumerate(histos):
		if histo in ["njets.png", "ngoodjets.png", "nnonbjets.png"]:
			ax = hist.plot1d(result[histo_names[i]].sum(*integrateover[i]), overlay='dataset')
		else:
			ax = hist.plot1d(result[histo_names[i]].sum(integrateover[i]), overlay='dataset')
		ax.figure.savefig(plot_dir + histo, format="png")
		plt.close(ax.figure)

	ax = hist.plot1d(result['mass'], overlay='dataset')
	ax.figure.savefig(plot_dir + "w_visible_mass.png", format="png")
	plt.close(ax.figure)