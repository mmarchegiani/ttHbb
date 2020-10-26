import argparse
import os
import sys

from coffea import processor, hist
from coffea.analysis_objects import JaggedCandidateArray
import matplotlib.pyplot as plt
import numpy as np

from lib_analysis import lepton_selection, jet_selection
from definitions_analysis import parameters, histogram_settings

class ttHbb(processor.ProcessorABC):
	def __init__(self):
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
		if is_mc:
			genparts = events.GenPart

		if args.year =='2017':
			metstruct = 'METFixEE2017'
			MET = events.METFixEE2017
		else:
			metstruct = 'MET'
			MET = events.MET

		print("MET choice: %s" % metstruct)

		muons.p4 = JaggedCandidateArray.candidatesfromcounts(muons.counts, pt=muons.pt.content, eta=muons.eta.content, phi=muons.phi.content, mass=muons.mass.content)
		jets.p4 = JaggedCandidateArray.candidatesfromcounts(jets.counts, pt=jets.pt, eta=jets.eta, phi=jets.phi, mass=jets.mass)
		METp4 = JaggedCandidateArray.candidatesfromcounts(np.ones_like(MET.pt), pt=MET.pt, eta=np.zeros_like(MET.pt), phi=MET.phi, mass=np.zeros_like(MET.pt))
		#METp4 = JaggedCandidateArray.candidatesfromcounts(np.ones_like(MET), pt=scalars[metstruct+"_pt"], eta=np.zeros_like(MET), phi=scalars[metstruct+"_phi"], mass=np.zeros_like(MET))
		
		"""
		for obj in [muons, electrons, jets, fatjets, PuppiMET, MET]:
			obj.masks = {}
			obj.masks['all'] = np.ones_like(obj.flatten(), dtype=np.bool)
		"""
		indices = {
		"leading"    : np.zeros(nEvents, dtype=np.int32),
		"subleading" : np.ones(nEvents, dtype=np.int32)
		}

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
	#    good_jets = jet_selection(jets, muons, (veto_muons | good_muons), parameters["jets"]) & jet_selection(jets, electrons, (veto_electrons | good_electrons) , parameters["jets"])
		#bjets_resolved = good_jets & (getattr(jets, parameters["btagging_algorithm"]) > parameters["btagging_WP"])
		#good_fatjets = jet_selection(fatjets, muons, good_muons, parameters["fatjets"]) & jet_selection(fatjets, electrons, good_electrons, parameters["fatjets"])
	#    good_fatjets = jet_selection(fatjets, muons, (veto_muons | good_muons), parameters["fatjets"]) & jet_selection(fatjets, electrons, (veto_electrons | good_electrons), parameters["fatjets"]) #FIXME remove vet_leptons

	#    higgs_candidates = good_fatjets & (fatjets.pt > 250)
	#    nhiggs = ha.sum_in_offsets(fatjets, higgs_candidates, mask_events, fatjets.masks["all"], NUMPY_LIB.int8)
	#    indices["best_higgs_candidate"] = ha.index_in_offsets(fatjets.pt, fatjets.offsets, 1, mask_events, higgs_candidates)
	#    best_higgs_candidate = NUMPY_LIB.zeros_like(higgs_candidates)
	#    best_higgs_candidate[ (fatjets.offsets[:-1] + indices["best_higgs_candidate"])[NUMPY_LIB.where( fatjets.offsets<len(best_higgs_candidate) )] ] = True
	#    best_higgs_candidate[ (fatjets.offsets[:-1] + indices["best_higgs_candidate"])[NUMPY_LIB.where( fatjets.offsets<len(best_higgs_candidate) )] ] &= nhiggs.astype(NUMPY_LIB.bool)[NUMPY_LIB.where( fatjets.offsets<len(best_higgs_candidate) )] # to avoid removing the leading fatjet in events with no higgs candidate

		######################################################

		cut = (muons.counts == 1) & (jets.counts >= 2)
		cut_goodmuons = cut & good_muons
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
	#parser.add_argument('--sample', action='store', help='sample name', type=str, default=None, required=True)
	#parser.add_argument('--categories', nargs='+', help='categories to be processed (default: sl_jge4_tge2)', default="sl_jge4_tge2")
	#parser.add_argument('--boosted', action='store_true', help='Flag to include boosted objects', default=False)
	parser.add_argument('--year', action='store', choices=['2016', '2017', '2018'], help='Year of data/MC samples', default='2017')
	#parser.add_argument('--parameters', nargs='+', help='change default parameters, syntax: name value, eg --parameters met 40 bbtagging_algorithm btagDDBvL', default=None)
	#parser.add_argument('--corrections', action='store_true', help='Flag to include corrections')
	#parser.add_argument('filenames', nargs=argparse.REMAINDER)
	args = parser.parse_args()

	samples = {
		'ttHbb': [
			"root://xrootd-cms.infn.it//store/user/algomez/tmpFiles/ttH/ttHTobb_M125_TuneCP5_PSweights_13TeV-powheg-pythia8/ttHTobb_nanoAODPostProcessor_2017_v03/201009_121211/0000/nano_postprocessed_18.root",
		],
		'tt semileptonic': [
			"root://xrootd-cms.infn.it//store/user/algomez/tmpFiles/ttH/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/TTToSemiLeptonic_nanoAODPostProcessor_2017_v03/200903_113849/0000/nano_postprocessed_97.root"
		]
	}

	print("Running uproot job...")
	result = processor.run_uproot_job(
		samples,
		"Events",
		ttHbb(),
		processor.futures_executor,
		{"nano": True, "workers": 10},
	    chunksize=30000,
	    maxchunks=6,
	)

	plot_dir = "plots/"
	if not os.path.exists(plot_dir):
		os.makedirs(plot_dir)
	ax = hist.plot1d(result['mass'], overlay='dataset')
	ax.figure.savefig(plot_dir + "mass_w_visible.png", format="png")
	plt.close(ax.figure)
	ax = hist.plot1d(result['muons'].sum('eta'), overlay='dataset')
	ax.figure.savefig(plot_dir + "pt_muons.png", format="png")
	plt.close(ax.figure)
	ax = hist.plot1d(result['muons'].sum('pt'), overlay='dataset')
	ax.figure.savefig(plot_dir + "eta_muons.png", format="png")
	plt.close(ax.figure)
	ax = hist.plot1d(result['good_muons'].sum('eta'), overlay='dataset')
	ax.figure.savefig(plot_dir + "pt_goodmuona.png", format="png")
	plt.close(ax.figure)
	ax = hist.plot1d(result['good_muons'].sum('pt'), overlay='dataset')
	ax.figure.savefig(plot_dir + "eta_goodmuons.png", format="png")
	plt.close(ax.figure)
	
