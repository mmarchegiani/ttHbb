import argparse

from coffea import hist, processor
from coffea.analysis_objects import JaggedCandidateArray
import numpy as np

class ttHbb(processor.ProcessorABC):
	def __init__(self):
		self._accumulator = processor.dict_accumulator({
			"sumw": processor.defaultdict_accumulator(float),
			"mass": hist.Hist(
				"entries",
				hist.Cat("dataset", "Dataset"),
				hist.Bin("mw_vis", "$M^{vis}_W$ [GeV]", 41, 0, 400),
			),
			"pt": hist.Hist(
				"entries",
				hist.Cat("dataset", "Dataset"),
				hist.Bin("pt_muon", "$p^{T}_{\mu}$ [GeV]", 41, 0, 400),
			),
		})

	@property
	def accumulator(self):
		return self._accumulator

	def process(self, events, parameters={}, samples_info={}, is_mc=True, lumimask=None, cat=False, boosted=False, uncertainty=None, uncertaintyName=None, parametersName=None, extraCorrection=None):
		output = self.accumulator.identity()
		dataset=events.metadata["dataset"]

		muons = events.Muon
		electrons = events.Electron
		#scalars = events.eventvars
		jets = events.Jet
		fatjets = events.FatJet
		MET = events.MET
		PuppiMET = events.PuppiMET
		if is_mc:
			genparts = events.GenPart

		"""
		if args.year=='2017':
			metstruct = 'METFixEE2017'
		else:
			metstruct = 'MET'
		"""

		muons.p4 = JaggedCandidateArray.candidatesfromcounts(muons.counts, pt=muons.pt, eta=muons.eta, phi=muons.phi, mass=muons.mass)
		jets.p4 = JaggedCandidateArray.candidatesfromcounts(jets.counts, pt=jets.pt, eta=jets.eta, phi=jets.phi, mass=jets.mass)
		METp4 = JaggedCandidateArray.candidatesfromcounts(np.ones_like(MET.pt), pt=MET.pt, eta=np.zeros_like(MET.pt), phi=MET.phi, mass=np.zeros_like(MET.pt))
		#METp4 = JaggedCandidateArray.candidatesfromcounts(np.ones_like(MET), pt=scalars[metstruct+"_pt"], eta=np.zeros_like(MET), phi=scalars[metstruct+"_phi"], mass=np.zeros_like(MET))
		nEvents = len(events.event)
	
		cut = (muons.counts == 1) & (jets.counts >= 2)
		#selected_events = events[cut]
		candidate_w = muons.p4[cut].cross(METp4[cut])
				
		output["sumw"][dataset] += nEvents
		output["mass"].fill(
			dataset=dataset,
			mw_vis=candidate_w.mass.flatten(),
		)
		output["pt"].fill(
			dataset=dataset,
			pt_muon=muons[cut].pt.flatten()
		)

		return output

	def postprocess(self, accumulator):
		return accumulator
