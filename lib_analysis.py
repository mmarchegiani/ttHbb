import argparse

import awkward1 as ak
import numpy as np
import math
import uproot

#from awkward.array.jagged import JaggedArray
from coffea import hist
#from coffea.analysis_objects import JaggedCandidateArray

def lepton_selection(leps, cuts, year):

	passes_eta = (np.abs(leps.eta) < cuts["eta"])
	passes_subleading_pt = (leps.pt > cuts["subleading_pt"])
	passes_leading_pt = (leps.pt > cuts["leading_pt"][year])

	if cuts["type"] == "el":
		sca = np.abs(leps.deltaEtaSC + leps.eta)
		passes_id = (leps.cutBased >= 4)
		passes_SC = np.invert((sca >= 1.4442) & (sca <= 1.5660))
		# cuts taken from: https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2#Working_points_for_92X_and_later
		passes_impact = ((leps.dz < 0.10) & (sca <= 1.479)) | ((leps.dz < 0.20) & (sca > 1.479)) | ((leps.dxy < 0.05) & (sca <= 1.479)) | ((leps.dxy < 0.1) & (sca > 1.479))

		#select electrons
		good_leps = passes_eta & passes_leading_pt & passes_id & passes_SC & passes_impact
		veto_leps = passes_eta & passes_subleading_pt & np.invert(good_leps) & passes_id & passes_SC & passes_impact

	elif cuts["type"] == "mu":
		passes_leading_iso = (leps.pfRelIso04_all < cuts["leading_iso"])
		passes_subleading_iso = (leps.pfRelIso04_all < cuts["subleading_iso"])
		passes_id = (leps.tightId == 1)

		#select muons
		good_leps = passes_eta & passes_leading_pt & passes_leading_iso & passes_id
		veto_leps = passes_eta & passes_subleading_pt & passes_subleading_iso & passes_id & np.invert(good_leps)

	return good_leps, veto_leps

def calc_dr2(pairs):
	
	deta = pairs.i0.eta - pairs.i1.eta
	dphi = pairs.i0.phi - pairs.i1.phi
	
	return deta**2 + dphi**2

"""
def calc_dr(objects1, objects2):

	pairs = objects1.cross(objects2)

	return ak.from_iter(np.sqrt(calc_dr2(pairs)))
"""

def pass_dr(pairs, dr):

	return calc_dr2(pairs) > dr**2

def jet_selection(jets, leps, mask_leps, cuts):

	nested_mask = jets.p4.match(leps.p4[mask_leps], matchfunc=pass_dr, dr=cuts["dr"])
	# Only jets that are more distant than dr to ALL leptons are tagged as good jets
	jets_pass_dr = nested_mask.all()
	good_jets = (jets.pt > cuts["pt"]) & (np.abs(jets.eta) < cuts["eta"]) & (jets.jetId >= cuts["jetId"]) & jets_pass_dr
	
	if cuts["type"] == "jet":
		good_jets = good_jets & ( (jets.pt < 50) & (jets.puId >= cuts["puId"]) ) | (jets.pt >= 50)

	return good_jets

def jet_nohiggs_selection(jets, fatjets, mask_fatjets, dr=1.2):
	
	nested_mask = jets.p4.match(fatjets.p4[mask_fatjets,0], matchfunc=pass_dr, dr=dr)
	jets_pass_dr = nested_mask.all()

	return jets_pass_dr

def get_leading_value(var1, var2):

	firsts1 = ak.firsts(var1)
	firsts2 = ak.firsts(var2)

	return ak.where(ak.is_none(firsts1), firsts2, firsts1)

def load_puhist_target(filename):
	fi = uproot.open(filename)

	h = fi["pileup"]
	edges = np.array(h.edges)
	values_nominal = np.array(h.values)
	values_nominal = values_nominal / np.sum(values_nominal)

	h = fi["pileup_plus"]
	values_up = np.array(h.values)
	values_up = values_up / np.sum(values_up)

	h = fi["pileup_minus"]
	values_down = np.array(h.values)
	values_down = values_down / np.sum(values_down)
	return edges, (values_nominal, values_up, values_down)

def remove_inf_nan(arr):
    arr[np.isinf(arr)] = 0
    arr[np.isnan(arr)] = 0
    arr[arr < 0] = 0

def compute_pu_weights(pu_corrections_target, weights, mc_nvtx, reco_nvtx):

	pu_edges, (values_nom, values_up, values_down) = pu_corrections_target

	src_pu_hist = hist.Hist("Pileup", coffea.hist.Bin("pu", "pu", pu_edges))
	src_pu_hist.fill(pu=mc_nvtx, weight=weights)
	norm = sum(src_pu_hist.values)
	histo.scale(1./norm)
	#src_pu_hist.contents = src_pu_hist.contents/norm
	#src_pu_hist.contents_w2 = src_pu_hist.contents_w2/norm

	ratio = values_nom / src_pu_hist.values
#    ratio = values_nom / mc_values
	remove_inf_nan(ratio)
	#pu_weights = np.zeros_like(weights)
	pu_weights = np.take(ratio, np.digitize(reco_nvtx, np.array(pu_edges)) - 1)

	return pu_weights
