import argparse

import awkward1 as ak
import numpy as np
import math
import uproot

#from awkward.array.jagged import JaggedArray
from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray

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

def get_leading_value(var1, var2=None, default=-999.9):

	default = ak.from_iter(len(var1)*[default])
	firsts1 = ak.firsts(var1)
	if type(var2) is type(None):
		#return ak.where(ak.is_none(firsts1), default, firsts1)
		return ak.from_iter(ak.where(ak.is_none(firsts1), default, firsts1))
	else:
		firsts2 = ak.firsts(var2)
		leading = ak.where(ak.is_none(firsts1), firsts2, firsts1)
		#return ak.where(ak.is_none(leading), default, leading)
		return ak.from_iter(ak.where(ak.is_none(leading), default, leading))

def calc_dr2(pairs):
	
	deta = pairs.i0.eta - pairs.i1.eta
	dphi = (pairs.i0.phi - pairs.i1.phi + np.pi) % (2*np.pi) - np.pi
	
	return deta**2 + dphi**2

def calc_dr(objects1, objects2):

	pairs = objects1.cross(objects2)

	return get_leading_value(ak.from_iter(np.sqrt(calc_dr2(pairs))))

def pass_dr(pairs, dr):

	return calc_dr2(pairs) > dr**2

def jet_selection(jets, leps, mask_leps, cuts):

	nested_mask = jets.p4.match(leps.p4[mask_leps], matchfunc=pass_dr, dr=cuts["dr"])
	# Only jets that are more distant than dr to ALL leptons are tagged as good jets
	jets_pass_dr = nested_mask.all()
	good_jets = (jets.pt > cuts["pt"]) & (np.abs(jets.eta) < cuts["eta"]) & (jets.jetId >= cuts["jetId"]) & jets_pass_dr
	
	if cuts["type"] == "jet":
		good_jets = good_jets & ( ( (jets.pt < 50) & (jets.puId >= cuts["puId"]) ) | (jets.pt >= 50) )

	return good_jets

def jet_nohiggs_selection(jets, fatjets, mask_fatjets, dr=1.2):
	
	nested_mask = jets.p4.match(fatjets.p4[mask_fatjets,0], matchfunc=pass_dr, dr=dr)
	jets_pass_dr = nested_mask.all()

	return jets_pass_dr

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

### PileUp weight
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

# lepton scale factors
def compute_lepton_weights(leps, evaluator, SF_list, lepton_eta=None, year=None):

	lepton_pt = leps.pt
	if lepton_eta is None:
		lepton_eta = leps.eta
	weights = np.ones(len(lepton_pt))

	for SF in SF_list:
		if SF.startswith('mu'):
			if year=='2016':
				if 'trigger' in SF:
					x = lepton_pt
					y = np.abs(lepton_eta)
				else:
					x = lepton_eta
					y = lepton_pt
			else:
				x = lepton_pt
				y = np.abs(lepton_eta)
		elif SF.startswith('el'):
			if 'trigger' in SF:
				x = lepton_pt
				y = lepton_eta
			else:
				x = lepton_eta
				y = lepton_pt
		else:
			raise Exception(f'unknown SF name {SF}')
		weights = weights*evaluator[SF](x, y)
		#weights *= evaluator[SF](x, y)
	
	per_event_weights = ak.prod(weights, axis=1)
	return per_event_weights

def METzCalculator_kernel(A, B, tmproot, tmpsol1, tmpsol2, pzlep, pznu, mask_rows):
	for i in range(len(tmpsol1)):
		if not mask_rows[i]:
			continue
		if tmproot[i]<0: pznu[i] = - B[i]/(2*A[i])
		else:
			tmpsol1[i] = (-B[i] + np.sqrt(tmproot[i]))/(2.0*A[i])
			tmpsol2[i] = (-B[i] - np.sqrt(tmproot[i]))/(2.0*A[i])
			if (abs(tmpsol2[i]-pzlep[i]) < abs(tmpsol1[i]-pzlep[i])):
				pznu[i] = tmpsol2[i]
				#otherSol_ = tmpsol1
			else:
				pznu[i] = tmpsol1[i]
				#otherSol_ = tmpsol2
				#### if pznu is > 300 pick the most central root
				if ( pznu[i] > 300. ):
					if (abs(tmpsol1[i])<abs(tmpsol2[i]) ):
						pznu[i] = tmpsol1[i]
						#otherSol_ = tmpsol2
					else:
						pznu[i] = tmpsol2[i]
						#otherSol_ = tmpsol1

def METzCalculator(lepton, MET, mask_rows):

	np.seterr(invalid='ignore') # to suppress warning from nonsense numbers in masked events
	M_W = 80.4
	M_lep = lepton.mass.content #.1056
	elep = lepton.E.content
	pxlep = lepton.x.content
	pylep = lepton.y.content
	pzlep = lepton.z.content
	pxnu = MET.x.content
	pynu = MET.y.content
	pznu = 0

	a = M_W*M_W - M_lep*M_lep + 2.0*pxlep*pxnu + 2.0*pylep*pynu
	A = 4.0*(elep*elep - pzlep*pzlep)
	#print(elep[np.isnan(A) & mask_rows], pzlep[np.isnan(A) & mask_rows])
	B = -4.0*a*pzlep
	C = 4.0*elep*elep*(pxnu*pxnu + pynu*pynu) - a*a
	#print(a, A, B, C)
	tmproot = B*B - 4.0*A*C

	tmpsol1 = np.zeros_like(A) #(-B + np.sqrt(tmproot))/(2.0*A)
	tmpsol2 = np.zeros_like(A) #(-B - np.sqrt(tmproot))/(2.0*A)
	pznu = np.zeros(len(M_lep), dtype=np.float32)
	METzCalculator_kernel(A, B, tmproot, tmpsol1, tmpsol2, pzlep, pznu, mask_rows)

	return pznu

def hadronic_W(jets, lepW):

	dijet = jets.choose(2)
	dijet.add_attributes(mass_diff=abs(dijet.mass - ak.fill_none(ak.min(lepW.mass, axis=1), 999.9)))
	#dijet.add_attributes(mass_diff=abs(dijet.mass - lepW.mass.content))
	min_akarray = ak.min(dijet.mass_diff, axis=1)
	min_mass_diff = ak.fill_none(min_akarray, value=9999.9)
	hadW = dijet[dijet.mass_diff <= min_mass_diff]
	n_hadW = np.array(np.invert(ak.is_none(min_akarray)), dtype=int)

	return hadW, n_hadW
