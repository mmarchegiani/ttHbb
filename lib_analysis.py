import argparse

import awkward1 as ak
import numpy as np
import math
import uproot
from uproot_methods import TLorentzVector
#from uproot3_methods import TLorentzVector

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

def get_charge_sum(electrons, muons):

	electron_charge = ak.sum(electrons.charge, axis=1)
	muon_charge = ak.sum(muons.charge, axis=1)

	return electron_charge + muon_charge

def get_dilepton_vars(electrons, muons, default=-999.9):

	nelectrons = electrons.counts
	nmuons = muons.counts
	default = ak.from_iter(len(electrons)*[default])
	e_pairs = electrons.choose(2)
	mu_pairs = muons.choose(2)
	e_mu_pairs = electrons.cross(muons)


	def get_var(varname):
		
		var_ee   = ak.max(getattr(e_pairs, varname), axis=1)
		var_mumu = ak.max(getattr(mu_pairs, varname), axis=1)
		var_e_mu = ak.max(getattr(e_mu_pairs, varname), axis=1)
		var = ak.where( ((nelectrons + nmuons) == 2) & (nelectrons == 2), var_ee, default )
		var = ak.where( ((nelectrons + nmuons) == 2) & (nmuons == 2), var_mumu, var )
		var = ak.where( ((nelectrons + nmuons) == 2) & (nelectrons == 1) & (nmuons == 1), var_e_mu, var )

		return ak.from_iter(var)

	varnames = ['pt', 'eta', 'phi', 'mass']
	variables = []
	for var in varnames:
		variables.append(get_var(var))

	return variables

def get_transverse_mass(dileptons, METs):

	default = ak.from_iter(len(dileptons)*[-999.9])
	phill = ak.max(dileptons.phi, axis=1)
	phimet = ak.max(METs.phi, axis=1)
	dphi = (phill - phimet + np.pi) % (2*np.pi) - np.pi
	ptll = ak.max(dileptons.pt, axis=1)
	ptmet = ak.max(METs.pt, axis=1)

	mt = np.sqrt(2*ptll*ptmet*(1 - np.cos(dphi)))

	return ak.from_iter(ak.where(ak.is_none(mt), default, mt))

def get_charged_var(varname, electrons, muons, charge, mask, default=-999.9):

	nelectrons = electrons.counts
	nmuons = muons.counts
	default = ak.from_iter(len(electrons)*[[default]])
	mask_ee = mask & ((nelectrons + nmuons) == 2) & (nelectrons == 2)
	mask_mumu = mask & ((nelectrons + nmuons) == 2) & (nmuons == 2)
	mask_emu = mask & ((nelectrons + nmuons) == 2) & (nelectrons == 1) & (nmuons == 1)

	var = ak.where(mask_ee, electrons[varname][electrons.charge == charge], default)
	var = ak.where(mask_mumu, muons[varname][muons.charge == charge], var)
	var = ak.where(mask_emu & ak.any(electrons.charge == charge, axis=1), electrons[varname][electrons.charge == charge], var)
	var = ak.where(mask_emu & ak.any(muons.charge == charge, axis=1), muons[varname][muons.charge == charge], var)

	return ak.from_iter(ak.flatten(var))

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

def calc_dphi(objects1, objects2):

	pairs = objects1.cross(objects2)

	dphi = abs( (pairs.i0.phi - pairs.i1.phi + np.pi) % (2*np.pi) - np.pi )

	return get_leading_value(ak.from_iter(dphi))

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

def jet_nohiggs_selection(jets, mask_jets, fatjets, dr=1.2):

	#nested_mask = jets.p4.match(fatjets.p4[mask_fatjets,0], matchfunc=pass_dr, dr=dr)
	nested_mask = jets.match(fatjets, matchfunc=pass_dr, dr=dr)
	jets_pass_dr = nested_mask.all()

	return mask_jets & jets_pass_dr

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

def METzCalculator_kernel(A, B, tmproot, tmpsol1, tmpsol2, pzlep, pznu):
	for i in range(len(tmpsol1)):
		#if not mask_rows[i]:
		#	continue
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
	return pznu

def METzCalculator(lepton, MET):

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

	return METzCalculator_kernel(A, B, tmproot, tmpsol1, tmpsol2, pzlep, pznu)

#def hadronic_W(jets, lepW, event):
def hadronic_W(jets):

	dijet = jets.choose(2)
	M_W = 80.4
	#dijet.add_attributes(mass_diff=abs(dijet.mass - ak.fill_none(ak.min(lepW.mass, axis=1), 999.9)))
	dijet.add_attributes(mass_diff=abs(dijet.mass - M_W))
	min_akarray = ak.min(dijet.mass_diff, axis=1)
	min_mass_diff = ak.fill_none(min_akarray, value=9999.9)
	hadW = dijet[dijet.mass_diff <= min_mass_diff]
	n_hadW = np.array(np.invert(ak.is_none(min_akarray)), dtype=int)

	return hadW, n_hadW

def pnuCalculator(leptons, leptons_bar, bjets, METs):

	# As we have no information regarding the charge of the jet, in principle we should iterate
	# over all the possible (b, b_bar) pairs of bjet pairs. In order to assign a bjet to b or b_bar
	# I could use the information of leptons, e.g. DeltaR(l, b) or m_lb

	pairs = bjets.choose(2)

	M_W = 80.4
	M_t = 173.1
	M_b = 4.2

	pnu    = {'x' : [], 'y' : [], 'z' : []}
	pnubar = {'x' : [], 'y' : [], 'z' : []}
	pbjets = {'x' : [], 'y' : [], 'z' : [], 'mass' : []}
	pbbarjets = {'x' : [], 'y' : [], 'z' : [], 'mass' : []}

	nEvents = len(pairs)
	mask_events_withsol = np.zeros(nEvents, dtype=np.bool)

	for ievt in range(nEvents):
		pnu_x_list = []
		pnu_y_list = []
		pnu_z_list = []
		pnubar_x_list = []
		pnubar_y_list = []
		pnubar_z_list = []
		pbjets_x_list = []
		pbjets_y_list = []
		pbjets_z_list = []
		pbjets_mass_list = []
		pbbarjets_x_list = []
		pbbarjets_y_list = []
		pbbarjets_z_list = []
		pbbarjets_mass_list = []
		m_w_plus_reco_list = []
		b_list = []
		b_bar_list = []
		l     = None
		l_bar = None
		MET   = None

		for reverse in [False, True]:
			if leptons.counts[ievt] == 0:
				if reverse == False:
					pnu['x'].append(-9999.9)
					pnu['y'].append(-9999.9)
					pnu['z'].append(-9999.9)
					pnubar['x'].append(-9999.9)
					pnubar['y'].append(-9999.9)
					pnubar['z'].append(-9999.9)
					pbjets['x'].append(-9999.9)
					pbjets['y'].append(-9999.9)
					pbjets['z'].append(-9999.9)
					pbjets['mass'].append(-9999.9)
					pbbarjets['x'].append(-9999.9)
					pbbarjets['y'].append(-9999.9)
					pbbarjets['z'].append(-9999.9)
					pbbarjets['mass'].append(-9999.9)
					#b_jets.append([])
					#b_bar_jets.append([])
				continue
			for i in range(pairs.counts[ievt]):
				l     = leptons.p4[ievt,0]
				l_bar = leptons_bar.p4[ievt,0]
				MET   = METs.p4[ievt,0]
				if not reverse:
					b     = pairs.i0.p4[ievt,i]
					b_bar = pairs.i1.p4[ievt,i]
					b_obj = pairs.i0[ievt,i]
					b_bar_obj = pairs.i1[ievt,i]
				else:
					b     = pairs.i1.p4[ievt,i]
					b_bar = pairs.i0.p4[ievt,i]
					b_obj = pairs.i1[ievt,i]
					b_bar_obj = pairs.i0[ievt,i]

				a1 = ( (b.energy + l_bar.energy)*(M_W**2 - l_bar.mass**2)
					   - l_bar.energy*(M_t**2 - M_b**2 - l_bar.mass**2)
					   + 2*b.energy*l_bar.energy**2 - 2*l_bar.energy*(b.x*l_bar.x + b.y*l_bar.y + b.z*l_bar.z) )
				a2 = 2*(b.energy*l_bar.x - l_bar.energy*b.x)
				a3 = 2*(b.energy*l_bar.y - l_bar.energy*b.y)
				a4 = 2*(b.energy*l_bar.z - l_bar.energy*b.z)

				b1 = ( (b_bar.energy + l.energy)*(M_W**2 - l.mass**2)
					   - l.energy*(M_t**2 - M_b**2 - l.mass**2)
					   + 2*b_bar.energy*l.energy**2 - 2*l.energy*(b_bar.x*l.x + b_bar.y*l.y + b_bar.z*l.z) )
				b2 = 2*(b_bar.energy*l.x - l.energy*b_bar.x)
				b3 = 2*(b_bar.energy*l.y - l.energy*b_bar.y)
				b4 = 2*(b_bar.energy*l.z - l.energy*b_bar.z)

				def coeffs(lept, coefficients):

					k1, k2, k3, k4 = coefficients
					F = (M_W**2 - lept.mass**2)
					pt2 = (lept.energy**2 - lept.z**2)
					K1 = k1/k4
					K2 = k2/k4
					K3 = k3/k4
					K12 = k1*k2/k4**2
					K13 = k1*k3/k4**2
					K23 = k2*k3/k4**2

					k22 = ( F**2 - 4*pt2*K1**2
							- 4*F*lept.z*K1 )
					k21 = ( 4*F*(lept.x - lept.z*K2)
							- 8*pt2*K12 - 8*lept.x*lept.z*K1 )
					k20 = ( - 4*(lept.energy**2 - lept.x**2) - 4*pt2*K2**2
							- 8*lept.x*lept.z*K2 )
					k11 = ( 4*F*(lept.y - lept.z*K3)
							- 8*pt2*K13 - 8*lept.y*lept.z*K1 )
					k10 = ( - 8*pt2*K23 + 8*lept.x*lept.y
							- 8*lept.x*lept.z*K3 - 8*lept.y*lept.z*K2 )
					k00 = ( - 4*(lept.energy**2 - lept.y**2) - 4*pt2*K3**2
							- 8*lept.y*lept.z*K3 )

					return (k22, k21, k20, k11, k10, k00)

				c22, c21, c20, c11, c10, c00 = coeffs(l_bar, (a1,a2,a3,a4))
				d22_, d21_, d20_, d11_, d10_, d00_ = coeffs(l, (b1,b2,b3,b4))

				d22 = (d22_ + (MET.x**2)*d20_ + (MET.y**2)*d00_ + MET.x*MET.y*d10_
					   + MET.x*d21_ + MET.y*d11_)
				d21 = - d21_ - 2*MET.x*d20_ - MET.y*d10_
				d20 = d20_
				d11 = - d11_ - 2*MET.y*d00_ - MET.x*d10_
				d10 = d10_
				d00 = d00_

				h4 = ((c00**2)*(d22**2) + c11*d22*(c11*d00 - c00*d11)
					  + c00*c22*(d11**2 - 2*d00*d22) + c22*d00*(c22*d00 - c11*d11))
				h3 = (c00*d21*(2*c00*d22 - c11*d11) + c00*d11*(2*c22*d10 + c21*d11)
					  + c22*d00*(2*c21*d00 - c11*d10) - c00*d22*(c11*d10 + c10*d11)
					  - 2*c00*d00*(c22*d21 + c21*d22) - d00*d11*(c11*c21 + c10*c22)
					  + c11*d00*(c11*d21 + 2*c10*d22))
				h2 = ((c00**2)*(2*d22*d20 + d21**2) - c00*d21*(c11*d10 + c10*d11)
					  + c11*d20*(c11*d00 - c00*d11) + c00*d10*(c22*d10 - c10*d22)
					  + c00*d11*(2*c21*d10 + c20*d11) + (2*c22*c20 + c21**2)*d00**2
					  - 2*c00*d00*(c22*d20 + c21*d21 + c20*d22)
					  + c10*d00*(2*c11*d21 + c10*d22) - d00*d10*(c11*c21 + c10*c22)
					  - d00*d11*(c11*c20 + c10*c21))
				h1 = (c00*d21*(2*c00*d20 - c10*d10) - c00*d20*(c11*d10 + c10*d11)
					  + c00*d10*(c21*d10 + 2*c20*d11) - 2*c00*d00*(c21*d20 + c20*d21)
					  + c10*d00*(2*c11*d20 + c10*d21) - c20*d00*(2*c21*d00 - c10*d11)
					  - d00*d10*(c11*c20 + c10*c21))
				h0 = ((c00**2)*(d20**2) + c10*d20*(c10*d00 - c00*d10)
					  + c20*d10*(c00*d10 - c10*d00) + c20*d00*(c20*d00 - 2*c00*d20))

				pnu_xs = np.roots((h0,h1,h2,h3,h4))
				pnu_xs = pnu_xs[np.isreal(pnu_xs)].real
				# Naive choice: the first solution or its real part is chosen
				#pnu_x  = np.real(pnu_xs).real[0]
				pnu_x = None
				pnu_y = None
				pnu_z = None
				m_w_plus_reco = None
				if len(pnu_xs) == 0:
					if ((reverse == True) & (len(pnu_x_list) == 0) & (i == pairs.counts[ievt]-1)):
						pnu_x_list = [-9999.9]
						pnu_y_list = [-9999.9]
						pnu_z_list = [-9999.9]
						pnubar_x_list = [-9999.9]
						pnubar_y_list = [-9999.9]
						pnubar_z_list = [-9999.9]
						pbjets_x_list = [-9999.9]
						pbjets_y_list = [-9999.9]
						pbjets_z_list = [-9999.9]
						pbjets_mass_list = [-9999.9]
						pbbarjets_x_list = [-9999.9]
						pbbarjets_y_list = [-9999.9]
						pbbarjets_z_list = [-9999.9]
						pbbarjets_mass_list = [-9999.9]
						m_w_plus_reco_list = [-9999.9]
					continue
				else:
					mask_events_withsol[ievt] = True					
					c0 = c00
					c1 = c11
					c2 = c22
					d0 = d00
					d1 = d11
					d2 = d22
					pnu_y = (c0*d2 - c2*d0)/(c1*d0 - c0*d1)
					masses = []
					pnu_zs = []
					for pnu_x_sol in pnu_xs:
						pnu_z_sol	  = - (a1 + a2*pnu_x_sol + a3*pnu_y)/a4
						pnu_zs.append(pnu_z_sol)
						neutrino	  = TLorentzVector(pnu_x_sol, pnu_y, pnu_z_sol, np.sqrt(pnu_x_sol**2 + pnu_y**2 + pnu_z_sol**2))
						lepton_plus	  = TLorentzVector(l_bar.x, l_bar.y, l_bar.z, np.sqrt(l_bar.x**2 + l_bar.y**2 + l_bar.z**2 + l_bar.mass**2))
						m_w = (neutrino + lepton_plus).mass
						masses.append(m_w)

					i_min = np.argmin(np.abs(np.array(masses) - M_W))
					pnu_x  = pnu_xs[i_min]
					pnu_z  = pnu_zs[i_min]
					m_w_plus_reco = masses[i_min]

				pnubar_x = MET.x - pnu_x
				pnubar_y = MET.y - pnu_y
				pnubar_z = - (b1 + b2*pnubar_x + b3*pnubar_y)/b4

				pnu_x_list.append(pnu_x)
				pnu_y_list.append(pnu_y)
				pnu_z_list.append(pnu_z)
				pnubar_x_list.append(pnubar_x)
				pnubar_y_list.append(pnubar_y)
				pnubar_z_list.append(pnubar_z)
				pbjets_x_list.append(b.x)
				pbjets_y_list.append(b.y)
				pbjets_z_list.append(b.z)
				pbjets_mass_list.append(b.mass)
				pbbarjets_x_list.append(b_bar.x)
				pbbarjets_y_list.append(b_bar.y)
				pbbarjets_z_list.append(b_bar.z)
				pbbarjets_mass_list.append(b_bar.mass)
				m_w_plus_reco_list.append(m_w_plus_reco)
				b_list.append(b_obj)
				b_bar_list.append(b_bar_obj)
		# Naive choice: the first (b, b_bar) configuration is chosen
		if len(pnu_x_list) != 0:
			j_min = np.argmin(np.abs(np.array(m_w_plus_reco_list) - M_W))
			pnu['x'].append(pnu_x_list[j_min])
			pnu['y'].append(pnu_y_list[j_min])
			pnu['z'].append(pnu_z_list[j_min])
			pnubar['x'].append(pnubar_x_list[j_min])
			pnubar['y'].append(pnubar_y_list[j_min])
			pnubar['z'].append(pnubar_z_list[j_min])
			pbjets['x'].append(pbjets_x_list[j_min])
			pbjets['y'].append(pbjets_y_list[j_min])
			pbjets['z'].append(pbjets_z_list[j_min])
			pbjets['mass'].append(pbjets_mass_list[j_min])
			pbbarjets['x'].append(pbbarjets_x_list[j_min])
			pbbarjets['y'].append(pbbarjets_y_list[j_min])
			pbbarjets['z'].append(pbbarjets_z_list[j_min])
			pbbarjets['mass'].append(pbbarjets_mass_list[j_min])
			#if len(b_list) == 2:
			#	b_jets.append([b_list[j_min]])
			#	b_bar_jets.append([b_bar_list[j_min]])
			#elif len(b_list) == 1:
			#	b_jets.append([b_list[0]])
			#	b_bar_jets.append([b_bar_list[0]])
			#elif len(b_list) == 0:
			#	b_jets.append([])
			#	b_bar_jets.append([])

	# Here we have to cleverly organise the output as the number of sets of solutions is equal to N_b(N_b - 1)
	# Then I have to choose a criterion in order to choose the correct (b, b_bar) pair
	for item in pnu.keys():
		pnu[item] = np.array(pnu[item])
		pnubar[item] = np.array(pnubar[item])
	for item in pbjets.keys():
		pbjets[item] = np.array(pbjets[item])
		pbbarjets[item] = np.array(pbbarjets[item])

	return pnu, pnubar, pbjets, pbbarjets, mask_events_withsol

def obj_reco(leptons, antileptons, neutrinos, antineutrinos, bjets, bbarjets, mask_events):

	wm      = leptons.p4 + antineutrinos.p4
	wp      = antileptons.p4 + neutrinos.p4
	top     = antileptons.p4 + neutrinos.p4 + bjets.p4
	antitop = leptons.p4 + antineutrinos.p4 + bbarjets.p4
	tt 		= top + antitop

	w_minus = {'pt' : [], 'eta' : [], 'phi' : [], 'mass' : []}
	w_plus  = {'pt' : [], 'eta' : [], 'phi' : [], 'mass' : []}
	t       = {'pt' : [], 'eta' : [], 'phi' : [], 'mass' : []}
	tbar    = {'pt' : [], 'eta' : [], 'phi' : [], 'mass' : []}
	ttbar   = {'pt' : [], 'eta' : [], 'phi' : [], 'mass' : []}

	def get_var(object, varname, default=-999.9):

		default = ak.from_iter(len(mask_events)*[default])
		values = ak.max(getattr(object, varname), axis=1)
		var    = ak.where(mask_events, values, default)
		return ak.from_iter(var)

	for varname in t.keys():
		if varname == 'eta':
			default = -9.
		else:
			default = -999.9
		w_minus[varname] = get_var(wm, varname, default)
		w_plus[varname]  = get_var(wp, varname, default)
		t[varname]       = get_var(top, varname, default)
		tbar[varname]    = get_var(antitop, varname, default)
		ttbar[varname]   = get_var(tt, varname, default)

	return w_minus, w_plus, t, tbar, ttbar

"""
def w_mass(leptons, neutrinos, mask_events):
	
	lepW = leptons.p4 + neutrinos.p4
	#for ievt in range(mask_events):
	#	if mask_events[i] == False:
	m_w = lepW.mass
	m_w_filled = []
	for (i, mass) in enumerate(m_w):
		if mask_events[i]:
			m_w_filled.append(mass[0])
		else:
			m_w_filled.append(-999.9)

	return ak.from_iter(m_w_filled)

def t_mass(leptons, neutrinos, bjets, mask_events):
	
	top = leptons.p4 + neutrinos.p4 + bjets.p4
	#for ievt in range(mask_events):
	#	if mask_events[i] == False:
	m_t = top.mass
	m_t_filled = []
	for (i, mass) in enumerate(m_t):
		if mask_events[i]:
			m_t_filled.append(mass[0])
		else:
			m_t_filled.append(-999.9)

	return ak.from_iter(m_t_filled)

def top_reco(leptons, antileptons, neutrinos, antineutrinos, bjets, bbarjets, mask_events):

	top     = antileptons.p4 + neutrinos.p4 + bjets.p4
	antitop = leptons.p4 + antineutrinos.p4 + bbarjets.p4
	tt 		= top + antitop

	m_tt = tt.mass
	m_tt_filled = []
	for (i, mass) in enumerate(m_tt):
		if mask_events[i]:
			m_tt_filled.append(mass[0])
		else:
			m_tt_filled.append(-999.9)

	return ak.from_iter(m_tt_filled)
"""