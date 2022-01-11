import argparse

import awkward as ak
import numpy as np
import math
import uproot
from vector import MomentumObject4D
import lhapdf

from coffea import hist
from coffea.nanoevents.methods import nanoaod

from definitions_dilepton_analysis import nureco

ak.behavior.update(nanoaod.behavior)

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

def get_dilepton_v7(electrons, muons, transverse=False):

	fields = {
			"pt": None,
			"eta": None,
			"phi": None,
			"mass": None,
			"charge": None,
	}

	electrons = ak.pad_none(electrons, 2)
	muons = ak.pad_none(muons, 2)

	nelectrons = ak.num(electrons[~ak.is_none(electrons, axis=1)])
	nmuons = ak.num(muons[~ak.is_none(muons, axis=1)])

	ee = electrons[:,0] + electrons[:,1]
	mumu = muons[:,0] + muons[:,1]
	emu = electrons[:,0] + muons[:,0]

	for var in fields.keys():
		fields[var] = ak.where( ((nelectrons + nmuons) == 2) & (nelectrons == 2), getattr(ee, var), getattr(ee, var) )
		fields[var] = ak.where( ((nelectrons + nmuons) == 2) & (nmuons == 2), getattr(mumu, var), fields[var] )
		fields[var] = ak.where( ((nelectrons + nmuons) == 2) & (nelectrons == 1) & (nmuons == 1), getattr(emu, var), fields[var] )

	if transverse:
		fields["eta"] = ak.zeros_like(fields["pt"])
	dileptons = ak.zip(fields, with_name="PtEtaPhiMCandidate")

	return dileptons

def get_diboson_v7(dileptons, MET, transverse=False):

	fields = {
			"pt": MET.pt,
			"eta": ak.zeros_like(MET.pt),
			"phi": MET.phi,
			"mass": ak.zeros_like(MET.pt),
			"charge": ak.zeros_like(MET.pt),
	}

	METs = ak.zip(fields, with_name="PtEtaPhiMCandidate")
	if transverse:
		dileptons_t = dileptons[:]
		dileptons_t["eta"] = ak.zeros_like(dileptons_t.eta)
		dibosons = (dileptons_t + METs)
	else:
		dibosons = (dileptons + METs)

	return dibosons

def get_charged_leptons_v7(electrons, muons, charge, mask):

	fields = {
			"pt": None,
			"eta": None,
			"phi": None,
			"mass": None,
			"energy": None,
			"charge": None,
			"x": None,
			"y": None,
			"z": None,
	}

	nelectrons = ak.num(electrons)
	nmuons = ak.num(muons)
	mask_ee = mask & ((nelectrons + nmuons) == 2) & (nelectrons == 2)
	mask_mumu = mask & ((nelectrons + nmuons) == 2) & (nmuons == 2)
	mask_emu = mask & ((nelectrons + nmuons) == 2) & (nelectrons == 1) & (nmuons == 1)

	for var in fields.keys():
		if var in ["eta", "phi"]:
			default = ak.from_iter(len(electrons)*[[-9.]])
		else:
			default = ak.from_iter(len(electrons)*[[-999.9]])
		if not var in ["energy", "x", "y", "z"]:
			fields[var] = ak.where( mask_ee, electrons[var][electrons.charge == charge], default )
			fields[var] = ak.where( mask_mumu, muons[var][muons.charge == charge], fields[var] )
			fields[var] = ak.where( mask_emu & ak.any(electrons.charge == charge, axis=1), electrons[var][electrons.charge == charge], fields[var] )
			fields[var] = ak.where( mask_emu & ak.any(muons.charge == charge, axis=1), muons[var][muons.charge == charge], fields[var] )
			fields[var] = ak.flatten(fields[var])
		else:
			fields[var] = ak.where( mask_ee, getattr(electrons, var)[electrons.charge == charge], default )
			fields[var] = ak.where( mask_mumu, getattr(muons, var)[muons.charge == charge], fields[var] )
			fields[var] = ak.where( mask_emu & ak.any(electrons.charge == charge, axis=1), getattr(electrons, var)[electrons.charge == charge], fields[var] )
			fields[var] = ak.where( mask_emu & ak.any(muons.charge == charge, axis=1), getattr(muons, var)[muons.charge == charge], fields[var] )
			fields[var] = ak.flatten(fields[var])

	charged_leptons = ak.zip(fields, with_name="PtEtaPhiMCandidate")

	return charged_leptons

# N.B.: This function works only with awkward v1.5.1 & coffea v0.7.9, it doesn't work with awkward 1.7.0 & coffea v0.7.11
def jet_selection_v7(jets, leps, mask_leps, cuts):

	# Only jets that are more distant than dr to ALL leptons are tagged as good jets
	sleps = leps[mask_leps]
	nlep_max = ak.max(ak.num(sleps, axis=1))

	# A fake lepton object filled with nan values is used for padding in the delta_r calculation
	pfake = {'pt' : [-9999.9], 'eta' : [-9999.9], 'phi' : [-9999.9], 'mass' : [-9999.9]}
	fakeLepton = ak.zip(pfake, with_name="PtEtaPhiMCandidate")
	slepsPadded = ak.fill_none(ak.pad_none(sleps, nlep_max), fakeLepton)
	good_jets = (jets.pt > cuts["pt"]) & (np.abs(jets.eta) < cuts["eta"]) & (jets.jetId >= cuts["jetId"])

	for i in range(nlep_max):
		jets_pass_dr = (jets.delta_r(slepsPadded[:,i]) > cuts["dr"])
		jets_pass_dr = ak.fill_none(jets_pass_dr, True)
		good_jets = good_jets & jets_pass_dr


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

# Function to compute the PDF weight 
def PDFweight(x1, x2, Q, pdf):

	weight = 0
	for p1, p2 in [(lhapdf.UP, lhapdf.AUP), (lhapdf.DOWN, lhapdf.ADOWN), (lhapdf.GLUON, lhapdf.GLUON)]:
		isGluon = (p1 == lhapdf.GLUON)
		if isGluon:
			weight += pdf.xfxQ(p1, x1, Q)*pdf.xfxQ(p2, x2, Q)
		else:
			weight += pdf.xfxQ(p1, x1, Q)*pdf.xfxQ(p2, x2, Q)
			weight += pdf.xfxQ(p1, x2, Q)*pdf.xfxQ(p2, x1, Q)	# permutation between up and anti-up quarks

	return weight

def pnuCalculator_v7(leptons, leptons_bar, bjets, METs, fatjets, scan=True):

	# As we have no information regarding the charge of the jet, in principle we should iterate
	# over all the possible (b, b_bar) pairs of bjet pairs. In order to assign a bjet to b or b_bar
	# I could use the information of leptons, e.g. DeltaR(l, b) or m_lb

	pairs = ak.combinations(bjets, 2)
	pdf = lhapdf.mkPDF(nureco['PDF'])

	if scan == True:
		M_t_grid = np.linspace(*nureco['m_t']['scan'])
		M_W_grid = np.linspace(*nureco['m_W']['scan'])
	else:
		M_t_grid = [ nureco['m_t']['nominal'] ]
		M_W_grid = [ nureco['m_W']['nominal'] ]

	M_b = nureco['m_b']['nominal']

	pnu    = {'x' : [], 'y' : [], 'z' : [], 'pt' : [], 'eta' : [], 'phi' : [], 'mass' : [], 'charge' : []}
	pnubar = {'x' : [], 'y' : [], 'z' : [], 'pt' : [], 'eta' : [], 'phi' : [], 'mass' : [], 'charge' : []}
	pbjets = {'x' : [], 'y' : [], 'z' : [], 'pt' : [], 'eta' : [], 'phi' : [], 'mass' : [], 'charge' : []}
	pbbarjets = {'x' : [], 'y' : [], 'z' : [], 'pt' : [], 'eta' : [], 'phi' : [], 'mass' : [], 'charge' : []}

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
		#m_w_plus_reco_list = []
		PDFweight_list = []
		l     = None
		l_bar = None
		MET   = None
		fatjet = fatjets[ievt]

		for M_t in M_t_grid:
			for M_W in M_W_grid:
				print("Reco with m_top =", M_t, "m_w =", M_W)
				for reverse in [False, True]:
					if leptons[ievt].pt < 0:
						#if reverse == False:
						#	for key in pnu.keys():
						#		pnu[key].append(-9999.9)
						#	for key in pnubar.keys():
						#		pnubar[key].append(-9999.9)
						#	for key in pbjets.keys():
						#		pbjets[key].append(-9999.9)
						#	for key in pbbarjets.keys():
						#		pbbarjets[key].append(-9999.9)
						continue
					for i in range(ak.num(pairs)[ievt]):
						l     = leptons[ievt]
						l_bar = leptons_bar[ievt]
						MET   = METs[ievt]
						if not reverse:
							b     = pairs['0'][ievt,i]
							b_bar = pairs['1'][ievt,i]
						else:
							b     = pairs['1'][ievt,i]
							b_bar = pairs['0'][ievt,i]

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
						if (len(pnu_xs) == 0) | (fatjet.pt == None):
							if ((reverse == True) & (len(pnu_x_list) == 0) & (i == ak.num(pairs)[ievt]-1)):
								pnu_x_list.append(-9999.9)
								pnu_y_list.append(-9999.9)
								pnu_z_list.append(-9999.9)
								pnubar_x_list.append(-9999.9)
								pnubar_y_list.append(-9999.9)
								pnubar_z_list.append(-9999.9)
								pbjets_x_list.append(-9999.9)
								pbjets_y_list.append(-9999.9)
								pbjets_z_list.append(-9999.9)
								pbjets_mass_list.append(-9999.9)
								pbbarjets_x_list.append(-9999.9)
								pbbarjets_y_list.append(-9999.9)
								pbbarjets_z_list.append(-9999.9)
								pbbarjets_mass_list.append(-9999.9)
								#m_w_plus_reco_list.append(-9999.9)
								PDFweight_list.append(-9999.9)
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
							pnu_ys = []
							pnu_zs = []
							pnubar_xs = []
							pnubar_ys = []
							pnubar_zs = []
							for pnu_x_sol in pnu_xs:
								# neutrino y,z momentum components from pnu_x
								pnu_z_sol	  = - (a1 + a2*pnu_x_sol + a3*pnu_y)/a4
								#pnu_zs.append(pnu_z_sol)
								#pnu_ys.append(pnu_y)
								# anti-neutrino momentum components from neutrino components
								pnubar_x = MET.x - pnu_x_sol
								pnubar_y = MET.y - pnu_y
								pnubar_z = - (b1 + b2*pnubar_x + b3*pnubar_y)/b4
								#pnubar_xs.append(pnubar_x)
								#pnubar_ys.append(pnubar_y)
								#pnubar_zs.append(pnubar_z)
								pnu_x_list.append(pnu_x_sol)
								pnu_y_list.append(pnu_y)
								pnu_z_list.append(pnu_z_sol)
								pnubar_x_list.append(pnubar_x)
								pnubar_y_list.append(pnubar_y)
								pnubar_z_list.append(pnubar_z)
								# reconstruction of W and top masses
								neutrino	  = MomentumObject4D.from_xyzt(pnu_x_sol, pnu_y, pnu_z_sol, np.sqrt(pnu_x_sol**2 + pnu_y**2 + pnu_z_sol**2))
								lepton_plus	  = MomentumObject4D.from_xyzt(l_bar.x, l_bar.y, l_bar.z, np.sqrt(l_bar.x**2 + l_bar.y**2 + l_bar.z**2 + l_bar.mass**2))
								b_4vec        = MomentumObject4D.from_xyzt(b.x, b.y, b.z, b.mass)
								antineutrino  = MomentumObject4D.from_xyzt(pnubar_x, pnubar_y, pnubar_z, np.sqrt(pnubar_x**2 + pnubar_y**2 + pnubar_z**2))
								lepton_minus  = MomentumObject4D.from_xyzt(l.x, l.y, l.z, np.sqrt(l.x**2 + l.y**2 + l.z**2 + l.mass**2))
								b_bar_4vec    = MomentumObject4D.from_xyzt(b_bar.x, b_bar.y, b_bar.z, b_bar.mass)
								fatjet		  = MomentumObject4D.from_xyzt(fatjet.x, fatjet.y, fatjet.z, fatjet.mass)
								#w_plus		  = (neutrino + lepton_plus)
								top			  = (neutrino + lepton_plus + b_4vec)
								#w_minus		  = (antineutrino + lepton_minus)
								antitop		  = (antineutrino + lepton_minus + b_bar_4vec)
								ttH			  = top + antitop + fatjet
								#m_w_plus	  = w_plus.mass
								#m_top		  = top.mass
								#m_w_minus	  = w_minus.mass
								#m_antitop	  = antitop.mass
								#m_w_plus_list.append(m_w_plus)
								#m_top_list.append(m_top)
								#m_w_minus_list.append(m_w_minus)
								#m_antitop_list.append(m_antitop)
								x1 = ( ttH.energy - ttH.z ) / nureco['Ecm']
								x2 = ( ttH.energy + ttH.z ) / nureco['Ecm']
								if ( (x1 < 0) | (x1 > 1) ) | ( (x2 < 0) | (x2 > 1) ):
									PDFweight_list.append( -9999.9 )
								else:
									PDFweight_list.append( PDFweight(x1, x2, Q=nureco['Q'], pdf=pdf) )

							
							pbjets_x_list += len(pnu_xs)*[b.x]
							pbjets_y_list += len(pnu_xs)*[b.y]
							pbjets_z_list += len(pnu_xs)*[b.z]
							pbjets_mass_list += len(pnu_xs)*[b.mass]
							pbbarjets_x_list += len(pnu_xs)*[b_bar.x]
							pbbarjets_y_list += len(pnu_xs)*[b_bar.y]
							pbbarjets_z_list += len(pnu_xs)*[b_bar.z]
							pbbarjets_mass_list += len(pnu_xs)*[b_bar.mass]

						#pnubar_x = MET.x - pnu_x
						#pnubar_y = MET.y - pnu_y
						#pnubar_z = - (b1 + b2*pnubar_x + b3*pnubar_y)/b4

						#pnu_x_list.append(pnu_x)
						#pnu_y_list.append(pnu_y)
						#pnu_z_list.append(pnu_z)
						#pnubar_x_list.append(pnubar_x)
						#pnubar_y_list.append(pnubar_y)
						#pnubar_z_list.append(pnubar_z)
						#pbjets_x_list.append(b.x)
						#pbjets_y_list.append(b.y)
						#pbjets_z_list.append(b.z)
						#pbjets_mass_list.append(b.mass)
						#pbbarjets_x_list.append(b_bar.x)
						#pbbarjets_y_list.append(b_bar.y)
						#pbbarjets_z_list.append(b_bar.z)
						#pbbarjets_mass_list.append(b_bar.mass)
						#m_w_plus_reco_list.append(m_w_plus_reco)
		
		# The solution with the highest PDF weight is chosen
		if len(pnu_x_list) == 0:
			for key in pnu.keys():
				pnu[key].append(-9999.9)
				pnubar[key].append(-9999.9)
				pbjets[key].append(-9999.9)
				pbbarjets[key].append(-9999.9)
		else:
			j_max = np.argmax(PDFweight_list)
			pnu['x'].append(pnu_x_list[j_max])
			pnu['y'].append(pnu_y_list[j_max])
			pnu['z'].append(pnu_z_list[j_max])
			p4 = MomentumObject4D.from_xyzt(pnu_x_list[j_max], pnu_y_list[j_max], pnu_z_list[j_max], np.sqrt(pnu_x_list[j_max]**2 + pnu_y_list[j_max]**2 + pnu_z_list[j_max]**2))
			pnu['pt'].append(p4.pt)
			pnu['eta'].append(p4.eta)
			pnu['phi'].append(p4.phi)
			pnu['mass'].append(p4.mass)
			pnu['charge'].append(0)
			pnubar['x'].append(pnubar_x_list[j_max])
			pnubar['y'].append(pnubar_y_list[j_max])
			pnubar['z'].append(pnubar_z_list[j_max])
			p4 = MomentumObject4D.from_xyzt(pnubar_x_list[j_max], pnubar_y_list[j_max], pnubar_z_list[j_max], np.sqrt(pnubar_x_list[j_max]**2 + pnubar_y_list[j_max]**2 + pnubar_z_list[j_max]**2))
			pnubar['pt'].append(p4.pt)
			pnubar['eta'].append(p4.eta)
			pnubar['phi'].append(p4.phi)
			pnubar['mass'].append(p4.mass)
			pnubar['charge'].append(0)
			pbjets['x'].append(pbjets_x_list[j_max])
			pbjets['y'].append(pbjets_y_list[j_max])
			pbjets['z'].append(pbjets_z_list[j_max])
			pbjets['mass'].append(pbjets_mass_list[j_max])
			p4 = MomentumObject4D.from_xyzt(pbjets_x_list[j_max], pbjets_y_list[j_max], pbjets_z_list[j_max], np.sqrt(pbjets_mass_list[j_max]**2 + pbjets_x_list[j_max]**2 + pbjets_y_list[j_max]**2 + pbjets_z_list[j_max]**2))
			pbjets['pt'].append(p4.pt)
			pbjets['eta'].append(p4.eta)
			pbjets['phi'].append(p4.phi)
			pbjets['charge'].append(-1./3.)
			pbbarjets['x'].append(pbbarjets_x_list[j_max])
			pbbarjets['y'].append(pbbarjets_y_list[j_max])
			pbbarjets['z'].append(pbbarjets_z_list[j_max])
			pbbarjets['mass'].append(pbbarjets_mass_list[j_max])
			p4 = MomentumObject4D.from_xyzt(pbbarjets_x_list[j_max], pbbarjets_y_list[j_max], pbbarjets_z_list[j_max], np.sqrt(pbbarjets_mass_list[j_max]**2 + pbbarjets_x_list[j_max]**2 + pbbarjets_y_list[j_max]**2 + pbbarjets_z_list[j_max]**2))
			pbbarjets['pt'].append(p4.pt)
			pbbarjets['eta'].append(p4.eta)
			pbbarjets['phi'].append(p4.phi)
			pbbarjets['charge'].append(+1./3.)

	# Here we have to cleverly organise the output as the number of sets of solutions is equal to N_b(N_b - 1)
	# Then we have to choose a criterion in order to choose the correct (b, b_bar) pair
	for item in pnu.keys():
		pnu[item] = np.array(pnu[item])
		pnubar[item] = np.array(pnubar[item])
	for item in pbjets.keys():
		pbjets[item] = np.array(pbjets[item])
		pbbarjets[item] = np.array(pbbarjets[item])
	pnu = ak.zip(pnu, with_name="PtEtaPhiMCandidate")
	pnubar = ak.zip(pnubar, with_name="PtEtaPhiMCandidate")
	pbjets = ak.zip(pbjets, with_name="PtEtaPhiMCandidate")
	pbbarjets = ak.zip(pbbarjets, with_name="PtEtaPhiMCandidate")

	return pnu, pnubar, pbjets, pbbarjets, mask_events_withsol
