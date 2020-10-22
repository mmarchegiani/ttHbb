import argparse

from coffea import processor, hist

import lib_analysis
from lib_analysis import ttHbb

data_dir = "/afs/cern.ch/work/m/mmarcheg/Coffea/test/"
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
	processor.iterative_executor,
	{"nano": True},
)

ax = hist.plot1d(result['pt_muon'], overlay='dataset')
ax.figure.savefig("pt_muon.png", format="png")
