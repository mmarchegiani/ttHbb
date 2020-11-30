import os
import json
import numpy as np
import matplotlib.pyplot as plt
from coffea import hist

def get_bin_centers(edges):
	bin_width = edges[1] -edges[0]
	return np.array([edge + 0.5*bin_width for edge in edges[:-1]])

error_opts = {
    #'label':'Stat. Unc.',
    #'hatch':'///',
    #'facecolor':'none',
    #'edgecolor':(0,0,0,.5),
    'yerr': 0
    #'visible': False
}

dir_hepaccelerate = "/eos/home-a/algomez/tmpFiles/hepacc/results/2017/v14/met20_btagDDBvL086/nominal/"
dir_coffea = "plots/comparison/nominal/"

dirs = {'hepaccelerate' : dir_hepaccelerate, 'coffea' : dir_coffea}
files = {'ttHTobb' : "out_ttHTobb_nominal_merged.json", 'TTToSemiLeptonic' : "out_TTToSemiLeptonic_nominal_merged.json"}

histograms = {}
keys = []
with open(dir_coffea + files['ttHTobb']) as json_file:
	data = json.load(json_file)
	keys = data.keys()
	json_file.close()

for sample, file in files.items():
	for dataset, directory in dirs.items():
		with open(directory + file) as json_file:
			data = json.load(json_file)
			for hist_name in keys:
				if not "2J2WdeltaR_weights_nominal" in hist_name:
					continue
				print(hist_name)
				contents = np.array(data[hist_name]['contents'])
				edges = np.array(data[hist_name]['edges'])
				split_name = hist_name.split('_')
				varname = split_name[1] + " " + split_name[2]
				if split_name[2] == "2J2WdeltaR":
					varname = split_name[1]
				try:
					histograms[hist_name].fill(dataset=dataset, values=get_bin_centers(edges), weight=contents)
					#ax = hist.plot1d(histograms[hist_name], overlay='dataset', error_opts=error_opts, density=True)
					ax = hist.plot1d(histograms[hist_name], overlay='dataset', density=True)
					plot_dir = "plots/comparison/nominal/" + sample + "/"
					if not os.path.exists(plot_dir):
						os.makedirs(plot_dir)
					ax.figure.savefig(plot_dir + hist_name + ".png", dpi=360, format="png")
					plt.close(ax.figure)
				except KeyError:
					histograms[hist_name] = hist.Hist("entries",
											hist.Cat("dataset", "Dataset"),
											hist.Bin("values", varname, np.array(edges) ) )
					histograms[hist_name].fill(dataset=dataset, values=get_bin_centers(edges), weight=contents)
		json_file.close()
