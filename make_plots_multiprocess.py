import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib import cm
from matplotlib import colors
import mplhep as hep
from coffea.util import load
from coffea.hist import plot
import coffea.hist as hist
import copy
import os
import sys
import time
import multiprocessing

def cms(ax, fontsize=20, hist2d=False):
    gap = 0.08
    if hist2d:
        gap = 0.1
    plt.text(0., 1., "CMS",
                  fontsize=fontsize,
                  horizontalalignment='left',
                  verticalalignment='bottom',
                  transform=ax.transAxes,
                  weight="bold"
                 )
    plt.text(gap, 1., "Simulation Preliminary",
                  fontsize=fontsize,
                  horizontalalignment='left',
                  verticalalignment='bottom',
                  transform=ax.transAxes,
                  style='italic',
                 )
    return

parser = argparse.ArgumentParser(description='Plot histograms from coffea file')
parser.add_argument('-i', '--input', type=str, help='Input histogram filename', required=True)
parser.add_argument('-o', '--output', type=str, default='plots_coffea/', help='Output directory', required=False)
parser.add_argument('-w', '--workers', type=int, default=32, help='Number of workers', required=False)
#parser.add_argument('--data', type=str, default='BTagMu', help='Data sample name')

args = parser.parse_args()

accumulator = load(args.input)

if args.output == parser.get_default('output'):
    args.output = "plots/" + (args.input.split("/")[-1]).split(".coffea")[0]
else:
    if not "plots/" in args.output:
        args.output = "plots/" + args.output
if not args.output.endswith('/'):
    args.output = args.output + '/'

data_err_opts = {
    'linestyle': 'none',
    'marker': '.',
    'markersize': 10.,
    'color': 'k',
    'elinewidth': 1,
}

plt.style.use(hep.style.ROOT)
if not os.path.exists(args.output):
    os.makedirs(args.output)

for histname in accumulator:
    if not 'hist' in histname: continue
    if any([histname.startswith(s) for s in ["hist_muons", "hist_goodmuons", "hist_electrons", "hist_goodelectrons", "hist_jets", "hist_goodjets"]]): continue
    hist1d = not 'hist2d' in histname

    h = accumulator[histname]

    if hist1d:
        for scale in ["linear", "log"]:
            print("Plotting", histname, "scale:", scale)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,9))
            cms(ax)
            #fig, (ax, rax) = plt.subplots(2, 1, figsize=(12,12), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
            #fig.subplots_adjust(hspace=.07)
            plot.plot1d(h, ax=ax, legend_opts={'loc':1})
            #plot.plot1d(h[args.data], ax=ax, legend_opts={'loc':1}, density=args.dense, error_opts=data_err_opts, clear=False)
            #plot.plotratio(num=h[args.data].sum('dataset'), denom=h[[dataset for dataset in datasets if args.data not in dataset]].sum('dataset'), ax=rax,
            #               error_opts=data_err_opts, denom_fill_opts={}, guide_opts={} )#, unc='num')
            if scale == "log":
                ax.set_ylim(0.1, None)
            ax.set_yscale(scale)
            #rax.set_ylabel('data/MC')
            #rax.set_yscale(args.scale)
            #rax.set_ylim(0.000001,2)
            filepath = args.output + histname + ".png"
            if scale != "linear":
                filepath = filepath.replace(".png", "_" + scale + ".png")
            print(f"Saving plot to {filepath}")
            plt.savefig(filepath, dpi=300, format="png")
            plt.close(fig)
    else:
        print("Plotting", histname)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,9))
        my_cmap = copy.copy(cm.get_cmap("viridis"))
        my_cmap.set_under('w',1)
        histo = h.sum("dataset")
        hist.plot2d(histo, xaxis='x', ax=ax, patch_opts={'cmap' : my_cmap, 'norm' : colors.Normalize(vmin=0.01)})
        cms(ax, hist2d=True)
        filepath = args.output + histname + ".png"
        print(f"Saving plot to {filepath}")
        plt.savefig(filepath, dpi=300, format="png")
        plt.close(fig)
