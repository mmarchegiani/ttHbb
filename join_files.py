import argparse
from coffea.util import save, load
import coffea.hist as hist
import os

parser = argparse.ArgumentParser(description='Join .coffea files into a single file')
parser.add_argument('-d', '--dir', type=str, help='Directory with files to join')

args = parser.parse_args()
if not args.dir.endswith("/"):
    args.dir = args.dir + "/"

files = [args.dir + file for file in os.listdir(args.dir) if '.coffea' in file]

output_split = []
for file in files:
    output = load(file)
    output_split.append(output)

accumulator = output_split[0]
histograms = output_split[0].keys()
for histname in histograms:
    for output in output_split:
        accumulator[histname].add(output[histname])

outfile = '_'.join(files[0].split("_")[:-1]) + ".coffea"
save(accumulator, outfile)
print(f"Saving output to {outfile}")
