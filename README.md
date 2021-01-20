# ttHbb
Framework for accelerated ttH(bb) columnar analysis with Coffea (https://coffeateam.github.io/coffea/) on flat centrally produced nanoAOD samples.
## Getting started
To run the preliminary version of the analysis script:
~~~
python dilepton_analysis.py --sample 2017 --machine lxplus
~~~
The executor parameters can be specified as arguments:
~~~
python dilepton_analysis.py --sample 2017 --machine lxplus --workers 10 --chunksize 30000 --maxchunks 25
~~~

