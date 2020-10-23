#Description:

Source code enabling replication of experiments in the paper
 by M. Romaszewski, P.Glomb, M. Cholewa, A. Sochan  
`**A Dataset for Evaluating Blood Detection in Hyperspectral Images**'
preprint: https://arxiv.org/abs/2008.10254

#The dataset associated with that source code: 

The dataset is available online:
https://zenodo.org/deposit/3984905

#Implementation:

Experiments were implemented in Python 3.6.9 using libraries:
numpy 1.16.4, scipy 1.3.1, scikit-learn 0.22.1, matplotlib 3.2.2

#Usage:

<ul>
<li> Ensure that the dataset patch in ds_load.py is correct
<li> Run experiments first, before generating results (unless it is dataset presentation)
</ul>

Files:
<ul>
<li> ds_load.py - data loading and utility functions
<li> results_util.py - implements detection performance measures
<li> saveload.py - universal serialisation functions
<li> target_detectors.py - MF detector implementation
<li> two_stage_detector.py - implementation of the TSMF from the paper
<li> experiment_classification.py - performs simple classification experiment
<li> experiment_detection.py - performs detection experiments
<li> fig_dataset.py - figures with dataset presentsion
<li> fig_detection.py - figures with detection results
<li> fig_example.py - detection examle with more detailed presentation
<li> fig_means.py - presentation of spectra before/after applying TSMF
<li> fig_no_pixels.py - figures visualising impact of the no. pixels on AUC
</ul> 

Warning:
experiments use the external blood library described in: 
**Majda, Alicja, et al. "Hyperspectral imaging and multivariate analysis in the dried blood spots investigations"  Applied Physics A 124.4 (2018): 312.**
Unless the library is provided, its spectra will replaced with a spectrum from the dataset.


#License:

The code is licensed under GNU General Public License v.3.0