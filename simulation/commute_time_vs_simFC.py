"""
Code to see if commute time can 
capture timeseries' correlations
from a toy Ising model.
In other words, can structure
dictate function? 

"""

import numpy as np
import sys
import matplotlib.pyplot as plt


sys.path.append('../utlts/')
from analysis_tools import delete_indi, no_edges, delete_empty_index
from structure_metrics import commute_time, deconstruct_cov

sys.path.append('../')
from commute_time_vs_FC import plotter


if __name__ == "__main__":

	atlas = 'DesKi'
	sub_id = '1000366'	#ukb subject
	lam = 6.0
	

	out_file = 'output/{0}/{1}/bold_lam{2}_sim0.csv'.format(atlas, sub_id, lam)

	raw_fxn = np.loadtxt(out_file, delimiter=',')
	fc = np.corrcoef(raw_fxn.T)    #fxn is captured by correlation of time-series between two regions

	dmri_file = '../data/ukb/{0}/dMRI/{1}_20250_2_0_density.txt'.format(atlas, sub_id)       
	dmri_file_len = '../data/ukb/{0}/dMRI/{1}_20250_2_0_length.txt'.format(atlas, sub_id)       
	structure = np.loadtxt(dmri_file)
	structure_len = np.loadtxt(dmri_file_len)
	np.fill_diagonal(structure,0)   #very important, otherwise self-loops are included in calculation of adjacency matrix

	structure, _ = delete_empty_index(structure, structure, atlas)	#second input in fxn should be fc, but not relevant here
	structure_len, _ = delete_empty_index(structure_len, structure, atlas)	#second input in fxn should be fc, but not relevant here

	indi = no_edges(structure)
	structure = delete_indi(indi, structure)
	structure_len = delete_indi(indi, structure_len)

	title='UK Biobank subject {0}, $\lambda=${1}'.format(sub_id, lam)
	plotter(title, commute_time(structure), fc, 'commute time (dMRI)', 'functional connectivity (Ising)')

	modes =2
	plotter(title, commute_time(structure),deconstruct_cov(fc, modes), 'commute time (dMRI)', 'top {0} modes of FC (Ising)'.format(modes))
