"""
Compare commute time to other structure-based 
metrics trying to predict functional connectivty.
Across 100 repeated simulations for the structure
of one arbitrary individual from UK Biobank.

"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd


sys.path.append('../utlts/')
from analysis_tools import get_structure, delete_indi, print_stats, no_edges, correlation, delete_empty_index
from structure_metrics import commute_time, deconstruct_cov, weighted_communicability, search_information

sys.path.append('../')
from performance_across_ppl import plotout, compare_distributions



def run(structure,structure_length, sub_id, lam, atlas):
	outputs = []

	for i in range(100):
		fxn_file = 'output/{0}/{1}/bold_lam{2}_sim{3}.csv'.format(atlas, sub_id, lam, i)
		raw_fxn = np.loadtxt(fxn_file, delimiter=',')
		fc = np.corrcoef(raw_fxn.T)    #fxn is captured by correlation of time-series between two regions

		fc_top_mode = deconstruct_cov(fc, 2)


		upper_tri = np.triu_indices(structure.shape[0], k=1)	#dont include diagonal elements, also dont double count
	
	
		outputs.append([correlation(structure, fc, upper_tri), 'all', 'adjacency'])
		outputs.append([correlation(structure, fc_top_mode, upper_tri), 'top 2', 'adjacency'])

		sis = search_information(structure, structure_length)
		outputs.append([correlation(-sis, fc, upper_tri), 'all', 'search info'])
		outputs.append([correlation(-sis, fc_top_mode, upper_tri), 'top 2', 'search info'])

		cmys = weighted_communicability(structure)
		outputs.append([correlation(cmys, fc, upper_tri), 'all', 'communicability'])
		outputs.append([correlation(cmys, fc_top_mode, upper_tri), 'top 2', 'communicability'])

		cts = commute_time(structure)
		outputs.append([-correlation(cts, fc, upper_tri), 'all', 'commute time'])
		outputs.append([-correlation(cts, fc_top_mode, upper_tri), 'top 2', 'commute time'])


	return outputs




if __name__ == "__main__":
	atlas = 'DesKi'
	sub_id = '1000366'
	lam = 6.0
	

	dmri_file = '../data/ukb/{0}/dMRI/{1}_20250_2_0_density.txt'.format(atlas, sub_id)       
	dmri_length_file = '../data/ukb/{0}/dMRI/{1}_20250_2_0_length.txt'.format(atlas, sub_id)       
	structure = np.loadtxt(dmri_file)
	structure_length = np.loadtxt(dmri_length_file)
	np.fill_diagonal(structure,0)   #very important, otherwise self-loops are included in calculation of adjacency matrix
	np.fill_diagonal(structure_length,0)

	structure, _ = delete_empty_index(structure, structure, atlas)	#second input in fxn should be fc, but not relevant here
	structure_length, _ = delete_empty_index(structure_length, structure_length, atlas)	#second input in fxn should be fc, but not relevant here

	indi = no_edges(structure)
	structure = delete_indi(indi, structure)
	structure_length = delete_indi(indi, structure_length)


	outputs = run(structure,structure_length, sub_id, lam, atlas)
	df = pd.DataFrame(outputs, columns = ['corr', 'FC modes', 'metric'])

	title = 'UK Biobank subject {0}, $\lambda=${1}, {2} repeats'.format(sub_id, lam, 100)
	plotout(df, title)

#print Kologomorov-Smirnov test summary for pairwise distribution comparision
	compare_distributions(df)
