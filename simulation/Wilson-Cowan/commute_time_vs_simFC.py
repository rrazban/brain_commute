"""
Code to see if commute time can 
capture timeseries' correlations
from Wilson-Cowan model.
In other words, does structure
dictate function? 

"""

import numpy as np
import sys

sys.path.append('../../utlts/')
from structure_metrics import commute_time, deconstruct_cov

sys.path.append('../../')
from commute_time_vs_FC import plotter


if __name__ == "__main__":

	out_dir = '/Users/zero622/Desktop/clean_code_ising/laplacian/wilson_cowan'
	other_dir = '/Users/zero622/Desktop/wilson_cowan'

	out_file='runCC0.06.csv'

	raw_fxn = np.loadtxt(out_file, delimiter=',')

	fc = np.corrcoef(raw_fxn.T)    #fxn is captured by correlation of time-series between two regions


	dmri_file = '{0}/tvb_96_connectivity_matrix.npy'.format(out_dir)       
	adjacency = np.load(dmri_file)
	np.fill_diagonal(adjacency,0)   #very important, otherwise self-loops are included in calculation of adjacency matrix

	delete_regions = [45, 47, 93, 95]   #Botond removed these regions from the simulation because little oscilattions seen
	adjacency = np.delete(adjacency, delete_regions, 0)
	adjacency = np.delete(adjacency, delete_regions, 1)

	title = 'The Virtual Brain, CC$=0.06$'
#	plt.xlim([0, 1300])
#	plt.ylim([-0.75, 1.1])
#	plotter(adjacency, fc, 'adjacency (CoCoMac/dMRI)')

	plotter(title, commute_time(adjacency), fc, 'commute time (CoCoMac/dMRI)', 'functional connectivity (Wilson-Cowan)')

	modes =2
	plotter(title, commute_time(adjacency),deconstruct_cov(fc, modes), 'commute time (CoCoMac/dMRI)', 'top two modes of FC (Wilson-Cowan)')
