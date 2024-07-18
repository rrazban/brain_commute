"""
Test whether commute time from an individual's 
structure can capture functional connectivities
from functional MRI data. All possible edges
are considered. Individuals from the UK Biobank 
or HCP Young Adult dataset can be chosen.

"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import gaussian_kde


sys.path.append('utlts/')
from analysis_tools import get_structure, delete_indi, no_edges, correlation, delete_empty_index, visualize_matrix
from structure_metrics import commute_time, deconstruct_cov


plt.rcParams.update({'font.size': 14})

def plotter(sub_id, xs, corrmat, xlabel, ylabel = 'functional connectivity (fMRI)'):
	upper_tri = np.triu_indices(corrmat.shape[0], k=1)  #get rid of double counting for symmetric matrices
	xs = xs[upper_tri]
	ys = corrmat[upper_tri]

	r,pvalue = spearmanr(xs, ys)

	# Calculate the point density
	xy = np.vstack([xs,ys])
	z = gaussian_kde(xy)(xy)

	plt.scatter(xs, ys, c=z, label="$\\rho=${0:.2f} ({1:.2E})\n$n=${2} edges".format(r, pvalue, len(xs)))
	plt.title('{0} subject {1}, {2} atlas'.format(dataset, sub_id, atlas))
#	plt.title('UK Biobank subject {0}'.format(sub_id))


	plt.legend()
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	dataset = 'ukb'	#ukb, hcp_ya_100
	atlas = 'DesKi'	#DesKi, Talairach (only for ukb)
	
	sub_id = '1000366'	#ukb
#	sub_id = 654754	#hcp_ya

	if dataset=='ukb':
		fmri = 'data/{0}/{1}/fMRI/{2}_20227_2_0.csv'.format(dataset, atlas, sub_id)
	elif dataset=='hcp_ya_100':
		fmri = 'data/{0}/{1}/fMRI/{2}_run-0.csv'.format(dataset, atlas, sub_id)

	raw_fxn = np.loadtxt(fmri, delimiter=',')
	fc = np.corrcoef(raw_fxn.T)    #fxn is captured by correlation of time-series between two regions

	_, structure = get_structure(fmri, dataset, atlas, 'density')

	structure, fc = delete_empty_index(structure, fc, atlas)

	indi = no_edges(structure)
	structure = delete_indi(indi, structure)
	fc = delete_indi(indi, fc)

	plotter(sub_id, commute_time(structure), fc, 'commute time (dMRI)')

	modes =1 
	plotter(sub_id, commute_time(structure),deconstruct_cov(fc, modes), 'commute time (dMRI)', 'top {0} mode(s) of FC (fMRI)'.format(modes))
