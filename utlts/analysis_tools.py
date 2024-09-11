"""
Useful functions to run analyses, print out statistics
or visualize matrices.

"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kstest


def visualize_matrix(mat, label):
	plt.imshow(mat)
	cbar = plt.colorbar()
	cbar.set_label(label)
	ax = plt.gca()
	ax.xaxis.set_ticks_position('bottom')
	plt.tight_layout()
	plt.show()


def delete_empty_index(structure, fc, atlas):
		if atlas=='Talairach':
			wm_index = [512, 513, 516, 1030, 1031, 1032, 1019, 526, 527, 530, 534, 536, 537, 540, 1053, 542, 544, 545, 546, 37, 551, 1064, 557, 46, 559, 1072, 1074, 564, 53, 565, 566, 1079, 571, 572, 1083, 1084, 1085, 577, 590, 593, 595, 597, 598, 601, 91, 92, 93, 98, 101, 104, 620, 628, 120, 641, 132, 135, 137, 139, 654, 655, 656, 657, 666, 669, 162, 167, 691, 698, 711, 200, 712, 210, 724, 217, 222, 738, 742, 745, 234, 747, 750, 751, 240, 241, 754, 755, 759, 760, 761, 765, 767, 770, 771, 262, 789, 279, 797, 287, 812, 304, 826, 827, 318, 840, 841, 331, 334, 855, 345, 858, 349, 865, 867, 885, 886, 888, 890, 388, 389, 391, 398, 400, 917, 926, 932, 427, 428, 434, 947, 973, 983, 472, 985, 992, 994, 995, 488, 490, 493, 494, 496, 1012, 503, 505, 1018, 507,0]	#added 0 background since fmri auto removes in output cuz doesnt want to be part of standardization

			structure = delete_indi(wm_index, structure)

			zerofmri_i = [44,94, 132, 135, 167, 469, 480, 654]
			fc = delete_indi(zerofmri_i, fc)
			structure = delete_indi(zerofmri_i, structure)


		elif atlas=='DesKi':
			empty_index = [2, 40, 83, 0]	#added 0 background
			structure = delete_indi(empty_index, structure)


		return structure, fc


def correlation(xs, ys, upper_tri):
	r, pval = spearmanr(xs[upper_tri], ys[upper_tri])
	return r	


def get_structure(fmri_file, dataset, atlas, which):	#maybe just have which and atlas
	sub = os.path.basename(fmri_file)
	sub = sub[:sub.index('_')]	#need this for multiple fMRI runs
#	sub = sub[:sub.index('.csv')]

	if dataset=='ukb':
		dmri_file = 'data/ukb/{0}/dMRI/{1}_20250_2_0_{2}.txt'.format(atlas, sub, which)
	else:
#		dmri_file = 'data/{0}/{1}/dMRI/{2}_{3}.txt'.format(dataset, atlas, sub, which)
		dmri_file = 'data/{0}/{1}/dMRI/probabilistic/density1/{2}_{3}.txt'.format(dataset, atlas, sub, which)

	structure = np.loadtxt(dmri_file)
	np.fill_diagonal(structure,0)	#very important, otherwise self-loops are included in calculation of adjacency matrix
	return sub, structure


def delete_indi(indi, mat):
	mat = np.delete(mat, indi, 0) 
	mat = np.delete(mat, indi, 1)
	return mat 

def plotter(xs, corrmat, xlabel, ylabel = 'functional connectivity'):
	#no one seems to use... every script has their own special one

	upper_tri = np.triu_indices(corrmat.shape[0], k=1)
	xs = xs[upper_tri]
	corrmat=corrmat[upper_tri]	#get rid of diagonal of 0s from consideration

	r,pvalue = spearmanr(corrmat.flatten(), xs.flatten())
	plt.title('subject id: {0}'.format(sub_id))
	plt.scatter(xs, corrmat, label="$\\rho=${0:.2f} ({1:.2E})\n$n=${2} edges".format(r, pvalue, len(xs)))
	plt.legend()
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.tight_layout()
	plt.show()

def no_edges(mat):
    exclude = list(np.where(~mat.any(axis=1))[0])    #indices of excluded regions cuz no edges formed
    return exclude 


def print_stats(data):
	print(np.mean(data), np.std(data))


