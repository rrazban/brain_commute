"""
Compare commute time to other structure-based 
metrics trying to predict functional connectivty.
Across all individuals from UK Biobank or 
HCP Young Adult dataset.

"""


import numpy as np
import sys, glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr, kstest
import seaborn as sns

sys.path.append('utlts/')
from analysis_tools import get_structure, delete_indi, print_stats, no_edges, correlation, delete_empty_index
from structure_metrics import commute_time, deconstruct_cov, weighted_communicability, search_information


plt.rcParams.update({'font.size': 14})



def run(fmri_files, which):
	subs = []
	outputs = []

	for fmri in fmri_files:
		sub, structure = get_structure(fmri, dataset, which, 'density')
		_, structure_length = get_structure(fmri, dataset, which, 'length')	#needed for search_information
		raw_fxn = np.loadtxt(fmri, delimiter=',')

		fc = np.corrcoef(raw_fxn.T)    #fxn is captured by correlation of time-series between two regions

		structure, fc = delete_empty_index(structure, fc, which)
		structure_length, _ = delete_empty_index(structure_length, fc, which)	#dont delete fc elements twice

		indi = no_edges(structure)
		structure = delete_indi(indi, structure)
		structure_length = delete_indi(indi, structure_length)
		fc = delete_indi(indi, fc)

		fc_top_mode = deconstruct_cov(fc, 1)
		upper_tri = np.triu_indices(structure.shape[0], k=1)	#dont include diagonal elements, also dont double count

		outputs.append([correlation(structure, fc, upper_tri), 'all', 'adjacency'])
		outputs.append([correlation(structure, fc_top_mode, upper_tri), 'top', 'adjacency'])

		sis = search_information(structure, structure_length)
		outputs.append([correlation(-sis, fc, upper_tri), 'all', 'search info'])
		outputs.append([correlation(-sis, fc_top_mode, upper_tri), 'top', 'search info'])

		cmys = weighted_communicability(structure)
		outputs.append([correlation(cmys, fc, upper_tri), 'all', 'communicability'])
		outputs.append([correlation(cmys, fc_top_mode, upper_tri), 'top', 'communicability'])

		cts = commute_time(structure)

		outputs.append([-correlation(cts, fc, upper_tri), 'all', 'commute time'])
		outputs.append([-correlation(cts, fc_top_mode, upper_tri), 'top', 'commute time'])


		subs.append(int(sub))

	return outputs

def plotout(df, title):
	plt.figure().set_figwidth(8.5)
	ax= sns.violinplot(data=df, x='metric', y='corr', hue='FC modes', inner='quart', split=True)
	sns.move_legend(ax, loc='upper left')


	plt.xticks(np.arange(4), ['adjacency', '$-$search info','$-$communicability', 'commute time'])
	plt.ylim([-0.1, 0.7])

	plt.xlabel('metrics calculated based on structure')
	plt.ylabel('$\\rho($structure, FC$)$')
	plt.title(title)
	plt.grid()
	plt.tight_layout()
	plt.show()


def compare_distributions(df):
#print out Kolmogorov-Smirnoff between distribution of correlations

	for metric1 in ['commute time', 'adjacency', 'search info', 'communicability']:

		pre_ct = df[df['metric']==metric1]
		ct = pre_ct[pre_ct['FC modes']=='all']['corr']

		for metric_name in ['adjacency', 'search info', 'communicability']:
			pre = df[df['metric']==metric_name]#[df['FC modes']=='all']['corr']
			metric = pre[pre['FC modes']=='all']['corr']

			kval, pval = kstest(ct, metric)
			print(metric1, metric_name, kval, pval)


if __name__ == "__main__":
	atlas = 'DesKi'	#Talairach, DesKi
	dataset = 'hcp_ya_100'	#ukb, hcp_ya_100

	fmri_files = glob.glob('data/{0}/{1}/fMRI/*'.format(dataset, atlas))

	outputs = run(fmri_files, atlas)
	df = pd.DataFrame(outputs, columns = ['corr', 'FC modes', 'metric'])


	if dataset=='ukb':
		title = 'UK Biobank ($N=${0} subjects)'.format(len(fmri_files))
	elif dataset=='hcp_ya_100':
		if atlas=='Glasser':
			title = 'HCP Young Adult, probabilistic tractography, Glasser atlas ($N=${0} subjects)'.format(len(ct))	#data from Rosen et al. 2021	
		else:
			title='HCP Young Adult ($N=${0} scans)'.format(len(fmri_files))
	plotout(df, title)

#print mean and std of correlations across individuals
#	pre_group = df[df['metric']=='commute time']
#	print_stats(pre_group[pre_group['FC modes']=='all'])
#	print_stats(pre_group[pre_group['FC modes']=='top'])

#print Kologomorov-Smirnov test summary for pairwise distribution comparision
#	compare_distributions(df)

