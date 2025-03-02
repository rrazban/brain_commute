"""
Check out commute time - fucntional connectivity performance
for subgroups of UK Biobank individuals diagnosed with mental
health disorders.

"""

import numpy as np
import sys, glob, os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest, spearmanr
import seaborn as sns

sys.path.append('utlts/')
from analysis_tools import get_structure, delete_indi, print_stats, no_edges, correlation, delete_empty_index
from structure_metrics import commute_time, deconstruct_cov


plt.rcParams.update({'font.size': 14})



def run(fmri_files, atlas, dataset):
	subs = []
	ct_fxn = []
	ct_topfxn = []

	for fmri in fmri_files:
		sub, structure = get_structure(fmri, dataset, atlas, 'density')
		raw_fxn = np.loadtxt(fmri, delimiter=',')
		fc = np.corrcoef(raw_fxn.T)    #fxn is captured by correlation of time-series between two regions

		structure, fc = delete_empty_index(structure, fc, atlas)

		indi = no_edges(structure)
		structure = delete_indi(indi, structure)
		fc = delete_indi(indi, fc)

		fc_top_mode = deconstruct_cov(fc, 1)

		upper_tri = np.triu_indices(structure.shape[0], k=1)	#dont include diagonal elements, also dont double count
	
		cts = commute_time(structure)

		ct_fxn.append(-correlation(cts, fc, upper_tri))
		ct_topfxn.append(-correlation(cts, fc_top_mode, upper_tri))

		subs.append(int(sub))

	return subs, ct_fxn, ct_topfxn 


def make_dataframe(outputs, df, mental_status, FC_mode):
	for corr in df:
		outputs.append([corr, mental_status, FC_mode])
	return outputs


def plotout(df):
	ax= sns.violinplot(data=df, x='disorder status', y='corr', hue='FC modes', inner='quart', split=True)
	sns.move_legend(ax, loc='lower right')

	plt.xticks(np.arange(3), ['mental disorder\n($N=${0})'.format(len(df_overlap)), 'brain disease\n($N=${0})'.format(len(df_overlap_neuro)), 'healthy\n($N=${0})'.format(len(df_healthy))])
	plt.title('UK Biobank')
	plt.ylabel('$\\rho(-$commute time, FC$)$')
	plt.xlabel('')
	plt.grid()
	plt.tight_layout()

	plt.show()


def compare_distributions(df):
	fc_mode = 'top'

	for group in ['healthy', 'brain disease', 'mental disorder']:
		pre_ct = df[df['disorder status']==group]
		ct = pre_ct[pre_ct['FC modes']==fc_mode]['corr']

		for group2 in ['brain disease', 'mental disorder']:
			pre = df[df['disorder status']==group2]
			metric = pre[pre['FC modes']==fc_mode]['corr']

			kval, pval = kstest(ct, metric)
			print(group, group2, kval, pval)


if __name__ == "__main__":
	atlas = 'DesKi'

	fmri_files = glob.glob('data/ukb/{0}/fMRI/*'.format(atlas))

	subs, ct_fxn, ct_topfxn = run(fmri_files, atlas, 'ukb')

	df_data = pd.DataFrame({'id': subs, 'corr':ct_fxn, 'corr_top': ct_topfxn})

	group_file = 'data/ukb/subject_info/mental_disorder'
	df_group = pd.read_csv('{0}.csv'.format(group_file))
	df_overlap = df_data.merge(df_group,on='id')

	neuro_file = 'data/ukb/subject_info/neuro_disease'
	df_neuro = pd.read_csv('{0}.csv'.format(neuro_file))
	df_overlap_neuro = df_data.merge(df_neuro,on='id')

	df_healthy = df_data[~df_data['id'].isin(df_overlap['id'])]
	df_healthy = df_healthy[~df_healthy['id'].isin(df_overlap_neuro['id'])]

#make dataframe to utilize seaborn's violinplotting
	outputs = []
	outputs = make_dataframe(outputs, df_overlap['corr'], 'mental disorder', 'all')
	outputs = make_dataframe(outputs, df_overlap['corr_top'], 'mental disorder', 'top')

	outputs = make_dataframe(outputs, df_overlap_neuro['corr'], 'brain disease', 'all')
	outputs = make_dataframe(outputs, df_overlap_neuro['corr_top'], 'brain disease', 'top')

	outputs = make_dataframe(outputs, df_healthy['corr'], 'healthy', 'all')
	outputs = make_dataframe(outputs, df_healthy['corr_top'], 'healthy', 'top')


	df = pd.DataFrame(outputs, columns = ['corr', 'disorder status', 'FC modes'])
	
	compare_distributions(df)
	plotout(df)


