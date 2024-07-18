"""
Check out commute time - fucntional connectivity performance
for subgroups of UK Biobank individuals diagnosed with mental
health disorders.

"""

import numpy as np
import sys, glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest
import seaborn as sns

sys.path.append('utlts/')
from analysis_tools import get_structure, delete_indi, print_stats, no_edges, correlation, delete_empty_index
from structure_metrics import laplacian, commute_time, deconstruct_cov, weighted_communicability, search_information


plt.rcParams.update({'font.size': 14})



def run(fmri_files, atlas):
	subs = []
	ct_fxn = []
	ct_topfxn = []

	for fmri in fmri_files:
		sub, structure = get_structure(fmri, 'ukb', atlas, 'density')
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

def compare_distributions(df):
	fc_mode = 'all'
	pre_ct = df[df['disorder status']==fc_mode]
	
	ct = pre_ct[pre_ct['FC modes']==fc_mode]['corr']

	for group in ['brain disease', 'mental disorder']:
		pre = df[df['disorder status']==group]
		metric = pre[pre['FC modes']==fc_mode]['corr']

		kval, pval = kstest(ct, metric)
		print(group, kval, pval)


if __name__ == "__main__":
	atlas = 'DesKi'

	fmri_files = glob.glob('data/ukb/{0}/fMRI/*'.format(atlas))

	subs, ct_fxn, ct_topfxn = run(fmri_files, atlas)

	df_data = pd.DataFrame({'id': subs, 'corr':ct_fxn, 'corr_top': ct_topfxn})

	group_file = 'data/ukb/subject_info/mental_disorder'
	df_group = pd.read_csv('{0}.csv'.format(group_file))
	df_overlap = df_data.merge(df_group,on='id')

	neuro_file = 'data/ukb/subject_info/neuro_disease'
	df_neuro = pd.read_csv('{0}.csv'.format(neuro_file))
	df_overlap_neuro = df_data.merge(df_neuro,on='id')


#make dataframe to utilize seaborn's violinplotting
	outputs = []
	outputs = make_dataframe(outputs, df_overlap['corr'], 'mental disorder', 'all')
	outputs = make_dataframe(outputs, df_overlap['corr_top'], 'mental disorder', 'top')

	outputs = make_dataframe(outputs, df_overlap_neuro['corr'], 'brain disease', 'all')
	outputs = make_dataframe(outputs, df_overlap_neuro['corr_top'], 'brain disease', 'top')

	outputs = make_dataframe(outputs, df_data['corr'], 'all', 'all')
	outputs = make_dataframe(outputs, df_data['corr_top'], 'all', 'top')


	df = pd.DataFrame(outputs, columns = ['corr', 'disorder status', 'FC modes'])

	ax= sns.violinplot(data=df, x='disorder status', y='corr', hue='FC modes', inner='quart', split=True)
	sns.move_legend(ax, loc='lower right')

	plt.xticks(np.arange(3), ['mental disorder\n($N=${0})'.format(len(df_overlap)), 'brain disease\n($N=${0})'.format(len(df_overlap_neuro)), 'all\n($N=${0})'.format(len(df_data))])
	plt.title('UK Biobank')
	plt.ylabel('$\\rho(-$commute time, FC$)$')
#	plt.ylim([-0.2, 0.7])
	plt.xlabel('')
	plt.grid()
	plt.tight_layout()

	compare_distributions(df)
	plt.show()
