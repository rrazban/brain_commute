"""
Compare mean values of commute time 
and functional connectivity seperately
to see possible drivers of the 
commute time - functional connectivity 
correlation with respect to disorder.
Across all individuals from UK Biobank.

"""

import numpy as np
import sys, glob, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from disorder_v_performance import compare_distributions 

sys.path.append('utlts/')
from analysis_tools import get_structure, delete_indi, print_stats, no_edges, correlation, delete_empty_index
from structure_metrics import commute_time


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
	
		cts = commute_time(structure)

		ct_fxn.append(np.mean(cts))
		ct_topfxn.append(np.mean(fc))

		subs.append(int(sub))

	return subs, ct_fxn, ct_topfxn 

def make_dataframe(outputs, df, mental_status, FC_mode):
	for corr in df:
		outputs.append([corr, mental_status, FC_mode])
	return outputs


def plotout(df_merged1, df_merged2):
	palette_tab10 = sns.color_palette("Set3")

	fig, ax0 = plt.subplots()


	sns.violinplot(df_merged1, x="disorder status", y='corr', hue="FC_modes",hue_order=['all','top'], ax=ax0, inner='quart', split=True, dodge=True, palette='Set3', legend=False)

	ax1 = ax0.twinx()

	hue_order = df_merged.FC_modes.unique()

	sns.violinplot(df_merged2, x="disorder status", y='corr', hue="FC_modes", hue_order=['all','top'], ax=ax1, inner='quart', split=True, dodge=True, palette='Set3') 



	handles, labels = ax1.get_legend_handles_labels()
	ax1.legend(handles = handles, labels = ["$\\langle $commute time$\\rangle $","$\\langle $FC$\\rangle $"])

	plt.xticks(np.arange(3), ['mental disorder\n($N=${0})'.format(len(df_overlap)), 'brain disease\n($N=${0})'.format(len(df_overlap_neuro)), 'healthy\n($N=${0})'.format(len(df_healthy))])
	plt.title('UK Biobank')
	ax0.set_ylabel('$\\langle$commute time$\\rangle $', color=palette_tab10[0], fontweight='bold')
	ax1.set_ylabel('$\\langle$FC$\\rangle $', color=sns.color_palette()[-2], fontweight='bold')

	ax0.set_xlabel('')
	plt.grid()
	plt.tight_layout()

	plt.show()


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

	df_merged = pd.DataFrame(outputs, columns = ['corr', 'disorder status', 'FC_modes'])


	df_merged1 = df_merged[df_merged['FC_modes']=='all']
	df_merged2 = df_merged[df_merged['FC_modes']=='top']


	plotout(df_merged1, df_merged2)

#	compare_distributions(df)
