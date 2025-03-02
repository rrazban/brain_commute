"""
Compare mean values of commute time 
and functional connectivity seperately
to see possible drivers of the 
commute time - functional connectivity 
correlation with respect to age.
Across all individuals from UK Biobank or 
HCP Young Adult dataset.

"""


import os
import numpy as np
import sys, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

from age_v_performance import hcp_assign_age_group, ukb_assign_age_group 


sys.path.append('utlts/')
from analysis_tools import get_structure, delete_indi, print_stats, no_edges, correlation, delete_empty_index
from structure_metrics import commute_time

plt.rcParams.update({'font.size': 14})



def run(fmri_files, atlas, dataset):
	subs = []
	outputs = []

	for fmri in fmri_files:
		sub, structure = get_structure(fmri, dataset, atlas, 'density')
		_, structure_length = get_structure(fmri, dataset, atlas, 'length')	#needed for search_information
		raw_fxn = np.loadtxt(fmri, delimiter=',')

		fc = np.corrcoef(raw_fxn.T)    #fxn is captured by correlation of time-series between two regions

		structure, fc = delete_empty_index(structure, fc, atlas)
		structure_length, _ = delete_empty_index(structure_length, fc, atlas)	#dont delete fc elements twice

		indi = no_edges(structure)
		structure = delete_indi(indi, structure)
		structure_length = delete_indi(indi, structure_length)
		fc = delete_indi(indi, fc)

		cts = commute_time(structure)
		outputs.append([int(sub), np.mean(cts), 'all', 'commute time'])
		outputs.append([int(sub), np.mean(fc), 'top', 'commute time'])


		subs.append(int(sub))
	
	return outputs


def plotout(df_merged1, df_merged2, title):
    rval, pval = (spearmanr(df_merged1['age'], df_merged1['corr']))
    rval2, pval2 = (spearmanr(df_merged2['age'], df_merged2['corr']))
	

    palette_tab10 = sns.color_palette("Set3")
    fig, ax0 = plt.subplots(figsize=(8.5,4.8))


    sns.violinplot(df_merged1, x="age_group", y='corr', hue="FC_modes",hue_order=['all','top'], ax=ax0, inner='quart', split=True, dodge=True, palette='Set3', legend=False)

    ax1 = ax0.twinx()

    hue_order = df_merged.FC_modes.unique()

    sns.violinplot(df_merged2, x="age_group", y='corr', hue="FC_modes", hue_order=['all','top'], ax=ax1, inner='quart', split=True, dodge=True, palette='Set3') 


    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles = handles, labels = ["$\\langle $commute time$\\rangle $, $\\rho=${0:.2f} ({1:.1E})".format(rval, pval),"$\\langle $FC$\\rangle $, $\\rho=${0:.2f} ({1:.1E})".format(rval2, pval2)])

    ax0.set_ylabel('$\\langle$commute time$\\rangle $', color=palette_tab10[0], fontweight='bold')
    ax1.set_ylabel('$\\langle$FC$\\rangle $', color=sns.color_palette()[-2], fontweight='bold')
    ax0.set_xlabel('age group')
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    plt.show()
	


if __name__ == "__main__":
	atlas = 'DesKi'
	dataset = 'hcp_ya_100'	#ukb, hcp_ya_100
	fmri_files = glob.glob('data/{0}/{1}/fMRI/*csv'.format(dataset, atlas))

    # Run analysis to get subject IDs and commute time matrices
	outputs = run(fmri_files, atlas, dataset)

    # Load the all.csv file with a comma separator
	all_data = pd.read_csv('data/{0}/subject_info/phenotypes.csv'.format(dataset), sep=',')	#for ukb


    # Ensure `id` is integer and `age` is consistent
	all_data['id'] = all_data['id'].astype(int)
	all_data['age'] = all_data['age'].astype(int)

    # Define the correct age group order
	if dataset=='ukb':
		all_data['age_group'] = all_data['age'].apply(ukb_assign_age_group)
		age_order = ['45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79']
		title='UK Biobank ($N=${0} scans)'.format(len(fmri_files))

	elif dataset=='hcp_ya_100':
		all_data['age_group'] = all_data['age'].apply(hcp_assign_age_group)
		age_order = ['22-25', '26-29', '30-33', '34-37']
		title='HCP Young Adult ($N=${0} scans)'.format(len(fmri_files))


	df_data = pd.DataFrame(outputs, columns = ['id', 'corr', 'FC_modes', 'metric'])
	df_merged = df_data.merge(all_data[['id', 'age_group', 'age']], on='id', how='inner')

	df_merged1 = df_merged[df_merged['FC_modes']=='all']
	df_merged2 = df_merged[df_merged['FC_modes']=='top']

	plotout(df_merged1, df_merged2, title)
