"""
Compare commute time - functional connectivity 
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
from scipy.stats import spearmanr


sys.path.append('utlts/')
from analysis_tools import get_structure, delete_indi, no_edges, correlation, delete_empty_index
from structure_metrics import commute_time, deconstruct_cov

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

		fc_top_mode = deconstruct_cov(fc, 1)
		upper_tri = np.triu_indices(structure.shape[0], k=1)	#dont include diagonal elements, also dont double count

#####		print('deconstructing structure to top mode!!!')
#		structure = deconstruct_cov(structure, 1)


		cts = commute_time(structure)
		outputs.append([int(sub), -correlation(cts, fc, upper_tri), 'all', 'commute time'])
		outputs.append([int(sub), -correlation(cts, fc_top_mode, upper_tri), 'top', 'commute time'])


		subs.append(int(sub))
	
	return outputs


def hcp_assign_age_group(age):
    """Assigns age to an age range group"""
    if 22 <= age <= 25:
        return '22-25'
    elif 26 <= age <= 29:
        return '26-29'
    elif 30 <= age <= 33:
        return '30-33'
    elif 34 <= age <= 37:
        return '34-37'
    else:
        return 'Other'

def ukb_assign_age_group(age):
    """Assigns age to an age range group"""
    if 45 <= age < 50:
        return '45-49'
    elif 50 <= age < 55:
        return '50-54'
    elif 55 <= age < 60:
        return '55-59'
    elif 60 <= age <65:
        return '60-64'
    elif 65 <= age <70:
        return '65-69'
    elif 70 <= age <75:
        return '70-74'
    elif 75 <= age <80:
        return '75-79'

    else:
        return 'Other'


def plotout(df_merged1, df_merged2):
	rval, pval = (spearmanr(df_merged1['age'], df_merged1['corr']))
	rval2, pval2 = (spearmanr(df_merged2['age'], df_merged2['corr']))
	
	plt.figure().set_figwidth(8.5)
	ax = sns.violinplot(data=df_merged, x='age_group', y='corr', hue='FC modes', inner='quart', split=True)#, labels = ["all, $\\rho=${0:.2f} ({1:.1E})".format(rval, pval),"top, $\\rho=${0:.1f} ({1:.2E})".format(rval2, pval2)])
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(title='FC modes', handles = handles, labels = ["all, $\\rho=${0:.2f} ({1:.1E})".format(rval, pval),"top, $\\rho=${0:.2f} ({1:.1E})".format(rval2, pval2)], loc = 'lower right')
	plt.xlabel('age group')
	plt.ylabel('$\\rho(-$commute time, FC$)$')
	plt.ylim([-0.16070026065475126, 0.717341464235392])


	plt.title(title)
	plt.tight_layout()
	plt.grid()
	plt.show()
	



if __name__ == "__main__":
	atlas = 'DesKi'
	dataset = 'hcp_ya_100'	#hcp_ya_100, ukb
	fmri_files = glob.glob('data/{0}/{1}/fMRI/*csv'.format(dataset, atlas))

    # Run analysis to get subject IDs and commute time matrices
	outputs = run(fmri_files, atlas, dataset)

	#get demographics data
	all_data = pd.read_csv('data/{0}/subject_info/phenotypes.csv'.format(dataset), sep=',')


    # Ensure `id` is integer and `age` is consistent
	all_data['id'] = all_data['id'].astype(int)
	all_data['age'] = all_data['age'].astype(int)

	if dataset=='ukb':
		all_data['age_group'] = all_data['age'].apply(ukb_assign_age_group)
		age_order = ['45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79']
		title='UK Biobank ($N=${0} scans)'.format(len(fmri_files))

	elif dataset=='hcp_ya_100':
		all_data['age_group'] = all_data['age'].apply(hcp_assign_age_group)
		age_order = ['22-25', '26-29', '30-33', '34-37']
		title='HCP Young Adult ($N=${0} scans)'.format(len(fmri_files))


    # Define the correct age group order
	all_data['age_group'] = pd.Categorical(all_data['age_group'], categories=age_order, ordered=True)

    # Merge the subject data with age group information
	df_data = pd.DataFrame(outputs, columns = ['id', 'corr', 'FC modes', 'metric'])
	df_merged = df_data.merge(all_data[['id', 'age_group', 'age']], on='id', how='inner')


	df_merged1 = df_merged[df_merged['FC modes']=='all']
	df_merged2 = df_merged[df_merged['FC modes']=='top']
	plotout(df_merged1, df_merged2)

