"""
Compare commute time - functional connectivity 
correlation with respect to Mini-Mental State
Examination Score.
Across all individuals for
HCP Young Adult dataset.

"""




import os
import numpy as np
import sys, glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from age_v_performance import run

plt.rcParams.update({'font.size': 14})


def plotout(df_merged1, df_merged2):

	rval, pval = (spearmanr(df_merged1['mmse'], df_merged1['corr']))
	rval2, pval2 = (spearmanr(df_merged2['mmse'], df_merged2['corr']))

	ax = sns.violinplot(data=df_merged, x='mmse', y='corr', hue='FC modes', inner='quart', split=True)
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(title='FC modes', handles = handles, labels = ["all, $\\rho=${0:.2f} ({1:.1E})".format(rval, pval),"top, $\\rho=${0:.2f} ({1:.1E})".format(rval2, pval2)])
	plt.ylabel('$\\rho(-$commute time, FC$)$')
	plt.ylim([-0.16070026065475126, 0.717341464235392])
	plt.xlabel('Mini-Mental State Examination score')
	plt.title('HCP Young Adult ($N=${0} scans)'.format(len(fmri_files)))
	plt.grid()
	plt.tight_layout()
	plt.show()
	

if __name__ == "__main__":
	atlas = 'DesKi'
	dataset = 'hcp_ya_100'	#fixed, we do not have mmse scores for ukb
	fmri_files = glob.glob('data/{0}/{1}/fMRI/*csv'.format(dataset, atlas))

    # Run analysis to get subject IDs and commute time matrices
	outputs = run(fmri_files, atlas, dataset)

    # Load the all.csv file with a comma separator
	all_data = pd.read_csv('data/{0}/subject_info/phenotypes.csv'.format(dataset), sep=',')


    # Ensure `id` is integer and `age` is consistent
	all_data['id'] = all_data['id'].astype(int)
	all_data['age'] = all_data['age'].astype(int)


	df_data = pd.DataFrame(outputs, columns = ['id', 'corr', 'FC modes', 'metric'])
	df_merged = df_data.merge(all_data[['id', 'mmse']], on='id', how='inner')

	df_merged1 = df_merged[df_merged['FC modes']=='all']
	df_merged2 = df_merged[df_merged['FC modes']=='top']


	plotout(df_merged1, df_merged2)
