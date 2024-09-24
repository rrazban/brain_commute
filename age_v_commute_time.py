"""
Check how commute time-functional connectivity 
correlations depend on individuals' ages

"""

import numpy as np
import sys, glob, os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from brain_disorder import run

plt.rcParams.update({'font.size': 14})


if __name__ == "__main__":
	atlas = 'DesKi'	#Talairach, DesKi
	dataset = 'ukb'	#ukb, hcp_ya_100

	fmri_files = glob.glob('data/{0}/{1}/fMRI/*'.format(dataset, atlas))
	subs, ct_fxn, ct_topfxn = run(fmri_files, atlas)

	df_data = pd.DataFrame({'id': subs, 'corr':ct_fxn, 'corr_top': ct_topfxn})

	if dataset=='ukb':
		title = 'UK Biobank ($N=${0} subjects)'.format(len(fmri_files))
	elif dataset=='hcp_ya_100':
		title='HCP Young Adult ($N=${0} scans)'.format(len(fmri_files))


	group_file = 'data/ukb/subject_info/phenotypes'
	df_group = pd.read_csv('{0}.csv'.format(group_file))
	df_overlap = df_data.merge(df_group,on='id')


	rval, pval = (spearmanr(df_overlap['age'], df_overlap['corr']))
	plt.scatter(df_overlap['age'], df_overlap['corr'], label = "$\\rho=${0:.2f} ({1:.2E})".format(rval, pval, len(df_overlap['age'])))
	plt.xlabel('age in years')
	plt.ylim([-0.09, 0.64])
	plt.ylabel('$\\rho(-$commute time, FC$)$')
	plt.title(title)
	plt.legend()
	plt.tight_layout()
	plt.show()

