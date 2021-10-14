import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from joblib import dump
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat

reclass ={'algae_crustose_coralline': 'algae',
 'algae_fleshy_brown': 'algae',
 'algae_fleshy_green': 'algae',
 'algae_fleshy_red': 'algae',
 'algae_turf': 'algae',
 'coral_blue':'coral',
 'coral_brown': 'coral',
 'mud': 'mud/sand',
 'octocoral': 'coral',
 'sand': 'mud/sand',
 'seagrass': 'seagrass'}

level = 1

# Load in situ data
home =os.path.expanduser('~')
data_dir = '%s/Dropbox/rs/sister/scripts/algorithms/benthic_mapping/bc/data/' % home
ben_spectra = loadmat('%s/Hochberg_spectral_library_395-705_nm_2021-07-09.mat' % data_dir)
output_dir = '%s/Dropbox/rs/sister/repos/sister-benthcover/data/' % home

waves =  ben_spectra['wavelength'][0]
spectra_df = pd.DataFrame(ben_spectra['reflectance'],columns =ben_spectra['wavelength'][0])
spectra_df['class2'] = np.array([x[0][0] for x in ben_spectra['classname']])
spectra_df['class1'] = spectra_df['class2'].map(reclass)

accuracy = []
accuracy_std = []
fwhms = np.arange(1,51,1)

meta_path = '%smodel_metadata.json' % (output_dir)
metadata = {}

for f,fwhm in enumerate(fwhms):

    metadata['logreg_4class_benthcover_%02dnm' % fwhm] = {}

    print(fwhm)
    spectra_resample = spectra_df.copy()

    # Apply gaussian filters to spectra
    sigma = fwhm/2*(2*np.log(2))**.5
    spectra_resample[waves] = gaussian_filter1d(spectra_df[waves],sigma)

    acc2 = []

    #Cycle through each possible set of wavelengthe centers for each interval
    for offset in range(0,fwhm):
        new_waves = np.arange(430,671,fwhm) +offset
        new_waves = new_waves[(new_waves < 671) & (new_waves >= 430)]

        X = spectra_resample[new_waves]

        #Vector normalize spectra
        X = X/np.linalg.norm(X,axis=1)[:,np.newaxis]

        # Aggregate data to 10nm
        spectra_df_resample = pd.DataFrame(X,columns=new_waves)
        spectra_df_resample['class1'] = spectra_df['class1']
        spectra_df_resample['class2'] = spectra_df['class2']

        #Export model with 0 offset build using full dataset
        if offset == 0:
            logreg.fit(spectra_df_resample[new_waves],
                       spectra_df_resample['class%s' % level])
            dump(logreg, output_dir + 'logreg_4class_benthcover_%02dnm.joblib' % fwhm)
            metadata['logreg_4class_benthcover_%02dnm' % fwhm]['wavelengths'] = new_waves.tolist()


        acc = []
        #Build a model 10 times each with a 60/40 split of the data
        for x in range(10):
            X_train, X_test, y_train, y_test = train_test_split(spectra_df_resample[new_waves],
                                                                spectra_df_resample['class%s' % level],
                                                                test_size=.4)
            logreg = LogisticRegression(C=1e5)
            logreg.fit(X_train, y_train)
            y_pred = logreg.predict(X_test)
            acc.append(cohen_kappa_score(y_test,y_pred))
        acc2.append(np.mean(acc))

    accuracy.append(np.mean(acc2))
    accuracy_std.append([np.min(acc2),np.max(acc2)])

    metadata['logreg_4class_benthcover_%02dnm' % fwhm]['accuracy_mean'] = accuracy[f]
    metadata['logreg_4class_benthcover_%02dnm' % fwhm]['accuracy_min'] = accuracy_std[f][0]
    metadata['logreg_4class_benthcover_%02dnm' % fwhm]['accuracy_max'] = accuracy_std[f][1]

#Export metadate to file
with open(meta_path, 'w') as outfile:
    json.dump(metadata,outfile)

accuracy_range = np.abs(np.array(accuracy)[:,np.newaxis]- np.array(accuracy_std)).T

plt.rc('text', usetex=False)
plt.rc('font', family='sans-serif')

fig = plt.figure(figsize = (5,3),facecolor='w')
ax1 = fig.add_subplot(111)

ax1.scatter(fwhms,accuracy,c='k')
ax1.errorbar(fwhms,accuracy,
             yerr=accuracy_range,
             fmt='none',c='k')
ax1.set_ylabel(r"$\mathrm{Kappa\ score}$",
               fontsize = 13)
ax1.set_xlabel(r"$\mathrm{Spectral\ resolution\ (nm)}$",
               fontsize = 13)
ax1.set_ylim(0.2,1)
ax1.set_xlim(0,51)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_linewidth(1.5)
ax1.spines['left'].set_linewidth(1.5)
ax1.tick_params(labelsize=10)

plt.savefig('%s/Dropbox/rs/sister/repos/sister-benthcover/examples/spectral_res_vs_accuracy.png' % home,
            dpi = 400,bbox_inches='tight')
