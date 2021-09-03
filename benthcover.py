"""
SISTER
Space-based Imaging Spectroscopy and Thermal PathfindER
Author: Adam Chlus
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import argparse
from joblib import load
import hytools_lite as htl
from hytools_lite.io import WriteENVI
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

CLASSES =  ['algae','coral','mud/sand','seagrass']
N_CLASSES = len(CLASSES)

def progbar(curr, total, full_progbar = 100):
    '''Display progress bar.
    Gist from:
    https://gist.github.com/marzukr/3ca9e0a1b5881597ce0bcb7fb0adc549
    Args:
        curr (int, float): Current task level.
        total (int, float): Task level at completion.
        full_progbar (TYPE): Defaults to 100.
    Returns:
        None.
    '''
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')

def main():
    'Benthic cover classifier'
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--model',  type=str, default = 'ranfor', choices = ['ranfor','logreg'])
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--prob', action='store_true',
                        help='Export probabilities')

    args = parser.parse_args()


    ###############
    ##Testing
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args([])
    # args.input = '/data2/desis/l2/DESIS_DT0595258704_002-20210604T122041-V0213_modtran/DESIS_DT0595258704_002-20210604T122041-V0213_rfl_prj_rfl_10nm_Rb'
    # args.out_dir ='/data2/temp/bethos_aop_testing/'
    # args.model ='ranfor'
    # args.verbose = True
    # args.prob = True

    root_dir =os.path.realpath(os.path.split(__file__)[0])
    classifier_file = root_dir + '/data/%s_4class_benthcover.joblib' % args.model
    classifier = load(classifier_file)

    #Load benthic reflectance image
    ben_rfl = htl.HyTools()
    ben_rfl.read_file(args.input,'envi')
    iterator =ben_rfl.iterate(by = 'chunk',chunk_size = (200,200))

    probability = np.full((ben_rfl.lines,ben_rfl.columns,N_CLASSES),-1)

    i = 0
    while not iterator.complete:
        chunk = iterator.read_next()
        data = chunk/np.linalg.norm(chunk,axis=2)[:,:,np.newaxis]
        data[np.isnan(data)] =0
        data =data.reshape(chunk.shape[0]*chunk.shape[1],chunk.shape[2])

        prob = classifier.predict_proba(data)
        prob = prob.reshape(chunk.shape[0],chunk.shape[1],N_CLASSES)
        prob = prob*100

        prob[chunk.sum(axis=2) <=0] = -9999
        probability[iterator.current_line:iterator.current_line+chunk.shape[0],
             iterator.current_column:iterator.current_column+chunk.shape[1],:] = prob

        i+=prob.shape[0]*prob.shape[1]
        if args.verbose:
            progbar(i,ben_rfl.lines*ben_rfl.columns, full_progbar = 100)

    out_header = ben_rfl.get_header()
    out_header['bands']= N_CLASSES
    out_header['wavelength']= []
    out_header['fwhm']= []

    # Export probabilities for each cover class
    if args.prob:
        out_header['band names'] = CLASSES
        prob_file = args.out_dir + '/' + ben_rfl.base_name.replace('Rb','cover_prob')
        writer = WriteENVI(prob_file,out_header)
        for band in range(N_CLASSES):
            writer.write_band(probability[:,:,band],band)

    # Export cover map
    out_header['bands']= 1
    out_header['class names'] = CLASSES
    out_header['band names'] = ['benthic_cover']

    cover_file = args.out_dir + '/' + ben_rfl.base_name.replace('Rb','cover_class')
    writer = WriteENVI(cover_file,out_header)
    class_max = probability.argmax(axis=2)
    class_max[probability[:,:,0] == -9999] = -9999
    writer.write_band(class_max,0)

if __name__ == "__main__":
    main()





