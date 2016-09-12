'''
dataprocessing.py -- utilities for processing event-level ntuples with track info for RNN studies
Notes:
------
* It assumes your root files are stored in './data/train/' and './data/test/'
* It saves hdf5 files in './data/'
* Input variables are hardcoded in
* Remove all tracks with grade == -10 (IP3D selection)
* Remove all events without tracks and that don't pass b-tagging selection
* Default value is -999 (if changed, need to change it in IPRNN.py too)
* Written and tested with keras v1.0.5
Run: 
----
python dataprocessing.py --train_files *Akt4EMTo._000001.root --test_files *Akt4EMTo._000010.root --output 30trk_hits --sort_by d0z0sig --ntrk 30
'''
import json
import numpy as np
import pandas as pd
import deepdish.io as io
import keras
from keras.utils import np_utils
import pandautils as pup
import tqdm 
import os

def flatten(event_df):
    '''
    Definition:
    -----------
        Turn event-flat dataframe into jet-flat dataframe
        Note: no event-level branches should be present in event_df
    Args:
    -----
        event_df: event-flat pandas dataframe
    
    Returns:
    --------
        jet_df: jet_flat pandas dataframe 
    '''
    jet_df = pd.DataFrame()
    for key in event_df.keys():
        jet_df[key] = pup.flatten(event_df[key])
    return jet_df


def applycuts(df):
    '''
    Definition:
    -----------
        remove bad jets from df -- those with 0 tracks and those that don't pass b-tagging cuts
    Args:
    -----
        df: jet-flat pandas dataframe
    
    Returns:
    --------
        df: modified jet_flat pandas dataframe where bad jets have been removed
    '''
    # -- b-tagging cuts for AntiKt4EMTopoJets
    cuts = (abs(df['jet_eta']) < 2.5) & \
           (df['jet_pt'] > 20e3) & \
           ((df['jet_JVT'] > 0.59) | (df['jet_pt'] > 60e3) | (abs(df['jet_eta']) > 2.4)) & \
           (df['jet_aliveAfterOR'] == 1)

    # -- remove jets with 0 tracks
    z0 = df['jet_trk_ip3d_z0']
    notracks = []
    for i in xrange(len(z0)):
        notracks.append(len(z0[i]) == 0)
    hastracks = -np.array(notracks)

    # -- apply cuts and reset df indices
    df = df[(cuts & hastracks) == 1]
    df = df.reset_index(drop=True)
    return df

def dphi(trk_phi, jet_phi):
    '''
    Definition:
    -----------
        Calculate Delta Phi between track and jet
        The result will be in the range [-pi, +pi]
    Args:
    -----
        trk_phi: series or np.array with the list of track phi's for each jet
        jet_phi: series of np.array with the phi value for each jet
    
    Returns:
    --------
        deltaphi: np.array with the list of values of delta phi between the jet and each track
    '''

    import math
    PI = math.pi
    deltaphi = trk_phi - jet_phi # automatically broadcasts jet_phi across all tracks

    # -- ensure that phi stays within -pi and pi
    for jetN in xrange(len(deltaphi)):
        deltaphi[jetN][deltaphi[jetN] > PI] -= 2*PI
        deltaphi[jetN][deltaphi[jetN] < -PI] += 2*PI

    return deltaphi

def dr(eta1, eta2, phi1, phi2):
    '''
    Definition:
    -----------
        Function that calculates DR between two objects given their etas and phis
    Args:
    -----
        eta1 = pandas series or array, eta of first object
        eta2 = pandas series or array, eta of second object
        phi1 = pandas series or array, phi of first object
        phi2 = pandas series or array, phi of second object
    Output:
    -------
        dr = float, distance between the two objects 
    '''
    deta = abs(eta1 - eta2)
    dphi = abs(phi1 - phi2)
    dphi = np.array([np.arccos(np.cos(a)) for a in dphi]) # hack to avoid |phi1-phi2| larger than 180 degrees
    return np.array([np.sqrt(pow(de, 2) + pow(dp, 2)) for de, dp in zip(deta, dphi)])

def athenaname(var):
    '''
    Definition:
    -----------
        Quick utility function to turn variable names into Athena format
        This function may vary according to the names of the variables currently being used
    
    Args:
    -----
        var: a string with the variable name

    Returns:
    --------
        a string with the modified variable name
    '''
    return var.replace('jet_trk_ip3d_', '').replace('jet_trk_', '')


def sort_tracks(trk, data, theta, grade, sort_by, n_tracks):
    ''' 
    Definition:
        Sort tracks by sort_by and put them into an ndarray called data.
        Pad missing tracks with -999 --> net will have to have Masking layer
    
    Args:
    -----
        trk: a dataframe
        data: an array of shape (nb_samples, nb_tracks, nb_features)
        theta: pandas series with the jet_trk_theta values, used to enforce custom track selection
        grade:
        sort_by: a string representing the column to sort the tracks by
        n_tracks: number of tracks to cut off at. if >, truncate, else, -999 pad
    
    Returns:
    --------
        modifies @a data in place. Pads with -999
    
    '''
    
    for i, jet in tqdm.tqdm(trk.iterrows()):

         # -- remove tracks with grade = -10
        trk_selection = grade[i] != -10 

        # tracks = [[pt's], [eta's], ...] of tracks for each jet 
        tracks = np.array(
                [v[trk_selection].tolist() for v in jet.get_values()],
                dtype='float32'
            )[:, (np.argsort(jet[sort_by][trk_selection]))[::-1]]

        # total number of tracks per jet      
        ntrk = tracks.shape[1] 

        # take all tracks unless there are more than n_tracks 
        data[i, :(min(ntrk, n_tracks)), :] = tracks.T[:(min(ntrk, n_tracks)), :] 

        # default value for missing tracks 
        data[i, (min(ntrk, n_tracks)):, :  ] = -999 


def scale(data, var_names, sort_by, file_name, savevars):
    ''' 
    Args:
    -----
        data: a numpy array of shape (nb_samples, nb_tracks, n_variables)
        var_names: list of keys to be used for the model
        sort_by: string with the name of the column used for sorting tracks
                 needed to return json file in format requested by keras2json.py
        file_name: str, tag that identifies the specific way in which the data was prepared, 
                   i.e. '30trk_hits'
        savevars: bool -- True for training, False for testing
                  it decides whether we want to fit on data to find mean and std 
                  or if we want to use those stored in the json file 

    Returns:
    --------
        modifies data in place
    '''    
    scale = {}
    if savevars:  
        for v, vname in enumerate(var_names):
            print 'Scaling feature %s of %s (%s).' % (v, len(var_names), vname)
            f = data[:, :, v]
            slc = f[f != -999]
            # -- first find the mean and std of the training data
            m, s = slc.mean(), slc.std()
            # -- then scale the training distributions using the found mean and std
            slc -= m
            slc /= s
            data[:, :, v][f != -999] = slc.astype('float32')
            scale[v] = {'name' : athenaname(vname), 'mean' : m, 'sd' : s}

        # -- write variable json for lwtnn keras2json converter
        variable_dict = {
            'inputs': [{
                'name': scale[v]['name'],
                'scale': float(1.0 / scale[v]['sd']),
                'offset': float(-scale[v]['mean']),
                'default': None
                } 
                for v in xrange(len(var_names))],
            'class_labels': ['pu', 'pc', 'pb', 'ptau'],
            'keras_version': keras.__version__,
            'miscellaneous': {
                'sort_by': sort_by
            }
        }
        with open('var' + file_name + '.json', 'wb') as varfile:
            json.dump(variable_dict, varfile)

    # -- when operating on the test sample, use mean and std from training sample
    # -- this info is stored in the json file
    else:
        with open('var' + file_name + '.json', 'rb') as varfile:
            varinfo = json.load(varfile)

        for v, vname in enumerate(var_names):
            print 'Scaling feature %s of %s (%s).' % (v, len(var_names), vname)
            f = data[:, :, v]
            slc = f[f != -999]
            ix = [i for i in xrange(len(varinfo['inputs'])) if varinfo['inputs'][i]['name'] == athenaname(vname)]
            offset = varinfo['inputs'][ix[0]]['offset']
            scale = varinfo['inputs'][ix[0]]['scale']
            slc += offset
            slc *= scale
            data[:, :, v][f != -999] = slc.astype('float32')


def process_data(trk, jet_inputs, n_tracks, sort_by, file_name, savevars=False):
    ''' 
    takes a dataframe directly from ROOT files and 
    processes, sorts, and scales the data for you
    
    Args:
    -----
        trk: the dataframe in question
        jet_inputs: list of variables just used for ftag cuts
                    these will need to be dropped before training
        n_tracks: int, max number of tracks per event
        sort_by: str, name of the column used to sort the tracks in each event
        file_name: str, tag that identifies the specific way in which the data was prepared, i.e. '30trk_hits'
        savevars: bool -- True for training, False for testing
                  it decides whether we want to fit on data to find mean and std 
                  or if we want to use those stored in the json file 

    Returns:
    --------
        a dictionary with keys = {'X', 'y', and 'ip3d'}
    
    '''
    # -- flatten to jet-flat, apply ftag jet cuts and remove jets with 0 tracks
    trk = applycuts(flatten(trk))

    # -- targets
    y = trk.jet_LabDr_HadF

    # -- new df with ip3d vars only for later comparison
    ip3d = trk[ ['jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc'] ] 
    pt = trk['jet_pt'].values

    # -- add dr and pTfrac
    trk['jet_trk_dR'] = dr(trk['jet_trk_eta'], trk['jet_eta'], trk['jet_trk_phi'], trk['jet_phi'])
    trk['jet_trk_pTfrac'] = trk['jet_trk_pt'] / trk['jet_pt']

    # variable also used for ordering tracks
    trk['d0z0sig'] = (trk.jet_trk_ip3d_d0sig.copy() ** 2 + trk.jet_trk_ip3d_z0sig.copy() ** 2).pow(0.5)

    # -- separate theta from the other variables because we don't train on it. Ugly hack :((
    theta = trk['jet_trk_theta']
    grade = trk['jet_trk_ip3d_grade'].values

    # -- drop variables from the df that we do not want to use for training
    trk.drop(jet_inputs + [
        'jet_trk_phi',
        'jet_trk_theta',
        'jet_trk_pt',
        'jet_trk_eta',
        'jet_trk_ip3d_grade'
        ],
        axis=1, inplace=True) # no longer needed - not an input               

    n_variables = trk.shape[1] 
    var_names = trk.keys()

    data = np.zeros((trk.shape[0], n_tracks, n_variables), dtype='float32')

    # -- call functions to build X (= data)                                                                                                                                                                      
    sort_tracks(trk, data, theta, grade, sort_by, n_tracks)
    del theta, grade
    
    print 'Scaling features ...'
    #print 'Training on: ', var_names
    scale(data, var_names, sort_by, file_name, savevars)

    # -- default values                                                                                                                                                                                          
    data[np.isnan(data)] = -999

    # -- make classes pretty for keras (4 classes: b vs c vs l vs tau)                                                                                                                                                                          
    for ix, flav in enumerate(np.unique(y)):
        y[y == flav] = ix
    y_train = np_utils.to_categorical(y, len(np.unique(y)))

    # ip3d and pt will be used for performance evaluation
    return {'X' : data, 'y' : y_train, 'ip3d' : ip3d, 'jet_pt' : pt}

# ------------------------------------------   

if __name__ == '__main__':

    track_inputs = [
                    'jet_trk_ip3d_grade',
                    'jet_trk_pt', 
                    'jet_trk_ip3d_d0',
                    'jet_trk_ip3d_z0', 
                    'jet_trk_ip3d_d0sig', 
                    'jet_trk_ip3d_z0sig',
                    'jet_trk_phi',
                    'jet_trk_eta',
                    'jet_trk_theta',
                    'jet_trk_nInnHits',
                    'jet_trk_nNextToInnHits', 
                    'jet_trk_nBLHits',
                    'jet_trk_nsharedBLHits', 
                    'jet_trk_nsplitBLHits',
                    'jet_trk_nPixHits', 
                    'jet_trk_nsharedPixHits',
                    'jet_trk_nsplitPixHits', 
                    'jet_trk_nSCTHits',
                    'jet_trk_nsharedSCTHits', 
                    'jet_trk_expectBLayerHit'
                    ] 

    jet_inputs = [
                  'jet_pt',
                  'jet_phi',
                  'jet_eta',
                  'jet_LabDr_HadF', 
                  'jet_ip3d_pu', 
                  'jet_ip3d_pb', 
                  'jet_ip3d_pc',
                  'jet_JVT', 
                  'jet_aliveAfterOR'
                  ]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_files', 
        required=True, 
        help='str, name or wildcard specifying the files for training')
    parser.add_argument('--test_files', 
        required=True, 
        help='str, name or wildcard specifying the files for testing')
    parser.add_argument('--output', 
        required=True, type=str, 
        help="Tag that refers to the name of the output file, i.e.: '30trk_hits'", )
    parser.add_argument('--sort_by', 
        default='d0z0sig', 
        help='str, name of the variable used to order tracks in an event')
    parser.add_argument('--ntrk', 
        default=30, type=int, 
        help="Maximum number of tracks per event. \
        If the event has fewer tracks, use padding; if is has more, only consider the first ntrk")
    args = parser.parse_args()

    print 'Loading dataframes...'
    # -- currently only training and testing on one file each!
    trk_train = pup.root2panda(
        os.path.join('data', 'train', args.train_files), 
        'bTag_AntiKt4EMTopoJets', 
        branches = track_inputs + jet_inputs
    )
    trk_test  = pup.root2panda(
        os.path.join('data', 'test', args.test_files), 
        'bTag_AntiKt4EMTopoJets', 
        branches = track_inputs + jet_inputs
    )
    print 'Processing training sample ...'
    train_dict = process_data(trk_train, jet_inputs, args.ntrk, args.sort_by, args.output, savevars=True)
    del trk_train
    io.save(os.path.join('data', 'train_dict_' + args.output + '.h5'), train_dict)

    print 'Processing test sample...'
    test_dict = process_data(trk_test, jet_inputs, args.ntrk, args.sort_by, args.output)
    del trk_test
    io.save(os.path.join('data', 'test_dict_' + args.output + '.h5'), test_dict)
