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
python dataprocessing.py --train_files *Akt4EMTo._000001.root --test_files *Akt4EMTo._000010.root --output 30trk_grade --sort_by absSd0 --ntrk 30 --inputs grade
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

C_FRAC = 0.07
L_FRAC = 0.61

def generate_inputlist(input_request):
    '''
    builds the track_inputs list and the jet_inputs list
    depending on the type of training you want to do (encoded
    in input_request)
    '''
    track_inputs = [
                    'jet_trk_ip3d_grade',
                    'jet_trk_pt', 
                    'jet_trk_ip3d_d0',
                    'jet_trk_ip3d_z0', 
                    'jet_trk_ip3d_d0sig', 
                    'jet_trk_ip3d_z0sig',
                    'jet_trk_phi',
                    'jet_trk_eta',
                    'jet_trk_theta'
                    ]
    jet_inputs = [
                  'jet_pt', # for cutting
                  'jet_phi', # for cutting
                  'jet_eta', # for cutting
                  'jet_pt_orig', # for training
                  'jet_phi_orig', # for training
                  'jet_eta_orig', # for training
                  'jet_LabDr_HadF', 
                  'jet_ip3d_pu', 
                  'jet_ip3d_pb', 
                  'jet_ip3d_pc',
                  'jet_JVT', 
                  'jet_aliveAfterOR'
                  ]
    if input_request == 'hits':
        track_inputs += [
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
    return track_inputs, jet_inputs

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
        Notes: the cuts are being applied to calibrated values
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
        trk: a dataframe or pandas serier
        data: an array of shape (nb_samples, nb_tracks, nb_features)
        theta: pandas series with the jet_trk_theta values, used to enforce custom track selection
        grade: numpy array with the grade values
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
            'inputs' : [],
            'input_sequences' : 
            [{
                'name' : 'grade_input',
                'variables' :
                [{
                    'name': 'grade',
                    'scale': 1,
                    'offset': 0
                }]
            },
            {
                'name' : 'track_input',
                'variables' :
                 [{
                    'name': scale[v]['name'],
                    'scale': float(1.0 / scale[v]['sd']),
                    'offset': float(-scale[v]['mean']),
                    'default': None
                    } 
                    for v in xrange(len(var_names))]
            }],
            'outputs': 
            [{
                'name': 'tagging',
                'labels': ['pu', 'pc', 'pb', 'ptau']
            }],
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
            ix = [i for i in xrange(len(varinfo['input_sequences'][1]['variables'])) if varinfo['input_sequences'][1]['variables'][i]['name'] == athenaname(vname)]
            offset = varinfo['input_sequences'][1]['variables'][ix[0]]['offset']
            scale = varinfo['input_sequences'][1]['variables'][ix[0]]['scale']
            slc += offset
            slc *= scale
            data[:, :, v][f != -999] = slc.astype('float32')


def reweight_to_l(jet_pt, jet_eta, y):
    '''
    Definition:
    -----------
        Extract weights by reweighting pT and eta to light-distribution
    '''
    jet_eta = abs(jet_eta)
    pt_bins = [10, 50, 100, 150, 200, 300, 500, 99999]
    eta_bins = [0, 0.5, 1.5, 2.5, 5, 10] #np.linspace(0, 2.5, 6)

    b_bins = np.histogram2d(jet_pt[y == 5] / 1000, jet_eta[y == 5], bins=[pt_bins, eta_bins])
    c_bins = np.histogram2d(jet_pt[y == 4] / 1000, jet_eta[y == 4], bins=[pt_bins, eta_bins])
    l_bins = np.histogram2d(jet_pt[y == 0] / 1000, jet_eta[y == 0], bins=[pt_bins, eta_bins])

    wl= np.ones(sum(y == 0))
    wt= np.ones(sum(y == 15)) # not reweighting taus

    wc = [(l_bins[0] / c_bins[0])[arg] for arg in zip(
        np.digitize(jet_pt[y == 4] / 1000, l_bins[1]) - 1, 
        np.digitize(jet_eta[y == 4], l_bins[2]) - 1
    )]
    

    wb = [(l_bins[0] / b_bins[0])[arg] for arg in zip(
        np.digitize(jet_pt[y == 5] / 1000, l_bins[1]) - 1, 
        np.digitize(jet_eta[y == 5], l_bins[2]) - 1
    )]

    n_light = wl.sum()
    n_charm = (n_light * C_FRAC) / L_FRAC
    n_bottom = (n_light * (1 - L_FRAC - C_FRAC )) / L_FRAC

    w = np.zeros(len(y))
    w[y == 5] = np.array(wb) * (n_bottom / sum(wb)) 
    w[y == 4] = np.array(wc) * (n_charm / sum(wc)) 
    w[y == 0] = wl
    w[y == 15] = wt

    return w


def process_data(trk, jet_inputs, n_tracks, sort_by, file_name, input_request, savevars=False):
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
        input_request: string to specify what inputs to use in training. One of: 'hits', 'grade'
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

    # -- add dr and pTFrac
    trk['jet_trk_dR'] = dr(trk['jet_trk_eta'], trk['jet_eta_orig'], trk['jet_trk_phi'], trk['jet_phi_orig'])
    trk['jet_trk_pTFrac'] = trk['jet_trk_pt'] / trk['jet_pt_orig']

    # variable also used for ordering tracks
    #trk['d0z0sig'] = (trk.jet_trk_ip3d_d0sig.copy() ** 2 + trk.jet_trk_ip3d_z0sig.copy() ** 2).pow(0.5)
    trk['absSd0'] = np.abs(trk.jet_trk_ip3d_d0)

    # -- separate theta from the other variables because we don't train on it. Ugly hack :((
    theta = trk['jet_trk_theta']
    grade = trk['jet_trk_ip3d_grade'].values

    # -- get weights
    w = reweight_to_l(trk['jet_pt_orig'].values, trk['jet_eta_orig'].values, y.values)

    # -- drop variables from the df that we do not want to use for training
    trk.drop(jet_inputs + [
        'jet_trk_phi',
        'jet_trk_theta',
        'jet_trk_pt',
        'jet_trk_eta',
        'jet_trk_ip3d_d0',
        'jet_trk_ip3d_z0',
        #'jet_trk_ip3d_grade'
        ],
        axis=1, inplace=True) # no longer needed - not an input 

    n_variables = trk.shape[1] 
    var_names = trk.keys().tolist()
    
    data = np.zeros((trk.shape[0], n_tracks, n_variables), dtype='float32')

    # -- call functions to build X (= data) and sorted_grade                                                                                                                                                                     
    sort_tracks(trk, data, theta, grade, sort_by, n_tracks)
    # ugly hacks to deal with grade embedding :(
    grade_index = np.argwhere(np.array(var_names) == 'jet_trk_ip3d_grade')[0][0]
    sorted_grade = data[:, :, grade_index]
    data = data[:, :, [i for i in range(len(var_names)) if i != grade_index]]
    sorted_grade = sorted_grade + 1 # shift everything up by 1
    sorted_grade[sorted_grade == -998] = 0 # replace -998 with 0
    var_names.remove('jet_trk_ip3d_grade')

    if sort_by not in ['jet_trk_ip3d_grade', 'jet_trk_ip3d_d0sig', 
        'jet_trk_ip3d_z0sig', 'jet_trk_dR', 'jet_trk_pTFrac']:
        data = data[:, :, :-1] # remove sorting var if it's not one of the training vars
        # WARNING!! Make sure the sorting var is the last one that was created!
        var_names.remove(sort_by)
    del theta, grade

    print 'Scaling features ...'
    scale(data, var_names, sort_by, file_name, savevars)

    # -- default values                                                                                                                                                                                          
    data[np.isnan(data)] = -999

    # -- make classes pretty for keras (4 classes: b vs c vs l vs tau)                                                                                                                                                                          
    for ix, flav in enumerate(np.unique(y)):
        y[y == flav] = ix
    y_train = np_utils.to_categorical(y, len(np.unique(y)))

    # ip3d and pt will be used for performance evaluation
    return {'X' : data,
            'y' : y_train,
            'ip3d' : ip3d,
            'jet_pt' : pt,
            'w' : w,
            'grade' : np.expand_dims(sorted_grade, -1)}

# ------------------------------------------   

if __name__ == '__main__':

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
        default='absSd0', 
        help='str, name of the variable used to order tracks in an event')
    parser.add_argument('--ntrk', 
        default=30, type=int, 
        help="Maximum number of tracks per event. \
        If the event has fewer tracks, use padding; if is has more, only consider the first ntrk")
    parser.add_argument('--inputs', 
        default='grade', 
        help='one of: hits, grade')
    args = parser.parse_args()

    track_inputs, jet_inputs = generate_inputlist(args.inputs)


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
    train_dict = process_data(trk_train, jet_inputs, args.ntrk, args.sort_by, args.output, args.inputs, savevars=True)
    del trk_train
    io.save(os.path.join('data', 'train_dict_' + args.output + '.h5'), train_dict)

    print 'Processing test sample...'
    test_dict = process_data(trk_test, jet_inputs, args.ntrk, args.sort_by, args.output, args.inputs,)
    del trk_test
    io.save(os.path.join('data', 'test_dict_' + args.output + '.h5'), test_dict)
