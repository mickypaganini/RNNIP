'''
Info:
    This script loads root files containing standard 
    b-tagging information + the IPMP outputs. 
    It turns the data into jet-flat structure,
    replicates the variable creation and modification
    that are present in MV2, scales the variables 
    and splits them into training and testing sets.
    Finally, the data is stored as dictionaries in
    HDF5 format.
Author: 
    Michela Paganini - Yale/CERN
    michela.paganini@cern.ch
To-do:
    Balance flavor fractions.
'''

import pandas as pd
import pandautils as pup
import numpy as np
import math
import os
import sys
import logging
import deepdish.io as io
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from rootpy.vector import LorentzVector, Vector3
import matplotlib.pyplot as plt

OUTNAME = '_large_'

def main(iptagger):
    configure_logging()
    logger = logging.getLogger("generate_data_DL1")
    logger.info("Running on: {}".format(iptagger))

    branches, training_vars = set_features(iptagger)
    logger.info('Creating dataframe...')
    df = pup.root2panda('../data/final_production/*', 
        'bTag_AntiKt4EMTopoJets', 
        branches = branches)

    logger.info('Transforming variables...')
    df = transformVars(df, iptagger)

    logger.info('Flattening df...')
    df.drop(['PVx', 'PVy', 'PVz'], axis=1, inplace=True)
    df_flat = pd.DataFrame({k: pup.flatten(c) for k, c in df.iterkv()})
    del df

    logger.info('Applying cuts...')
    df_flat = apply_calojet_cuts(df_flat)

    logger.info('Creating X, y, w, mv2c10...')
    y = df_flat['jet_LabDr_HadF'].values
    mv2c10 = df_flat['jet_mv2c10'].values
    # -- slice df by only keeping the training variables
    X = df_flat[training_vars].values
    pt_col = np.argwhere(np.array(training_vars) == 'jet_pt')[0][0]
    eta_col = np.argwhere(np.array(training_vars) == 'abs(jet_eta)')[0][0]
    w = reweight_to_b(X, y, pt_col, eta_col)
    #w = reweight_to_l(X, y, pt_col, eta_col)
    del df_flat

    logger.info('Shuffling, splitting, scaling...')
    ix = np.array(range(len(y)))
    X_train, X_test, y_train, y_test, w_train, w_test, ix_train, ix_test, mv2c10_train, mv2c10_test = train_test_split(X, y, w, ix, mv2c10, train_size=0.6)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train = {
        'X' : X_train,
        'y' : y_train,
        'w' : w_train,
        'ix': ix_train,
        'mv2c10': mv2c10_train
    }

    test = {
        'X' : X_test,
        'y' : y_test,
        'w' : w_test,
        'ix': ix_test,
        'mv2c10': mv2c10_test
    }

    logger.info('Saving dictionaries to hdf5...')
    io.save(os.path.join('..', 'data', 'DL1-' + iptagger + OUTNAME + '-train-db.h5'), train)
    io.save(os.path.join('..', 'data', 'DL1-' + iptagger + OUTNAME + '-test-db.h5'), test)


# -----------------------------------------------------------------    
  
# def match_shape(arr, ref):
#     '''
#     function to replace 1d array into array of arrays to match event-jets structure
#     Args:
#     -----
#         arr: jet-flat array 
#         ref: array of arrays (event-jet structure) as reference for shape matching
#     Returns:
#     --------
#         the initial arr but with the same event-jet structure as ref
#     Raises:
#     -------
#         ValueError
#     '''
#     shape = [len(a) for a in ref]
#     if len(arr) != np.sum(shape):
#         raise ValueError('Incompatible shapes: len(arr) = {}, total elements in ref: {}'.format(len(arr), np.sum(shape)))
  
#     return [arr[ptr:(ptr + nobj)].tolist() for (ptr, nobj) in zip(np.cumsum([0] + shape[:-1]), shape)]

# ----------------------------------------------------------------- 

def set_features(iptagger):
    '''
    '''

    branches = [
        'jet_pt',
        'jet_eta',
        'jet_phi',
        'jet_m',
        'jet_sv1_vtx_x',
        'jet_sv1_vtx_y',
        'jet_sv1_vtx_z',
        'jet_sv1_ntrkv',
        'jet_sv1_m',
        'jet_sv1_efc',
        'jet_sv1_n2t',
        'jet_sv1_sig3d',
        'jet_jf_n2t',
        'jet_jf_ntrkAtVx',
        'jet_jf_nvtx',
        'jet_jf_nvtx1t',
        'jet_jf_m',
        'jet_jf_efc',
        'jet_jf_sig3d',
        'jet_jf_deta',
        'jet_jf_dphi',
        'PVx',
        'PVy',
        'PVz',
        'jet_aliveAfterOR',
        'jet_aliveAfterORmu',
        'jet_nConst',
        'jet_LabDr_HadF',
        'jet_mv2c10'
        ]

    training_vars = [
        'jet_pt', 
        'abs(jet_eta)', 
        'jet_sv1_ntrkv',
        'jet_sv1_m',
        'jet_sv1_efc',
        'jet_sv1_n2t',
        'jet_sv1_Lxy',
        'jet_sv1_L3d',
        'jet_sv1_sig3d',
        'jet_sv1_dR',
        'jet_jf_n2t',
        'jet_jf_ntrkAtVx',
        'jet_jf_nvtx',
        'jet_jf_nvtx1t',
        'jet_jf_m',
        'jet_jf_efc',
        'jet_jf_dR',
        'jet_jf_sig3d'
        ]

    if iptagger == 'ip3d':
        branches += [
                'jet_ip2d_pu', 
                'jet_ip2d_pc',
                'jet_ip2d_pb',
                'jet_ip3d_pu',
                'jet_ip3d_pc',
                'jet_ip3d_pb'
                ]
        training_vars += [
                'jet_ip2',
                'jet_ip2_c',
                'jet_ip2_cu',
                'jet_ip3',
                'jet_ip3_c',
                'jet_ip3_cu'
                ]

    elif iptagger == 'ipmp':
        branches += [
                'jet_ipmp_pu',
                'jet_ipmp_pc',
                'jet_ipmp_pb',
                'jet_ipmp_ptau'
                ]
        training_vars += [
                'jet_ip',
                'jet_ip_c',
                'jet_ip_cu'
                ]
    else:
        raise ValueError('iptagger can only be ip3d or ipmp')
    return branches, training_vars

# ----------------------------------------------------------------- 

def transformVars(df, iptagger):
    '''
    modifies the variables to create the ones that mv2 uses, inserts default values when needed, saves new variables
    in the dataframe
    Args:
    -----
        df: pandas dataframe containing all the interesting variables as extracted from the .root file
        iptagger: string, either 'ip3d' or 'ipmp'
    Returns:
    --------
        modified mv2-compliant dataframe
    '''
    # -- modify features and set default values
    df['abs(jet_eta)'] = abs(df['jet_eta'])

    # -- create new IPxD features
    if iptagger == 'ip3d':
        for (pu,pb,pc) in zip(df['jet_ip2d_pu'],df['jet_ip2d_pb'],df['jet_ip2d_pc']) :
            pu[np.logical_or(pu >= 10, pu <-1)] = -1
            pb[np.logical_or(pu >= 10, pu <-1)] = -1
            pc[np.logical_or(pu >= 10, pu <-1)] = -1
        for (pu,pb,pc) in zip(df['jet_ip3d_pu'],df['jet_ip3d_pb'],df['jet_ip3d_pc']) :
            pu[pu >= 10] = -1
            pb[pu >= 10] = -1
            pc[pu >= 10] = -1       
        df['jet_ip2'] = (df['jet_ip2d_pb'] / df['jet_ip2d_pu']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
        df['jet_ip2_c'] = (df['jet_ip2d_pb'] / df['jet_ip2d_pc']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
        df['jet_ip2_cu'] = (df['jet_ip2d_pc'] / df['jet_ip2d_pu']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
        df['jet_ip3'] = (df['jet_ip3d_pb'] / df['jet_ip3d_pu']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
        df['jet_ip3_c'] = (df['jet_ip3d_pb'] / df['jet_ip3d_pc']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
        df['jet_ip3_cu'] = (df['jet_ip3d_pc'] / df['jet_ip3d_pu']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
    
    elif iptagger == 'ipmp':
        for (pu,pb,pc) in zip(df['jet_ipmp_pu'],df['jet_ipmp_pb'],df['jet_ipmp_pc']) :
            pu[pu >= 10] = -1
            pb[pu >= 10] = -1
            pc[pu >= 10] = -1 
        df['jet_ip'] = (df['jet_ipmp_pb'] / df['jet_ipmp_pu']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
        df['jet_ip_c'] = (df['jet_ipmp_pb'] / df['jet_ipmp_pc']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
        df['jet_ip_cu'] = (df['jet_ipmp_pc'] / df['jet_ipmp_pu']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
    
    else:
        raise ValueError('iptagger can only be ip3d or ipmp')

    # -- SV1 features
    dx = df['jet_sv1_vtx_x']-df['PVx']
    dy = df['jet_sv1_vtx_y']-df['PVy']
    dz = df['jet_sv1_vtx_z']-df['PVz']

    v_jet = LorentzVector()
    pv2sv = Vector3()
    sv1_L3d = []
    sv1_Lxy = []
    dR = [] 

    for index, dxi in enumerate(dx): # loop thru events
        sv1_L3d_ev = []
        sv1L_ev = []
        dR_ev = []
        for jet in xrange(len(dxi)): # loop thru jets
            v_jet.SetPtEtaPhiM(df['jet_pt'][index][jet], df['jet_eta'][index][jet], df['jet_phi'][index][jet], df['jet_m'][index][jet])
            if (dxi[jet].size != 0):
                sv1_L3d_ev.append(np.sqrt(pow(dx[index][jet], 2) + pow(dy[index][jet], 2) + pow(dz[index][jet], 2))[0])
                sv1L_ev.append(math.hypot(dx[index][jet], dy[index][jet]))
                
                pv2sv.SetXYZ(dx[index][jet], dy[index][jet], dz[index][jet])
                jetAxis = Vector3(v_jet.Px(), v_jet.Py(), v_jet.Pz())
                dR_ev.append(pv2sv.DeltaR(jetAxis))
            else: 
                dR_ev.append(-1)   
                sv1L_ev.append(-100)
                sv1_L3d_ev.append(-100)
             
        sv1_Lxy.append(sv1L_ev)
        dR.append(dR_ev) 
        sv1_L3d.append(sv1_L3d_ev)
        
    df['jet_sv1_dR'] = dR 
    df['jet_sv1_Lxy'] = sv1_Lxy
    df['jet_sv1_L3d'] = sv1_L3d

    # -- add more default values for sv1 variables
    sv1_vtx_ok = pup.match_shape(np.asarray([len(el) for event in df['jet_sv1_vtx_x'] for el in event]), df['jet_pt'])

    for (ok4event, sv1_ntkv4event, sv1_n2t4event, sv1_mass4event, sv1_efrc4event, sv1_sig34event) in zip(sv1_vtx_ok, df['jet_sv1_ntrkv'], df['jet_sv1_n2t'], df['jet_sv1_m'], df['jet_sv1_efc'], df['jet_sv1_sig3d']): 
        sv1_ntkv4event[np.asarray(ok4event) == 0] = -1
        sv1_n2t4event[np.asarray(ok4event) == 0] = -1 
        sv1_mass4event[np.asarray(ok4event) == 0] = -1000
        sv1_efrc4event[np.asarray(ok4event) == 0] = -1 
        sv1_sig34event[np.asarray(ok4event) == 0] = -100

    # -- JF features
    jf_dR = []
    for eventN, (etas, phis, masses) in enumerate(zip(df['jet_jf_deta'], df['jet_jf_dphi'], df['jet_jf_m'])): # loop thru events
        jf_dR_ev = []
        for m in xrange(len(masses)): # loop thru jets
            if (masses[m] > 0):
                jf_dR_ev.append(np.sqrt(etas[m] * etas[m] + phis[m] * phis[m]))
            else:
                jf_dR_ev.append(-10)
        jf_dR.append(jf_dR_ev)
    df['jet_jf_dR'] = jf_dR

    # -- add more default values for jf variables
    for (jf_mass,jf_n2tv,jf_ntrkv,jf_nvtx,jf_nvtx1t,jf_efrc,jf_sig3) in zip(df['jet_jf_m'],df['jet_jf_n2t'],df['jet_jf_ntrkAtVx'],df['jet_jf_nvtx'],df['jet_jf_nvtx1t'],df['jet_jf_efc'],df['jet_jf_sig3d']):
        jf_n2tv[jf_mass <= 0] = -1;
        jf_ntrkv[jf_mass <= 0] = -1;
        jf_nvtx[jf_mass <= 0]  = -1;
        jf_nvtx1t[jf_mass <= 0]= -1;
        jf_mass[jf_mass <= 0]  = -1e3;
        jf_efrc[jf_mass <= 0]  = -1;
        jf_sig3[jf_mass <= 0]  = -100;

    return df

# ----------------------------------------------------------------- 

def _replaceInfNaN(x, value):
    '''
    function to replace Inf and NaN with a default value
    Args:
    -----
        x:     arr of values that might be Inf or NaN
        value: default value to replace Inf or Nan with
    Returns:
    --------
        x:     same as input x, but with Inf or Nan raplaced by value
    '''
    x[np.isfinite( x ) == False] = value 
    return x

# -----------------------------------------------------------------  

def apply_calojet_cuts(df):

    cuts = (abs(df['jet_eta']) < 2.5) & \
           (df['jet_pt'] > 10e3) & \
           (df['jet_aliveAfterOR'] == 1) & \
           (df['jet_aliveAfterORmu'] == 1) & \
           (df['jet_nConst'] > 1)

    df = df[cuts].reset_index(drop=True)
    return df

# ----------------------------------------------------------------- 

def reweight_to_b(X, y, pt_col, eta_col):
    '''
    Definition:
    -----------
        Reweight to b-distribution
    '''

    pt_bins = [10, 50, 100, 150, 200, 300, 500, 99999]
    eta_bins = np.linspace(0, 2.5, 6)

    b_bins = plt.hist2d(X[y == 5, pt_col] / 1000, X[y == 5, eta_col], bins=[pt_bins, eta_bins])
    c_bins = plt.hist2d(X[y == 4, pt_col] / 1000, X[y == 4, eta_col], bins=[pt_bins, eta_bins])
    l_bins = plt.hist2d(X[y == 0, pt_col] / 1000, X[y == 0, eta_col], bins=[pt_bins, eta_bins])

    wb= np.ones(X[y == 5].shape[0])

    wc = [(b_bins[0] / c_bins[0])[arg] for arg in zip(
        np.digitize(X[y == 4, pt_col] / 1000, b_bins[1]) - 1, 
        np.digitize(X[y == 4, eta_col], b_bins[2]) - 1
    )]

    wl = [(b_bins[0] / l_bins[0])[arg] for arg in zip(
        np.digitize(X[y == 0, pt_col] / 1000, b_bins[1]) - 1, 
        np.digitize(X[y == 0, eta_col], b_bins[2]) - 1
    )]

    w = np.zeros(len(y))
    w[y == 5] = wb 
    w[y == 4] = wc
    w[y == 0] = wl
    return w

# ----------------------------------------------------------------- 

def reweight_to_l(X, y, pt_col, eta_col):
    '''
    Definition:
    -----------
        Reweight to light-distribution
    '''

    pt_bins = [10, 50, 100, 150, 200, 300, 500, 99999]
    eta_bins = np.linspace(0, 2.5, 6)

    b_bins = plt.hist2d(X[y == 5, pt_col] / 1000, X[y == 5, eta_col], bins=[pt_bins, eta_bins])
    c_bins = plt.hist2d(X[y == 4, pt_col] / 1000, X[y == 4, eta_col], bins=[pt_bins, eta_bins])
    l_bins = plt.hist2d(X[y == 0, pt_col] / 1000, X[y == 0, eta_col], bins=[pt_bins, eta_bins])

    wl= np.ones(X[y == 0].shape[0])

    wc = [(l_bins[0] / c_bins[0])[arg] for arg in zip(
        np.digitize(X[y == 4, pt_col] / 1000, l_bins[1]) - 1, 
        np.digitize(X[y == 4, eta_col], l_bins[2]) - 1
    )]

    wb = [(l_bins[0] / b_bins[0])[arg] for arg in zip(
        np.digitize(X[y == 5, pt_col] / 1000, l_bins[1]) - 1, 
        np.digitize(X[y == 5, eta_col], l_bins[2]) - 1
    )]

    w = np.zeros(len(y))
    w[y == 5] = wb 
    w[y == 4] = wc
    w[y == 0] = wl
    return w

# ----------------------------------------------------------------- 

def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(format="%(levelname)-8s\033[1m%(name)-21s\033[0m: %(message)s")
    logging.addLevelName(logging.WARNING, "\033[1;31m{:8}\033[1;0m".format(logging.getLevelName(logging.WARNING)))
    logging.addLevelName(logging.ERROR, "\033[1;35m{:8}\033[1;0m".format(logging.getLevelName(logging.ERROR)))
    logging.addLevelName(logging.INFO, "\033[1;32m{:8}\033[1;0m".format(logging.getLevelName(logging.INFO)))
    logging.addLevelName(logging.DEBUG, "\033[1;34m{:8}\033[1;0m".format(logging.getLevelName(logging.DEBUG)))

# ----------------------------------------------------------------- 

if __name__ == '__main__':
    import argparse
    # -- read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('iptagger', help="select 'ip3d' or 'ipmp'")
    args = parser.parse_args()

    sys.exit(main(args.iptagger))









