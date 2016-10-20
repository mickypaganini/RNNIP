'''
Info:
    This script loads root files containing standard 
    b-tagging information + the IPMP outputs. 
    It turns the data into jet-flat structure,
    replicates the variable creation and modification
    that are present in MV2, scales the variables 
    and splits them into training and testing sets.
    Finally, the data is stored as dictionaries in
    HDF5 format.Parallelized using joblib.
Author: 
    Michela Paganini - Yale/CERN
    michela.paganini@cern.ch
Example:
    python parallel_generate_data_DL1 ./variables.yaml ../data/final_production/*.root
'''
import glob
import pandas as pd
import numpy as np
import math
import os
import sys
import logging
import yaml
import deepdish.io as io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

C_FRAC = 0.07
L_FRAC = 0.61
OUTNAME = '_paralleltrial_'

def main(yaml_file, root_paths):
    '''
    Args:
    -----
        root_paths: list of strings with the root file paths
    Returns:
    --------
        n_train_events: total number of events for training, across all root files
        n_test_events: total number of events for testing, across all root files
        n_validate_events: total number of events for validating, across all root files
    Alternatively, you can return the paths to hdf5 files being created, for logging
    '''
    # -- logging
    configure_logging()
    logger = logging.getLogger("parallel_generate_data_DL1")
    logger.debug('Files to process: {}'.format(root_paths))

    # -- open and process files in parallel
    from joblib import Parallel, delayed
    n_events = Parallel(n_jobs=-1, verbose=5, backend="multiprocessing") \
        (delayed(process)(i, filepath, yaml_file) for i, filepath in enumerate(root_paths))

    # -- add up events in the list of results to get the total number of events per type 
    n_train_events = sum(zip(*n_events)[0])
    n_test_events = sum(zip(*n_events)[1])
    n_validate_events = sum(zip(*n_events)[2])
    logger.info('There are {n_train_events} training events, {n_test_events} testing events,\
        and {n_validate_events} validating events'.format(
            n_train_events=n_train_events,
            n_test_events=n_test_events,
            n_validate_events=n_validate_events
            )
        )
    return n_train_events, n_test_events, n_validate_events

    # -- Alternatively, you can return the paths to hdf5 files being created, for logging
    # hdf5_paths = Parallel(n_jobs=-1, verbose=5, backend="multiprocessing") \
    #     (delayed(f)(i, filepath) for i, filepath in enumerate(root_paths))
    # logger.debug('Saved the following hdf5 archives: {}'.format(hdf5_paths))
    # return hdf5_paths

# -----------------------------------------------------------------

def process(i, filepath, yaml_file): 
    '''
    '''   
    import pandautils as pup

    branches, training_vars, ip3d_training_vars, ipmp_training_vars = set_features(yaml_file)
    logger = logging.getLogger("ETL Service")
    logger.info('Operating on {}'.format(filepath))
    logger.info('Creating dataframe...')
    df = pup.root2panda(filepath, 'bTag_AntiKt4EMTopoJets', branches=branches)

    logger.info('Transforming variables...')
    df = transformVars(df)

    logger.info('Flattening df...')
    df.drop(['PVx', 'PVy', 'PVz'], axis=1, inplace=True)
    df_flat = pd.DataFrame({k: pup.flatten(c) for k, c in df.iteritems()})
    del df

    logger.info('Applying cuts...')
    df_flat = apply_calojet_cuts(df_flat)

    logger.info('Creating X, y, w, mv2c10...')
    y = df_flat['jet_LabDr_HadF'].values
    mv2c10 = df_flat['jet_mv2c10'].values
    jet_pt = df_flat['jet_pt'].values
    ip3d_vars = df_flat[ip3d_training_vars].values
    ipmp_vars = df_flat[ipmp_training_vars].values
    # -- slice df by only keeping the training variables
    X = df_flat[training_vars].values

    # -- Find weights by reweighting to the light distribution
    # -- TO DO: pass the pt and eta columns directly, instead of passing their indices
    pt_col = np.argwhere(np.array(training_vars) == 'jet_pt')[0][0]
    eta_col = np.argwhere(np.array(training_vars) == 'abs(jet_eta)')[0][0]
    #w = reweight_to_b(X, y, pt_col, eta_col)
    w = reweight_to_l(X, y, pt_col, eta_col)
    del df_flat

    logger.info('Shuffling, splitting, scaling...')
    ix = np.array(range(len(y)))
    X_train, X_test, y_train, y_test, w_train, w_test, ix_train, ix_test, \
    mv2c10_train, mv2c10_test, jet_pt_train, jet_pt_test, \
    ip3d_vars_train, ip3d_vars_test, ipmp_vars_train, ipmp_vars_test = train_test_split(
        X,
        y,
        w,
        ix,
        mv2c10,
        jet_pt,
        ip3d_vars,
        ipmp_vars,
        train_size=0.6)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    ip3d_vars_train = scaler.fit_transform(ip3d_vars_train)
    ip3d_vars_test = scaler.transform(ip3d_vars_test)
    ipmp_vars_train = scaler.fit_transform(ipmp_vars_train)
    ipmp_vars_test = scaler.transform(ipmp_vars_test)

    X_train, X_validate, y_train, y_validate, w_train, w_validate, ix_train, ix_validate, \
    mv2c10_train, mv2c10_validate, jet_pt_train, jet_pt_validate, \
    ip3d_vars_train, ip3d_vars_validate, ipmp_vars_train, ipmp_vars_validate = train_test_split(
        X_train,
        y_train,
        w_train,
        ix_train,
        mv2c10_train,
        jet_pt_train,
        ip3d_vars_train,
        ipmp_vars_train,
        train_size=0.7)

    train = {
        'X' : X_train,
        'ip3d_vars': ip3d_vars_train,
        'ipmp_vars': ipmp_vars_train,
        'y' : y_train,
        'w' : w_train,
        'ix': ix_train,
        'mv2c10': mv2c10_train,
        'pt': jet_pt_train
    }

    test = {
        'X' : X_test,
        'ip3d_vars': ip3d_vars_test,
        'ipmp_vars': ipmp_vars_test,
        'y' : y_test,
        'w' : w_test,
        'ix': ix_test,
        'mv2c10': mv2c10_test,
        'pt': jet_pt_test
    }

    validate = {
        'X' : X_validate,
        'ip3d_vars': ip3d_vars_validate,
        'ipmp_vars': ipmp_vars_validate,
        'y' : y_validate,
        'w' : w_validate,
        'ix': ix_validate,
        'mv2c10': mv2c10_validate,
        'pt': jet_pt_validate
    }

    logger.info('Saving dictionaries to hdf5...')
    hdf5_train_path = os.path.join('..', 'data', 'DL1-' + OUTNAME + str(i) +'-train-db.h5')
    hdf5_test_path = os.path.join('..', 'data', 'DL1-' + OUTNAME + str(i) +'-test-db.h5')
    hdf5_validate_path = os.path.join('..', 'data', 'DL1-' + OUTNAME + str(i) +'-validate-db.h5')

    io.save(hdf5_train_path, train)
    io.save(hdf5_test_path, test)
    io.save(hdf5_validate_path, validate)
    logger.debug('Saved hdf5 archives: {}, {}, {}'. format(hdf5_train_path, hdf5_test_path, hdf5_validate_path))

    return (y_train.shape[0], y_test.shape[0], y_validate.shape[0])
    #return (hdf5_train_path, hdf5_test_path, hdf5_validate_path)

# -----------------------------------------------------------------    

def set_features(yaml_file):
    '''
    Info:
    -----
        Load names of branches to use from a yaml file
        This will contain 4 entries: 'branches', 'training_vars', 'ip3d_training_vars', 'ipmp_training_vars'
        - 'branches': list of names of the branches to directly extract from the TTree
        - 'training_vars': list of names of variables to always be used for learning
        - 'ip3d_training_vars': list of names of variables to be used for\
        learning only if we want to include the ip3d vars
        - 'ipmp_training_vars': list of names of variables to be used for\
        learning only if we want to include the ipmp vars
    Returns:
    --------
    '''
    with open(yaml_file, 'r') as stream:
        try:
            s = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return s['branches'], s['training_vars'], s['ip3d_training_vars'], s['ipmp_training_vars']

# ----------------------------------------------------------------- 

def transformVars(df):
    '''
    modifies the variables to create the ones that mv2 uses, inserts default values when needed, saves new variables
    in the dataframe
    Args:
    -----
        df: pandas dataframe containing all the interesting variables as extracted from the .root file
    Returns:
    --------
        modified mv2-compliant dataframe
    '''
    from rootpy.vector import LorentzVector, Vector3
    import pandautils as pup

    # -- modify features and set default values
    df['abs(jet_eta)'] = abs(df['jet_eta'])

    # -- create new IPxD features
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
    
    # -- create new IPMP features
    for (pu,pb,pc) in zip(df['jet_ipmp_pu'],df['jet_ipmp_pb'],df['jet_ipmp_pc']) :
        pu[pu >= 10] = -1
        pb[pu >= 10] = -1
        pc[pu >= 10] = -1 
    df['jet_ip'] = (df['jet_ipmp_pb'] / df['jet_ipmp_pu']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
    df['jet_ip_c'] = (df['jet_ipmp_pb'] / df['jet_ipmp_pc']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))
    df['jet_ip_cu'] = (df['jet_ipmp_pc'] / df['jet_ipmp_pu']).apply(lambda x : np.log( x )).apply(lambda x: _replaceInfNaN(x, -20))

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

    b_bins = np.histogram2d(X[y == 5, pt_col] / 1000, X[y == 5, eta_col], bins=[pt_bins, eta_bins])
    c_bins = np.histogram2d(X[y == 4, pt_col] / 1000, X[y == 4, eta_col], bins=[pt_bins, eta_bins])
    l_bins = np.histogram2d(X[y == 0, pt_col] / 1000, X[y == 0, eta_col], bins=[pt_bins, eta_bins])

    wb= np.ones(X[y == 5].shape[0])

    wc = [(b_bins[0] / c_bins[0])[arg] for arg in zip(
        np.digitize(X[y == 4, pt_col] / 1000, b_bins[1]) - 1, 
        np.digitize(X[y == 4, eta_col], b_bins[2]) - 1
    )]

    wl = [(b_bins[0] / l_bins[0])[arg] for arg in zip(
        np.digitize(X[y == 0, pt_col] / 1000, b_bins[1]) - 1, 
        np.digitize(X[y == 0, eta_col], b_bins[2]) - 1
    )]

    C_FRAC = 0.07
    L_FRAC = 0.61
    n_light = wl.sum()
    n_charm = (n_light * C_FRAC) / L_FRAC
    n_bottom = (n_light * (1 - L_FRAC - C_FRAC)) / L_FRAC

    w = np.zeros(len(y))
    w[y == 5] = np.array(wb) * (n_bottom / sum(wb)) 
    w[y == 4] = np.array(wc) * (n_charm / sum(wc))
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

    b_bins = np.histogram2d(X[y == 5, pt_col] / 1000, X[y == 5, eta_col], bins=[pt_bins, eta_bins])
    c_bins = np.histogram2d(X[y == 4, pt_col] / 1000, X[y == 4, eta_col], bins=[pt_bins, eta_bins])
    l_bins = np.histogram2d(X[y == 0, pt_col] / 1000, X[y == 0, eta_col], bins=[pt_bins, eta_bins])

    wl= np.ones(X[y == 0].shape[0])

    wc = [(l_bins[0] / c_bins[0])[arg] for arg in zip(
        np.digitize(X[y == 4, pt_col] / 1000, l_bins[1]) - 1, 
        np.digitize(X[y == 4, eta_col], l_bins[2]) - 1
    )]

    wb = [(l_bins[0] / b_bins[0])[arg] for arg in zip(
        np.digitize(X[y == 5, pt_col] / 1000, l_bins[1]) - 1, 
        np.digitize(X[y == 5, eta_col], l_bins[2]) - 1
    )]

    n_light = wl.sum()
    n_charm = (n_light * C_FRAC) / L_FRAC
    n_bottom = (n_light * (1 - L_FRAC - C_FRAC)) / L_FRAC

    w = np.zeros(len(y))
    w[y == 5] = np.array(wb) * (n_bottom / sum(wb)) 
    w[y == 4] = np.array(wc) * (n_charm / sum(wc)) 
    w[y == 0] = wl
    return w

# ----------------------------------------------------------------- 

def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(format="%(levelname)-8s\033[1m%(name)-21s[pid=%(process)d]\033[0m: %(message)s")
    logging.addLevelName(logging.WARNING, "\033[1;31m{:8}\033[1;0m".format(logging.getLevelName(logging.WARNING)))
    logging.addLevelName(logging.ERROR, "\033[1;35m{:8}\033[1;0m".format(logging.getLevelName(logging.ERROR)))
    logging.addLevelName(logging.INFO, "\033[1;32m{:8}\033[1;0m".format(logging.getLevelName(logging.INFO)))
    logging.addLevelName(logging.DEBUG, "\033[1;34m{:8}\033[1;0m".format(logging.getLevelName(logging.DEBUG)))

# ----------------------------------------------------------------- 

if __name__ == '__main__':
    import argparse
    # -- read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('variables', type=str, help="path to the yaml file containing lists named\
        'branches','training_vars','ip3d_training_vars','ipmp_training_vars'") # maybe add extra explanation to this
    parser.add_argument('input', type=str, nargs="+", help="e.g.: /path/to/pattern*.root")
    args = parser.parse_args()

    sys.exit(main(args.variables, args.input))
