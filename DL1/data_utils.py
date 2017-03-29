import numpy as np
import matplotlib.pyplot as plt


def replaceInfNaN(x, value):
    '''
    replace Inf and NaN with a default value
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
    '''
    Apply recommended cuts for Akt4EMTopoJets
    '''
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
        Reweight to b-distribution in eta and pt
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

    # -- hardcoded, standard flavor fractions
    C_FRAC = 0.07
    L_FRAC = 0.61
    n_light = wl.sum()
    n_charm = (n_light * C_FRAC) / L_FRAC
    n_bottom = (n_light * (1 - L_FRAC - C_FRAC)) / L_FRAC

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
        Reweight to light-distribution in eta and pt
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

    # -- hardcoded, standard flavor fractions
    C_FRAC = 0.07
    L_FRAC = 0.61
    n_light = wl.sum()
    n_charm = (n_light * C_FRAC) / L_FRAC
    n_bottom = (n_light * (1 - L_FRAC - C_FRAC)) / L_FRAC

    w = np.zeros(len(y))
    w[y == 5] = wb 
    w[y == 4] = wc
    w[y == 0] = wl
    return w

# ----------------------------------------------------------------- 