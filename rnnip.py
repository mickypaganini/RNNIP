''' 
'''
import numpy as np
import sys
import cPickle
import os
import logging
from errno import EEXIST

import matplotlib
matplotlib.use('TkAgg')

import deepdish.io as io

from keras.layers import GRU, LSTM, Dense, Dropout, Activation, Masking, Embedding, merge, Input, Flatten
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.constraints import unitnorm

import pandautils as pup

def safe_mkdir(path):
    '''
    Safe mkdir (i.e., don't create if already exists, 
    and no violation of race conditions)
    '''
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception


def configure_logging():
    '''
    Configure pretty, colerful logger
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(format="%(levelname)-8s\033[1m%(name)-21s\033[0m: %(message)s")
    logging.addLevelName(logging.WARNING,
        "\033[1;31m{:8}\033[1;0m".format(logging.getLevelName(logging.WARNING)))
    logging.addLevelName(logging.ERROR,
        "\033[1;35m{:8}\033[1;0m".format(logging.getLevelName(logging.ERROR)))
    logging.addLevelName(logging.INFO,
        "\033[1;32m{:8}\033[1;0m".format(logging.getLevelName(logging.INFO)))
    logging.addLevelName(logging.DEBUG,
        "\033[1;34m{:8}\033[1;0m".format(logging.getLevelName(logging.DEBUG)))


def main(embed_size, normed, input_id, run_name):

    configure_logging()
    logger = logging.getLogger("RNNIP Training")

    logger.info("Loading hdf5's")
    test_dict = io.load(os.path.join('data', 'test_dict_' + input_id + '.h5'))
    train_dict = io.load(os.path.join('data', 'train_dict_' + input_id + '.h5'))
    
    X_train_stream0 = train_dict['grade']
    X_train_stream1 = train_dict['X']
    y_train = train_dict['y']    

    X_test_stream0 = test_dict['grade']
    X_test_stream1 = test_dict['X']
    y_test = test_dict['y']

    ip3d = test_dict['ip3d'] 

    logger.info('Building model')
    model = build_model(X_train_stream0, X_train_stream1, embed_size, normed)
    model.summary()

    logger.info('Compiling model')
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    #-- if the pre-trained model exists, load it in, otherwise start from scratch
    safe_mkdir('weights')
    weights_file = os.path.join('weights', 'rnnip_' + run_name +'.h5')
    try:
        model.load_weights(weights_file)
        logger.info('Loaded pre-trained model from ' + weights_file)
    except IOError:
        logger.info('No pre-trained model found in ' + weights_file)

    logger.info('Training:')
    try:
        model.fit([X_train_stream0, X_train_stream1], y_train, batch_size=512,
            callbacks = [
                EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                ModelCheckpoint(
                    weights_file, 
                    monitor='val_loss', verbose=True, save_best_only=True
                )
            ],
        epochs=300, 
        validation_split = 0.2) 
        
    except KeyboardInterrupt:
        logger.info('Training ended early.')

    # -- load in best network
    logger.info('Loading best epoch')
    model.load_weights(weights_file)

    json_string = model.to_json()
    safe_mkdir('json_models')
    open(os.path.join('json_models', run_name +'.json'), 'w').write(json_string)

    logger.info('Testing')
    safe_mkdir('predictions')
    yhat = model.predict([X_test_stream0, X_test_stream1], verbose=True, batch_size=10000) 
    io.save(os.path.join('predictions', 'yhat'+ run_name +'.h5'), yhat) 
     
    logger.info('Plotting ROC')
    plot_ROC(y_test, yhat, ip3d, run_name)
     

def build_model(X_train_stream0, X_train_stream1, embed_size, normed):
    n_grade_categories = len(np.unique(pup.flatten(X_train_stream0)))
    grade_input = Input(shape=X_train_stream0.shape[1:], dtype='int32', name='grade_input')
    track_input = Input(shape=X_train_stream1.shape[1:], name='track_input')
    track_masked = Masking(mask_value=-999)(track_input)

    # -- grade embedding
    if normed:
        embedded_grade = Embedding(
            input_dim=n_grade_categories, output_dim=embed_size, mask_zero=True,
            input_length=X_train_stream0.shape[1], 
            W_constraint=unitnorm(axis=1))(Flatten()(grade_input))
    else:
        embedded_grade = Embedding(
            input_dim=n_grade_categories, output_dim=embed_size, mask_zero=True,
            input_length=X_train_stream0.shape[1])(Flatten()(grade_input))

    #x = Concatenate([embedded_grade, track_input])
    x = merge([embedded_grade, track_masked], mode='concat')
    x = LSTM(25, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    y = Dense(4, activation='softmax')(x)
    return Model(inputs=[grade_input, track_input], outputs=y)


def plot_ROC(y_test, yhat, ip3d, run_name):
    ''' 
    Args:
    -----
        y_test: the truth labels for the test set
        yhat: the predicted probabilities of each class in the test set
        ip3d:
        run_name:
    '''
    from viz import calculate_roc, ROC_plotter, add_curve
    logger = logging.getLogger("plot ROC")

    # -- bring classes back to usual format: [0,2,3,0,1,2,0,2,2,...]
    y = np.array([np.argmax(ev) for ev in y_test])
    
    # -- for b VS. light
    bl_sel = (y == 0) | (y == 2)
    # -- for c VS. light
    cl_sel = (y == 0) | (y == 1)
    # -- for b VS. c
    bc_sel = (y == 1) | (y == 2)

    # -- add ROC curves
    discs = {}
    add_curve(r'IP3D', 'black', 
        calculate_roc(
            (y[ bl_sel & np.isfinite(np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pu).values) ] == 2), 
            np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pu)[ bl_sel & np.isfinite(np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pu).values) ]
        ),
        discs
    )
    add_curve('RNNIP', 'blue', 
        calculate_roc(
            (y[ bl_sel & np.isfinite(np.log(yhat[:,2] / yhat[:,0]))] == 2), 
            np.log(yhat[:,2] / yhat[:,0])[bl_sel &  np.isfinite(np.log(yhat[:,2] / yhat[:,0]))]
        ), 
        discs
    )

    discs_bc = {}
    add_curve(r'IP3D', 'black', 
        calculate_roc(
            (y[ bc_sel & np.isfinite(np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pc).values) ] == 2), 
            np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pc)[ bc_sel & np.isfinite(np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pc).values) ]
        ), 
        discs_bc
    ) 
    add_curve('RNNIP', 'blue',
        calculate_roc(
            (y[ bc_sel & np.isfinite(np.log(yhat[:,2] / yhat[:,1]))] == 2), 
            np.log(yhat[:,2] / yhat[:,1])[bc_sel &  np.isfinite(np.log(yhat[:,2] / yhat[:,1]))]
        ), 
        discs_bc
    )
    logger.info('Pickling ROC curves')
    safe_mkdir('roc_pickles')
    cPickle.dump(
        discs['RNNIP'], open(os.path.join('roc_pickles', run_name +'_bl.pkl'), 'wb'),
        cPickle.HIGHEST_PROTOCOL
    )
    cPickle.dump(
        discs_bc['RNNIP'], open(os.path.join('roc_pickles', run_name +'_bc.pkl'), 'wb'),
        cPickle.HIGHEST_PROTOCOL
    )
    logger.info('Plotting')
    safe_mkdir('plots')
    fg = ROC_plotter(
        discs, title=r'Impact Parameter Taggers',
        min_eff = 0.5, max_eff=1.0, logscale=True
    )
    fg.savefig(os.path.join('plots', 'roc' + run_name +'.pdf'))
    fg = ROC_plotter(
        discs_bc, title=r'Impact Parameter Taggers',
        min_eff = 0.5, max_eff=1.0, logscale=True, ymax=10**2
    )
    fg.savefig(os.path.join('plots', 'roc' + run_name +'_bc.pdf'))


if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_size',
        type=int,
        help='output dimension of the embedding layer')
    parser.add_argument('--input_id',
        help='string that identifies the name of the hdf5 file to load'
        'corresponding to --output argument in dataprocessing.py')
    parser.add_argument('--run_name',
        help='string that identifies this specific run, used to label outputs')
    parser.add_argument('--normed',
        action='store_true', default=False,
        help='Pass this flag if you want to normalize the embedding output')
    args = parser.parse_args()
    sys.exit(main(args.embed_size, args.normed, args.input_id, args.run_name))




