"""
This file contains the hyperparameters that affect the search space and the search method. Hyperparameters that can be
experimented with are called via main's argument parser.
"""

import sys
if not '../' in sys.path:
    sys.path.append('../')
import os
from pathlib import Path
from pytorch_model.Aggregation import NoAggregation, Addition, Concatenation
from pytorch_model.Operation import *
from pytorch_model.Blocks import *
from copy import copy

print('settings is here')
TASK = 'fashion-mnist'

# Use the blocks-based search space? Also drives initial number of layers of architectures and depth of
# initial layer
BLOCK_SEARCH_SPACE = False
# Allow skip connections?
SKIP_ALLOWED = False
# Training and search regimes -- fast for easy tasks, slower for complex tasks. Also drives PAD_METHOD
# and INTER_METHOD and data augmentation.
if TASK in ['mnist', 'fashion-mnist', 'stocks']:
    IS_COMPLEX = False
else:
    IS_COMPLEX = True

# Dummy training for networks? (for testing purposes)
DUMMY = False  # Change to False when not debugging
# Placeholder: do we use the performance predictor? (value set by main.py arguments):
PREDICT = None
# Paths
EXEC_PATH = Path(os.getcwd())
RUN_PATH, WTS_PATH, POPS_PATH, LOG_PATH, PRED_BATCH_PATH = None,None,None,None,None
PRED_BATCH_FILENAME = None
VAE_WTS_PATH = None
# Path to dataset
DATA_PATH = EXEC_PATH/Path(r"../datasets/clnn_data")  # valid for datasets that ship with Pytorch
# DATA_PATH = Path(r"G:/datasets/stock_prediction_n_dates")  # otherwise enter full path to data
# Constant to prevent division by zero
EPS = 1e-10


AGG_LOOKUP = {0: NoAggregation,
              1: Addition,
              2: Concatenation,
}

if not BLOCK_SEARCH_SPACE:
    OP_LOOKUP = {0: Identity,
                 1: Conv,
                 2: SepConv,
                 3: GrpConv,
                 4: Pool,
    }
    OP_PARAMS = {
        0: None,
        1: ['conv_k_size', 'batch_norm', 'activation'],  # Conv
        2: ['conv_k_size', 'batch_norm', 'activation'],  # SepConv
        3: ['conv_k_size', 'batch_norm', 'activation'],  # GrpConv
        4: ['max_or_avg', 'pool_k_size', 'ch_factor'],  # Pool
    }
else:
    OP_LOOKUP = {
        0: Identity,
        1: ResBasic,
        2: ResBottleneck,
        3: DenseBasic,
        4: DenseBottleneck,
        5: InceptionResnet_A,
        6: InceptionResnet_BC,
        7: Pool,
    }
    OP_PARAMS = {
        0: None,
        1: ['res_k_size', 'res_downsample'],  # ResBasic
        2: ['res_k_size', 'res_downsample'],  # ResBottleneck
        3: ['dense_growth_factor', 'dense_transition'],  # DenseBasic
        4: ['dense_growth_factor', 'dense_transition'],  # DenseBottleneck
        5: ['incres_k_size', 'incres_bottle_factor'],  # InceptionResnet-A
        6: ['incres_k_size', 'incres_bottle_factor'],  # InceptionResnet-BC
        7: ['max_or_avg', 'pool_k_size'],  # Pool
    }

MAX_OUTDEG_INPUT = 999
MAX_INDEG = 2  # Do not change these values inconsiderately! Untested behaviour could occur.
MAX_OUTDEG_HIDDEN = 2  # Max outdegree of hidden nodes (i.e. all except input node (Node 0) and classifier node)

# Default hyperparameters for defining Network topologies
DEFAULT_NTW_HPARAMS = dict(
    max_outdeg_input=999,
    max_indeg= 1 + SKIP_ALLOWED,
    max_outdeg_hidden= 1 + SKIP_ALLOWED,
    layers_ini=3 * (1 + BLOCK_SEARCH_SPACE),  # number of layers of initial architectures
    out_ch_0=64 // (1 + BLOCK_SEARCH_SPACE),  # number of channels out of the input node
)

# Hyperparameters for training and evaluating individual Networks during search
PARTIAL_EVAL_HPARAMS=dict(
    max_epochs=15 * (IS_COMPLEX + 1),  # !!!!!!!!!!!!!!!!!!!!!!
    lr_ini=0.1 / (IS_COMPLEX + 1),
    lr_min=1.e-7,
    betas=(0.95, 0.99),
    weight_decay=1.e-5,
    init_params=dict(method='he_normal', b_value=0.),  # use `he_normal` (`he_uniform` is dubious in Pytorch)
    grad_norm_clip=1.,
    n_restarts=0,  # no restarts during search, we want good anytime performance
    T_mult=1,
    early_stop=True,
    patience=2,
    thresh=0.005 if not IS_COMPLEX else 0.003,
    verbose=True,
    n_nodes=1,
    n_workers=4  # set to zero on a Windows machine (bug in Pytorch)
)

FINAL_EVAL_HPARAMS = copy(PARTIAL_EVAL_HPARAMS)  # hyperparameters for training final committee members
FINAL_EVAL_HPARAMS['max_epochs'] = 210 if IS_COMPLEX else 50
FINAL_EVAL_HPARAMS['lr_ini'] /= 3.
FINAL_EVAL_HPARAMS['lr_min'] = 0.
FINAL_EVAL_HPARAMS['n_restarts'] = 1 + IS_COMPLEX
FINAL_EVAL_HPARAMS['T_mult'] = 1 + IS_COMPLEX
FINAL_EVAL_HPARAMS['early_stop'] = False

DEFAULT_RSEARCH_HPARAMS = dict(  # default hyperparameters determining general run conditions for AIS algorithms
    layers_ini=3 * (1 + BLOCK_SEARCH_SPACE),  # number of layers of initial architectures
    out_ch_0=64 // (1 + BLOCK_SEARCH_SPACE),  # number of channels out of the input node
    max_walltime=999999,
    verbose=True,
)

DEFAULT_CLONALG_HPARAMS = dict(  # default hyperparameters determining general run conditions for AIS algorithms
    patience_inner=2,  # patience for inner loop stopping
    thresh_inner=0.005 if IS_COMPLEX else 0.0075,  # fitness threshold for inner loop stopping
    patience_outer=2,  # patience for outer loop early stopping
    thresh_outer=0.005 if IS_COMPLEX else 0.0075,  # fitness threshold for outer loop stopping
    insert_period=1,
    max_gens=40,  # max number of generations for training
    max_in_arch_ratio=0.95,
    use_special_fit='mean',  # None, 'mean' or 'adj'
    verbose=True,
    max_time=999999,  # for comparison with DeepSwarm
)

ALIGN_METHOD = 'inter'
PAD_METHOD = 'constant' if not IS_COMPLEX else 'reflect'  # change for CIFAR-10 experiments
INTER_METHOD = 'nearest' if not IS_COMPLEX else 'bilinear'  # nearest / bilinear

SEARCH_DATA_HPARAMS = dict(
    train_size=0.2,  # !!!!!!!!!!!!!!!!!!!!!! 0.2
    valid_size=0.2,  # !!!!!!!!!!!!!!!!!!!!!! 0.2
    test_size=1.,
    batch_size=64 if TASK == 'stocks' else 128 * (2 - IS_COMPLEX),
    augment_val=False,  # set to True if overfitting the validation set.
    cutout=IS_COMPLEX,
    pad_crop=False if TASK=='stocks' else True,
    pad_fill='zero' if not IS_COMPLEX else 'reflect',
    flip=IS_COMPLEX,
)
COMMITTEE_DATA_HPARAMS = dict(
    train_size=0.998, # !!!!!!!!!! 0.998
    valid_size=0.002,
    test_size = 1.,
    batch_size=64 * (2 - IS_COMPLEX),
    augment_val=False,
    whiten=False,
    cutout=IS_COMPLEX,
    pad_crop=False if TASK=='stocks' else True,
    pad_fill='zero' if not IS_COMPLEX else 'reflect',
    flip=IS_COMPLEX,
)

# Grammar-VAE hyperparameters
VAE_HPARAMS = dict(
    # Note: when using the integrated VAE/Predictor, all these keys will be renamed vae_*
    weights_path=None,
    max_depth=10,
    n_chars=29,
    #device='cuda',
    max_len=56,  # maximum absolute length of all strings produced, regardless of the number of layers in the arch.
                 # only use even numbers when using a CNN decoder. Set as close as possible as maximum number of
                 # productions that can be used with the specified max number of layers.
    weighted_sampling=True,
    temperature=-.1,  # softmax temperature for sampling sentence lengths; 0 = uniform distribution
    layer_symbol='LAY',
    # data_size=1,  # how many sequences constitute an epoch
    batch_size=256,  # must be >= 2
    latent_size=256,  # increasing latent size doesn't help convergence (but helps performance predictions)
    enc_type='cnn',  # 'cnn' or 'gru' or 'bi_gru
    dec_type='both',  # 'cnn', 'gru' or 'both'
    drop_rate=0.,
    length_encoding=True,

    # CNN encoder/decoder parameters
    channels='32,64,64,128,128',
    k_sizes ='1,3,3,5,5',
    strides ='1,2,1,2,1',
    fc_dim=512,

    # RNN decoder parameters
    rnn_hidden=512,
    rnn_layers=3,
    epsilon_std=0.01,  # stdev for the random variable sampled from N(0, std). Critical for performance

    # training parameters
    max_steps=50000,
    lr_ini=1e-4,
    lr_min=1e-6,
    lr_reduce_patience=1500,
    lr_reduce_factor=0.2,
    early_stop_patience=2500,
    kld_wt_zero_for_epochs=1000,
    kld_wt_one_at_epoch=2000,
    kld_zero_value=1.e-15,
    kld_full_value=0.5,
    num_workers=4,
    categorical_channel=10,
    cont_gamma= 30.,
    disc_gamma= 30.,
    alpha=30.,
    con_min_capacity = 0.,
    con_max_capacity = 25.,
    dis_min_capacity= 0.,
    dis_max_capacity= 25.,
    cont_iter=100,
    disc_iter=100,
    anneal_rate=3e-5,
    anneal_interval=100
)

# Performance predictor hyperparameters
PRED_HPARAMS = dict(
    # Note: when using the integrated VAE/Predictor, all these keys will be renamed pred_*
    model_type='ensemble',  # 'mlp', 'exu', 'both', 'ensemble'
    mixup=False,
    mixup_alpha = 0.4,
    mc_dropout=False,
    activation='relu',  # activation function in hidden layers only
    weights_path=None,
    warm_up_period=3,  # number of generations to wait for before starting to train the predictor
    max_depth=VAE_HPARAMS['max_depth'],
    in_dim=VAE_HPARAMS['latent_size'],
    c_dim=VAE_HPARAMS['categorical_channel'],
      # should be the same value as VAE_HPARAMS['latent_size']
    hid_dims='2048,2048,2048,1024',  # dimensions of the hidden fully-connected layers
    droprate_hidden=0.,
    droprate_head=0.2,
    max_epochs=20,  # number of epochs per training round
    batch_sz=128,
    w_decay=1.e-5,
    lr_ini=1e-4,
    lr_reduce_on_plateau_factor=.5,  # only used in independent experiments, not in NAS
    lr_reduce_patience=100,  # only used in independent experiments, not in NAS
    lr_reduce_mult_factor=0.99,
    lr_min=1.e-6,
    val_set_pct=0.,  # set to >0. in independent experiments but set to 0. when running NAS (not enough data)
    clone_split_ratio=.67,  # proportion of clones to train
    batch_gen_n=5,  # The number of generations of samples kept in fitness batch to train predictor,
    num_workers=4,
)
