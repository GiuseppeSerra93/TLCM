# -*- coding: utf-8 -*-
import os

# Paths of working directories
DIR_CWD = os.getcwd()
DIR_DATA = DIR_CWD + '/data/'
DIR_PREPROCESSED_DATA = DIR_CWD + '/preprocessed_data/'
DIR_TRAIN_TEST_DATA = DIR_CWD + '/train_test_data/'
DIR_RESULTS = DIR_CWD + '/output_data/'

# dir_results = dir_data + category + '/results_test_{0}_{1}/'.format(K, L)

V = 2000 # vocabulary dimensionality

SIGMA_USERS = 4.0       # concentration parameter for user channel noise
SIGMA_PRODUCTS = 3.0    # concentration parameter for product channel noise
K = 25  # number of user latent classes
L= 16   # number of product latent classes

# parameters for softening the hard-assignment for P(z_u|u) and P(z_p|p)
A_USERS = 0.3
A_PRODUCTS = 0.3
B_USERS = (1-A_USERS)/(K-1)
B_PRODUCTS = (1-A_PRODUCTS)/(L-1)

# parameters for EM training
TOLERANCE = 5e-4    # tolerance
MAX_ITER = 50       # max number of iteration
MAX_PATIENCE = 5    # patience steps

# parameters for rating prediction part
NUM_EPOCHS = 200
BATCH_SIZE = 256
LEARNING_RATE = 0.05
NUM_RUNS = 5

CATEGORIES = ['automotive', 'baby', 'beauty',
              'cellphones', 'digital_music', 'food', 'health',
              'instant_videos', 'musical_instruments', 'office',
              'patio', 'pet', 'sports', 'tools', 'toys', 'videogames']
CATEGORIES = ['automotive']
