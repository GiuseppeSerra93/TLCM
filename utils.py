# /*
#  *     THIS FILE BELONGS TO THE PROGRAM: TLCM 
#  *
#  *        File: utils.py
#  *
#  *     Authors: Deleted for purposes of anonymity 
#  *
#  *     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
#  * 
#  * The software and its source code contain valuable trade secrets and shall be maintained in
#  * confidence and treated as confidential information. The software may only be used for 
#  * evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
#  * license agreement or nondisclosure agreement with the proprietor of the software. 
#  * Any unauthorized publication, transfer to third parties, or duplication of the object or
#  * source code---either totally or in part---is strictly prohibited.
#  *
#  *     Copyright (c) 2021 Proprietor: Deleted for purposes of anonymity
#  *     All Rights Reserved.
#  *
#  * THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR 
#  * IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY 
#  * AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT 
#  * DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION. 
#  * 
#  * NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
#  * IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE 
#  * LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
#  * FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
#  * OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
#  * ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
#  * TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
#  * THE POSSIBILITY OF SUCH DAMAGES.
#  * 
#  * For purposes of anonymity, the identity of the proprietor is not given herewith. 
#  * The identity of the proprietor will be given once the review of the 
#  * conference submission is completed. 
#  *
#  * THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#  */

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
MAX_PATIENCE = 10    # patience steps

# parameters for rating prediction part
NUM_EPOCHS = 200
BATCH_SIZE = 256
LEARNING_RATE = 0.02
NUM_RUNS = 5

CATEGORIES = ['automotive', 'baby', 'beauty',
              'cellphones', 'digital_music', 'food', 'health',
              'instant_videos', 'musical_instruments', 'office',
              'patio', 'pet', 'sports', 'tools', 'toys', 'videogames']
CATEGORIES = ['patio']
