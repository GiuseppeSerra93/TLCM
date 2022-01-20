#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=utils.K, help='Number of user latent classes')
parser.add_argument("--L", type=int, default=utils.L, help='Number of product latent classes')
args = parser.parse_args()

K = args.K
L = args.L
dir_preprocessed_data = utils.DIR_PREPROCESSED_DATA
dir_training_data = utils.DIR_TRAIN_TEST_DATA
dir_results = utils.DIR_RESULTS

categories = utils.CATEGORIES

def find_missing_idx(test_list, train_list):
    missing_values = set(test_list).difference(set(train_list))
    missing_idx = [i for i, j in enumerate(test_list) if j in missing_values]
    return missing_idx


for category in categories:
    print("Processing {0} category:".format(category))
    dir_preprocessed_data_c = dir_preprocessed_data + category + '/'
    dir_training_data_c = dir_training_data + category + '/'
    dir_results_c = dir_results + category + '/results_EM_{0}_{1}/'.format(K, L)
    dir_rating_data = dir_training_data_c + 'rating_data_{0}_{1}/'.format(K, L)
    
    ratings = pickle.load(open(dir_preprocessed_data_c + 'ratings.pkl', 'rb'))
    reviews = pickle.load(open(dir_preprocessed_data_c + 'reviews_input.pkl', 'rb'))

    users_map = pickle.load(open(dir_preprocessed_data_c + 'users_map.pkl', 'rb'))  # {user_ID: idx_u}
    products_map = pickle.load(open(dir_preprocessed_data_c + 'products_map.pkl', 'rb')) # {prod_ID: idx_p}
    inv_users_map = {v:k for k,v in users_map.items()} # {idx_u: user_ID}
    inv_products_map = {v:k for k,v in products_map.items()} # {idx_p: prod_ID}

    p_zu_u = pickle.load(open(dir_results_c + 'p_zu_u.pkl', 'rb'))
    p_zp_p = pickle.load(open(dir_results_c + 'p_zp_p.pkl', 'rb'))
    
    print('\tp_zu_u.shape[0] == len(users_map)',p_zu_u.shape[0] == len(users_map))
    print('\tp_zp_p.shape[0] == len(products_map)',p_zp_p.shape[0] == len(products_map))
    print('\tlen(reviews)==len(ratings)',len(reviews)==len(ratings))
    print('\tlen(reviews)',len(reviews))
    
    random_seed = 7
    random.seed(random_seed)
    random.shuffle(reviews)
    random.seed(random_seed)
    random.shuffle(ratings)

    features_input = []
    images_u = []
    images_p = []
    user_ids = []
    prod_ids = []
    for i in range(len(reviews)):
        review = reviews[i]
        rating = ratings[i]

        user_id = review[0]
        prod_id = review[1]
        features = np.concatenate([p_zu_u[user_id],p_zp_p[prod_id]])
        images_u.append(p_zu_u[user_id])
        images_p.append(p_zp_p[prod_id])
        features_input.append(features)
        
        user_ids.append(user_id)
        prod_ids.append(prod_id)

    features_input = np.array(features_input)
    ratings = np.array(ratings, dtype=np.int64)
    
    X_train, X_test, y_train, y_test = train_test_split(features_input, ratings, test_size=.2, random_state=random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.5, random_state=random_seed)
    
    user_train, user_test, prod_train, prod_test = train_test_split(user_ids, prod_ids, test_size=.2, random_state=random_seed)
    user_val, user_test, prod_val, prod_test = train_test_split(user_test, prod_test, test_size=.5, random_state=random_seed)
    
    missing_values_prod = find_missing_idx(prod_val, prod_train)
    missing_values_user = find_missing_idx(user_val, user_train)
    missing_values_val = list(set(missing_values_prod).union(set(missing_values_user)))

    missing_values_prod = find_missing_idx(prod_test, prod_train)
    missing_values_user = find_missing_idx(user_test, user_train)
    missing_values_test = list(set(missing_values_prod).union(set(missing_values_user)))    
    
    images_u = np.array(images_u).reshape(len(reviews), int(np.sqrt(K)), int(np.sqrt(K)), 1)
    images_p = np.array(images_p).reshape(len(reviews), int(np.sqrt(L)), int(np.sqrt(L)), 1)
    train_u, test_u, train_p, test_p = train_test_split(images_u, images_p, test_size=.2, random_state=random_seed)
    val_u, test_u, val_p, test_p = train_test_split(test_u, test_p, test_size=.5, random_state=random_seed)
    
    if len(missing_values_val)>0:
        print('\tMissing values validation set')
        val_u = np.delete(val_u, missing_values_val, axis=0)
        val_p = np.delete(val_p, missing_values_val, axis=0)
        X_val = np.delete(X_val, missing_values_val, axis=0)
        y_val = np.delete(y_val, missing_values_val, axis=0)
        user_val = np.delete(user_val, missing_values_val, axis=0)
        prod_val = np.delete(prod_val, missing_values_val, axis=0)
        
    if len(missing_values_test)>0:
        print('\tMissing values test set')
        test_u = np.delete(test_u, missing_values_test, axis=0)
        test_p = np.delete(test_p, missing_values_test, axis=0)
        X_test = np.delete(X_test, missing_values_test, axis=0)
        y_test = np.delete(y_test, missing_values_test, axis=0)
        user_test = np.delete(user_test, missing_values_val, axis=0)
        prod_test = np.delete(prod_test, missing_values_val, axis=0)

    print('\t{} {} {}'.format(*[X_train.shape, X_test.shape, X_val.shape]))
    print('\t{} {} {}'.format(*[train_u.shape, test_u.shape, val_u.shape]))
    print('\t{} {} {}'.format(*[train_p.shape, test_p.shape, val_p.shape]))
    print('\ttrain_u[0][0][0]==X_train[0][0]',train_u[0][0][0]==X_train[0][0])
    print('\ttrain_p[0][0][0]==X_train[0][K]',train_p[0][0][0]==X_train[0][K])
    print('\ttest_u[0][0][0]==X_test[0][0]',test_u[0][0][0]==X_test[0][0])
    print('\ttest_p[0][0][0]==X_test[0][K]',test_p[0][0][0]==X_test[0][K])
    print('\tval_u[0][0][0]==X_val[0][0]',val_u[0][0][0]==X_val[0][0])
    print('\tval_p[0][0][0]==X_val[0][K]',val_p[0][0][0]==X_val[0][K])
    print()
    
    print('\tSaving files...')
    if not os.path.exists(dir_rating_data):  # create directory if it does not exist
        print('\tCreate new output directory:', dir_rating_data)
        os.makedirs(dir_rating_data)
    print()
        
    out_filename = dir_rating_data + 'X_train.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(X_train, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'X_test.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(X_test, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'X_val.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(X_val, outfile)
        outfile.close()

    out_filename = dir_rating_data + 'y_train.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(y_train, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'y_test.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(y_test, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'y_val.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(y_val, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'train_u.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(train_u, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'test_u.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(test_u, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'val_u.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(val_u, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'train_p.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(train_p, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'test_p.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(test_p, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'val_p.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(val_p, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'user_train.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(user_train, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'user_test.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(user_test, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'user_val.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(user_val, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'prod_train.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(prod_train, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'prod_test.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(prod_test, outfile)
        outfile.close()
        
    out_filename = dir_rating_data + 'prod_val.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(prod_val, outfile)
        outfile.close()
        
    fig, axs = plt.subplots(1, 3, figsize=(12,3))
    axs[0].bar(Counter(y_train).keys(), Counter(y_train).values())
    axs[1].bar(Counter(y_test).keys(), Counter(y_test).values())
    axs[2].bar(Counter(y_val).keys(), Counter(y_val).values())
    axs[0].set_xlabel('Rating values')
    axs[1].set_xlabel('Rating values')
    axs[2].set_xlabel('Rating values')
    axs[0].set_title('Train')
    axs[1].set_title('Test')
    axs[2].set_title('Validation')
    axs[0].set_ylabel('Count')
    plt.savefig(dir_rating_data + 'rating_distr.png', format='png', transparent=True, dpi=200, bbox_inches='tight')
    plt.close()