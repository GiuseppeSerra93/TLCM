# /*
#  *     THIS FILE BELONGS TO THE PROGRAM: TLCM 
#  *
#  *        File: build_dataset.py
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

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pickle
import os
import random
from collections import Counter, defaultdict
from tqdm import tqdm
import utils

def create_training_data(reviews, dir_output, max_len=20):
    random.shuffle(reviews)     # in-place shuffling
    
    # randomly select two common users and remove them from list
    # common, i.e., with at least 10 reviews
    user_count = Counter([r[0] for r in reviews])
    most_common_users = [i for i,j in user_count.most_common() if j == 10]
    random.shuffle(most_common_users)
    users_test = [most_common_users.pop() for idx in range(2)]

    user_IDs = []
    product_IDs = []
    words = []
    len_reviews = []

    ratings = []
    reviews_input = [] # list of [user_id, prod_id, review]
    reviews_test = []  # list of [user_id, prod_id, review] for the out-of-sample extension
    
    users_mapping = {}
    products_mapping = {}
    idx_u = 0
    idx_p = 0

    for line in tqdm(reviews):
        user_ID = line[0]
        product_ID = line[1]
        review = line[2][:max_len]
        rating = line[3]
        length = line[4]
        
        if user_ID in users_test:
            reviews_test.append([user_ID, product_ID, [w-1 for w in review]])

        else:
            if users_mapping.get(user_ID) == None:
                users_mapping[user_ID] = idx_u
                idx_u += 1

            if products_mapping.get(product_ID) == None:
                products_mapping[product_ID] = idx_p
                idx_p += 1

            idx_user = users_mapping[user_ID]
            idx_prod = products_mapping[product_ID]

            for word in review:
                word = word-1 
                words.append(word)
                len_reviews.append(length)
                user_IDs.append(idx_user)
                product_IDs.append(idx_prod)  

            ratings.append(rating)
            
            # word index from 0 to 2000, originally from 1 to 2001 (due to padding)
            # --> [w-1 for w in review]
            reviews_input.append([idx_user, idx_prod, [w-1 for w in review]]) 

    print(len(user_IDs), len(set(user_IDs)))
    print(len(product_IDs), len(set(product_IDs)))
    print(len(users_mapping), len(products_mapping))
    print(len(ratings))
    print(len(len_reviews))
    print(len(reviews_input))
    print(len(reviews_test))
    print()
    print('Saving files..\n')
    
    if not os.path.exists(dir_output):
        print('\nCreate new output directory: ', dir_output)
        os.makedirs(dir_output)

    with open(dir_output + 'user_IDs.pkl', "wb") as outfile:
        pickle.dump(user_IDs, outfile)
        outfile.close()

    with open(dir_output + 'product_IDs.pkl', "wb") as outfile:
        pickle.dump(product_IDs, outfile)
        outfile.close()
        
    with open(dir_output + 'users_map.pkl', "wb") as outfile:
        pickle.dump(users_mapping, outfile)
        outfile.close()

    with open(dir_output + 'products_map.pkl', "wb") as outfile:
        pickle.dump(products_mapping, outfile)
        outfile.close()

    with open(dir_output + 'ratings.pkl', "wb") as outfile:
        pickle.dump(ratings, outfile)
        outfile.close()

    with open(dir_output + 'words.pkl', "wb") as outfile:
        pickle.dump(words, outfile)
        outfile.close()

    with open(dir_output + 'lengths.pkl', "wb") as outfile:
        pickle.dump(len_reviews, outfile)
        outfile.close()
        
    with open(dir_output + 'reviews_input.pkl', "wb") as outfile:
        pickle.dump(reviews_input, outfile)
        outfile.close()
        
    with open(dir_output + 'users_test.pkl', 'wb') as outfile:
        pickle.dump(reviews_test, outfile)
        outfile.close()
        
def create_train_test_split(reviews_input, dir_output, threshold = 10, test_percentage = 0.05):
    random.shuffle(reviews_input)
    
    user_count = Counter([r[0] for r in reviews_input])
    most_common_users = [i for i,j in user_count.most_common() if j >= threshold]
    # reviews_common collects the most common users (i.e. with at least 10 reviews)
    reviews_common = []
    for r in reviews_input:  
        user_id = r[0]
        if user_id in most_common_users:
            reviews_common.append(r)

    # train and test sets for the EM part are created from the reviews_common list
    # train and test sets for the rating prediction part are created in a later step
    test_size = int(np.ceil(len(reviews_input)*test_percentage))
    test_set = random.sample(reviews_common, test_size)
    train_set = [r for r in reviews_input if r not in test_set]
    print(len(reviews_input), len(train_set), len(test_set))
    
    if not os.path.exists(dir_output):
        print('\nCreate new output directory: ', dir_output)
        os.makedirs(dir_output)
        
    out_filename = dir_output + 'reviews_train.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(train_set, outfile)
        outfile.close()

    out_filename = dir_output + 'reviews_test.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(test_set, outfile)
        outfile.close()
        
def create_additional_data(reviews, dir_output):
    # this function creates and saves additional data that will be used during the EM training
    # to speed up the computational part.
    user_words = defaultdict(list)
    product_words = defaultdict(list)
    B_w = defaultdict(list)
    W_up = {}
    W_pu = {}    
    
    for line in tqdm(reviews):
        user_ID = line[0]
        product_ID = line[1]
        review = line[2]
        tuple_IDs = (user_ID, product_ID)
        
        if W_up.get(user_ID) == None:
            W_up[user_ID] = defaultdict(list)

        if W_pu.get(product_ID) == None:
            W_pu[product_ID] = defaultdict(list)

        for word in review:
            W_up[user_ID][product_ID].append(word)
            W_pu[product_ID][user_ID].append(word)
            
            B_w[word].append(tuple_IDs)
            user_words[user_ID].append(word)
            product_words[product_ID].append(word)
            
            
    with open(dir_output + 'words_up.pkl', "wb") as outfile:
        pickle.dump(W_up, outfile)
        outfile.close()

    with open(dir_output + 'words_pu.pkl', "wb") as outfile:
        pickle.dump(W_pu, outfile)
        outfile.close()

    with open(dir_output + 'user_words.pkl', "wb") as outfile:
        pickle.dump(user_words, outfile)
        outfile.close()
        
    with open(dir_output + 'product_words.pkl', "wb") as outfile:
        pickle.dump(product_words, outfile)
        outfile.close()
        
    with open(dir_output + 'b_words.pkl', "wb") as outfile:
        pickle.dump(B_w, outfile)
        outfile.close()


categories = utils.CATEGORIES
dir_preprocessed_data = utils.DIR_PREPROCESSED_DATA
dir_training_data = utils.DIR_TRAIN_TEST_DATA
dir_results = utils.DIR_RESULTS
for category in categories:
    dir_preprocessed_data_c = dir_preprocessed_data + category + '/'
    dir_training_data_c = dir_training_data + category + '/'

    indexed_kw_reviews = pickle.load(open(dir_preprocessed_data_c + 'reviews_keywordslist.pkl', 'rb'))  

    print('Creating training data..\n')
    create_training_data(indexed_kw_reviews, dir_preprocessed_data_c)

    print('Train test splitting..\n')
    reviews_input = pickle.load(open(dir_preprocessed_data_c + 'reviews_input.pkl', "rb")) # {(idx_u,idx_p): [review]}            
    create_train_test_split(reviews_input, dir_training_data_c)

    print('Creating additional training data..\n')
    reviews_train = pickle.load(open(dir_training_data_c + 'reviews_train.pkl', "rb")) # {(idx_u,idx_p): [review]}
    create_additional_data(reviews_train, dir_training_data_c)

