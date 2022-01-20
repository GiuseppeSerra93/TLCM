#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pickle
import os
from tqdm import tqdm
from collections import Counter, defaultdict
from minisom import MiniSom
import argparse
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=utils.K, help='Number of user latent classes')
parser.add_argument("--L", type=int, default=utils.L, help='Number of product latent classes')
args = parser.parse_args()

K = args.K
L = args.L
V = utils.V
sigma_users = utils.SIGMA_USERS
sigma_prods = utils.SIGMA_PRODUCTS
a_users = utils.A_USERS
b_users = utils.B_USERS
a_prods = utils.A_PRODUCTS
b_prods = utils.B_PRODUCTS
categories = utils.CATEGORIES

dir_preprocessed_data = utils.DIR_PREPROCESSED_DATA
dir_training_data = utils.DIR_TRAIN_TEST_DATA
dir_results = utils.DIR_RESULTS

for category in categories:
    dir_preprocessed_data_c = dir_preprocessed_data + category + '/'
    dir_training_data_c = dir_training_data + category + '/'
    dir_results_c = dir_results + category + '/results_EM_{0}_{1}/'.format(K, L)
    
    if not os.path.exists(dir_results_c):  # create directory if it does not exist
        print('\tCreate new output directory:', dir_results_c)
        os.makedirs(dir_results_c)


    users_map = pickle.load(open(dir_preprocessed_data_c + 'users_map.pkl', 'rb'))  # {user_ID: idx_u}
    products_map = pickle.load(open(dir_preprocessed_data_c + 'products_map.pkl', 'rb')) # {prod_ID: idx_p}
    inv_users_map = {v:k for k,v in users_map.items()} # {idx_u: user_ID}
    inv_products_map = {v:k for k,v in products_map.items()} # {idx_u: user_ID}
    
    reviews_train = pickle.load(open(dir_training_data_c + 'reviews_train.pkl', "rb")) # {(idx_u,idx_p): [review]}
    reviews_test = pickle.load(open(dir_training_data_c + 'reviews_test.pkl', "rb")) # {(idx_u,idx_p): [review]}
    user_words = pickle.load(open(dir_training_data_c + 'user_words.pkl', "rb"))  # {idx_u: [words_u]}
    prod_words = pickle.load(open(dir_training_data_c + 'product_words.pkl', "rb")) # {idx_p: [words_p]}
    w_up = pickle.load(open(dir_training_data_c + 'words_up.pkl', "rb"))  # {idx_u: {idx_p : [words]}
    w_pu = pickle.load(open(dir_training_data_c + 'words_pu.pkl', "rb"))  # {idx_p: {idx_u : [words]}
    b_w = pickle.load(open(dir_training_data_c + 'b_words.pkl', "rb"))    # {idx_word: [(idx_u, idx_p)]}

    def dist(xy, xy2):
        x1, y1 = xy
        x2, y2 = xy2
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return dist

    def neighborhood_function(z1, z, sigma=0.5):
        output = []
        num = sum([np.exp(-dist(z1,zz)/(2*sigma)) for zz in z])
        for z2 in z:
            den = np.exp(-dist(z1,z2)/(2*sigma))
            output.append(den/num)
        return output

    grid_user = [(float(i), float(j)) for i in range(int(np.sqrt(K))) for j in range(int(np.sqrt(K)))]
    grid_prod = [(float(i), float(j)) for i in range(int(np.sqrt(L))) for j in range(int(np.sqrt(L)))]
    map_grid_user = {grid_user[i]:i for i in range(len(grid_user))}
    map_grid_prod = {grid_prod[i]:i for i in range(len(grid_prod))}
    alpha_user = np.array([neighborhood_function(point, grid_user, sigma=sigma_users) for point in grid_user])
    alpha_prod = np.array([neighborhood_function(point, grid_prod, sigma=sigma_prods) for point in grid_prod])
    
    # ==============================================================================
    #     SOM INITIALIZATION FOR USERS
    # ==============================================================================
    print('SOM Initialization for users...\n')
    input_som_users_tem = []
    
    for user in sorted(user_words.keys()):
        vector_user = np.zeros(V)
        for k,v in Counter(user_words[user]).items():
            vector_user[k] = v
        input_som_users_tem.append(vector_user)

    input_som_users = np.array(input_som_users_tem) 
    print(input_som_users.shape)
    print(Counter(user_words[2])[1189] == input_som_users[2][1189])
    print()

    som_users = MiniSom(int(np.sqrt(K)), int(np.sqrt(K)), V, sigma=0.5, neighborhood_function='gaussian',
                        learning_rate=0.1)
    som_users.pca_weights_init(input_som_users)
    som_users.train(input_som_users, 500)

    with open(dir_results_c + 'som_users.p', 'wb') as outfile:
        pickle.dump(som_users, outfile)
        outfile.close()

    results_user = []
    for i in range(input_som_users.shape[0]):
        results_user.append(som_users.winner(input_som_users[i]))

    with open(dir_results_c + 'som_results_user.pkl', 'wb') as outfile:
        pickle.dump(results_user, outfile)
        outfile.close()

    p_zu_u = np.full((len(users_map.keys()),K), b_users)
    for i in range(len(results_user)):
        idx = map_grid_user.get(results_user[i])
        p_zu_u[i][idx] = a_users

    print(p_zu_u.shape)
    print(sum(np.sum(p_zu_u, axis=-1)))

    # ==============================================================================
    #     SOM INITIALIZATION FOR PRODUCTS
    # ==============================================================================
    print('SOM Initialization for products...\n')
    input_som_prods_tem = []
    
    for prod in sorted(prod_words.keys()):
        vector_prod = np.zeros(V)
        for k,v in Counter(prod_words[prod]).items():
            vector_prod[k] = v
        input_som_prods_tem.append(vector_prod)

    input_som_prods = np.array(input_som_prods_tem) 
    print(input_som_prods.shape)
    print(Counter(prod_words[10])[1353] == input_som_prods[10][1353])
    print()

    som_prods = MiniSom(int(np.sqrt(L)), int(np.sqrt(L)), V, sigma=0.5, neighborhood_function='gaussian',
                        learning_rate=0.2)
    som_prods.pca_weights_init(input_som_prods)
    som_prods.train(input_som_prods, 5000)

    with open(dir_results_c + 'som_prods.p', 'wb') as outfile:
        pickle.dump(som_prods, outfile)
        outfile.close()

    results_prod = []
    for j in range(input_som_prods.shape[0]):
        results_prod.append(som_prods.winner(input_som_prods[j]))

    with open(dir_results_c + 'som_results_prod.pkl', 'wb') as outfile:
        pickle.dump(results_prod, outfile)
        outfile.close()

    p_zp_p = np.full((len(products_map.keys()),L), b_prods)
    for i in range(len(results_prod)):
        idx = map_grid_prod.get(results_prod[i])
        p_zp_p[i][idx] = a_prods

    print(p_zp_p.shape)
    print(sum(np.sum(p_zp_p, axis=-1)))

    # ==============================================================================
    #     COMPUTE EMPIRICAL DISTRIBUTION P(w|y_u^{k'},y_p^{\ell'})
    # ==============================================================================
    # random sampling from p(z_u|user)
    random_samples_zu = []
    sampled_zu = defaultdict(list)     # {class_idx : [users belonging to the class]}
    yu_assignment = {}                 # (user : y_uk')

    for user in sorted(inv_users_map.keys()):
        # for each user, latent class assignment from p(z_u|u) (multinomial)
        random_sample_zu = np.random.multinomial(1, p_zu_u[user])
        idx_grid_zu = np.where(random_sample_zu==1)[0][0]
        sampled_point_zu = grid_user[idx_grid_zu]

        # channel noise via neighborhood function
        random_sample_zu_final = np.random.multinomial(1, alpha_user[idx_grid_zu])
        idx_grid_zu_final = np.where(random_sample_zu_final==1)[0][0]
        random_samples_zu.append(idx_grid_zu_final)

        sampled_zu[idx_grid_zu_final].append(user)
        yu_assignment[user] = idx_grid_zu_final


    # random sampling for p(z_p|prod)
    random_samples_zp = []
    sampled_zp = defaultdict(list)     # {class_idx : [products belonging to the class]}
    yp_assignment = {}                 # (prod : y_pl')

    for prod in sorted(inv_products_map.keys()):  
        # for each prod, latent class assignment from p(z_p|p) (multinomial)
        random_sample_zp = np.random.multinomial(1, p_zp_p[prod])
        idx_grid_zp = np.where(random_sample_zp==1)[0][0]
        sampled_point_zp = grid_prod[idx_grid_zp]

        # channel noise via neighborhood function
        random_sample_zp_final = np.random.multinomial(1, alpha_prod[idx_grid_zp])
        idx_grid_zp_final = np.where(random_sample_zp_final==1)[0][0]
        random_samples_zp.append(idx_grid_zp_final)

        sampled_zp[idx_grid_zp_final].append(prod)
        yp_assignment[prod] = idx_grid_zp_final

    # word counting for each latent class combination (k,l)
    # i.e. count how many times the words appear in the dataset for each latent class combination (k,l)
    count_w_yuyp = {}          # {(k', l') = [word count]}

    for review in reviews_train:
        user_id = review[0]
        prod_id = review[1]
        words = review[2]

        # get the latent classes k' and l' for each user and product respectively
        y_uk = yu_assignment[user_id]
        y_pl = yp_assignment[prod_id]

        if count_w_yuyp.get(y_uk) == None:
            count_w_yuyp[y_uk] = defaultdict(list)

        # save the words for the given (k',l') combination
        count_w_yuyp[y_uk][y_pl].extend(words)


    # empirical distribution p(w|y_uk', y_pl')
    p_w_yuyp = []
    m = 1. # parameter for Laplacian correction

    for k in sorted(count_w_yuyp.keys()):
        for l in sorted(count_w_yuyp[k].keys()):
            count = Counter(count_w_yuyp[k][l])
            N_w = len(count_w_yuyp[k][l])
            p_w_kl = np.zeros(V)

            for w in range(V):
                if count.get(w) == None:
                    den = m
                else:
                    den = count[w] + m

                num = N_w + (V*m)
                p_w_kl[w] = den/num

            p_w_yuyp.append(p_w_kl)

    p_w_yuyp = np.array(p_w_yuyp)


    # ==============================================================================
    #     E-STEP
    # ==============================================================================
    def e_step(reviews, p_zu_u, p_zp_p, p_w_yuyp, alpha_u, alpha_p, K, L, V):
        alpha_user = np.repeat(alpha_u, L, axis=1).reshape(K,K*L)
        alpha_prod = alpha_p.reshape(L,L,1)
        alpha_prod = np.tile(alpha_prod,(1,K)).reshape(L,K*L)

        # p(y_uk'|z_uk)*p(y_pl'|z_pl) for each combination (k',l')
        alpha_mul = np.array([np.multiply(alpha_user[k],alpha_prod[l]) for k in range(K) for l in range(L)])

        p_w_yuyp_T = p_w_yuyp.T
        output_dict_zuzp = {i: defaultdict(dict) for i in range(V)}  # {word_id:{user_id: {prod_id: P(z_uk,z_pl|w,u,p)}}}
        output_dict_yuyp = {i: defaultdict(dict) for i in range(V)}  # {word_id:{user_id: {prod_id: P(y_uk',y_pl'|w,u,p)}}}

        for review in tqdm(reviews):
            user_id = review[0]
            prod_id = review[1]
            words = review[2]

            p_zu_user = p_zu_u[user_id]  # p(z_uk | user_id) = (K,)
            p_zp_prod = p_zp_p[prod_id]  # p(z_pl | prod_id) = (L,)

            # sum_k p(y_uk'|z_uk)*p(z_uk|user_id) = (K,)
            p_yu_zu = np.sum(np.multiply(p_zu_user.reshape(K,1), alpha_u), axis=0)

            # sum_l p(y_pl'|z_pl)*p(z_pl|prod_id) = (L,)
            p_yp_zp = np.sum(np.multiply(p_zp_prod.reshape(L,1), alpha_p), axis=0)

            # (p_yu_zu * p_yp_zp) for each combination (k', l') = (K*L,)
            p_yuyp_zuzp = np.array([p_yu_zu[k]*p_yp_zp[l] for k in range(K) for l in range(L)])

            # p(z_uk|user)p(z_pl|prod) for each combination (k,l) = (K*L,)
            p_zu_zp = np.array([p_zu_user[k]*p_zp_prod[l] for k in range(K) for l in range(L)])

            for word in words:
                # ==============================================================================
                #     P(z_uk,z_pl|w,u,p)
                # ==============================================================================
                p_word_yuyp = p_w_yuyp_T[word].reshape(K*L, 1)   # p(word_id | y_uk', y_pl') = (K*L,) (is not a distribution)
                alpha_mul_tem = np.multiply(p_word_yuyp, alpha_mul)
                sum_mul_tem = np.sum(alpha_mul_tem, axis=1)   # sum over k', sum over l'
                num = np.multiply(p_zu_zp, sum_mul_tem)
                den = np.sum(num)
                res = num/den
                output_dict_zuzp[word][user_id][prod_id] = res

                # ==============================================================================
                #     P(y_uk',y_pl'|w,u,p)
                # ==============================================================================
                num = np.multiply(p_w_yuyp_T[word], p_yuyp_zuzp)
                den = np.sum(num)
                res = num/den
                output_dict_yuyp[word][user_id][prod_id] = res

        return output_dict_zuzp, output_dict_yuyp


    # ==============================================================================
    #     M-STEP
    # ==============================================================================
    def m_step(output_dict_zuzp, output_dict_yuyp, b_w, w_up, w_pu, K, L, V):
        sum_l = [np.arange(L)+L*i for i in range(K)]
        sum_k = [np.arange(0, K*L, L)+1*i for i in range(L)]

        # ==============================================================================
        #     P(w|y_uk',y_pl')
        # ==============================================================================
        den = np.zeros(K*L)
        for word_id in b_w.keys():
            for user_id, prod_id in b_w[word_id]:
                den += output_dict_yuyp[word_id][user_id][prod_id]

        tem_w_yuyp = np.zeros((V, K*L))
        for word_id in b_w.keys(): 
            num = np.zeros(K*L)
            for user_id,prod_id in b_w[word_id]:
                num += output_dict_yuyp[word_id][user_id][prod_id]

            res = num/den
            tem_w_yuyp[word_id] = res  # (#words, K*L)

        output_w_yuyp = np.array(tem_w_yuyp).T  # (K*L, #words)

        # ==============================================================================
        #     P(z_uk|u)
        # ==============================================================================
        tem_zu_u = []
        for user_id in sorted(w_up.keys()):
            num_tem = np.zeros(K*L) 
            den = 0        
            for prod_id in w_up[user_id].keys():
                for word_id in w_up[user_id][prod_id]:
                    num_tem += output_dict_zuzp[word_id][user_id][prod_id]
                    den += 1

            num = np.array([sum(num_tem[sum_l[i]]) for i in range(K)]) # sum over l = (K,)
            res = num/den
            tem_zu_u.append(res)

        output_zu_u = np.array(tem_zu_u)


        # ==============================================================================
        #     P(z_pl|p)
        # ==============================================================================
        tem_zp_p = []
        for prod_id in sorted(w_pu.keys()):
            num_tem = np.zeros(K*L)
            den = 0
            for user_id in w_pu[prod_id].keys():
                for word_id in w_pu[prod_id][user_id]:
                    num_tem += output_dict_zuzp[word_id][user_id][prod_id]
                    den += 1

            num = np.array([sum(num_tem[sum_k[i]]) for i in range(L)]) # sum over k = (L,)
            res = num/den
            tem_zp_p.append(res)

        output_zp_p = np.array(tem_zp_p)

        return output_w_yuyp, output_zu_u, output_zp_p

    # ==============================================================================
    #     COMPUTE NEG-LOG-LIKELIHOOD
    # ==============================================================================
    def compute_nll(reviews, p_w_yuyp, p_zu_u, p_zp_p, alpha_u, alpha_p, K, L):
        ll_data = 0.
        p_w_yuyp_T = p_w_yuyp.T

        for review in tqdm(reviews):
            user_id = review[0]
            prod_id = review[1]
            words = review[2]

            p_zu_user = p_zu_u[user_id] # (K,)
            p_zp_prod = p_zp_p[prod_id] # (L,)       

            # sum_k p(y_uk'|z_uk)*p(z_uk|user_id) = (K,)
            p_yu_zu = np.sum(np.multiply(p_zu_user.reshape(K,1), alpha_u), axis=0)

            # sum_l p(y_pl'|z_pl)*p(z_pl|prod_id) = (L,)
            p_yp_zp = np.sum(np.multiply(p_zp_prod.reshape(L,1), alpha_p), axis=0)

            # (p_yu_zu * p_yp_zp) for each combination (k', l') = (K*L,)
            p_yuyp_zuzp = np.array([p_yu_zu[k]*p_yp_zp[l] for k in range(K) for l in range(L)])

            ll_r = np.sum([np.sum(np.multiply(p_w_yuyp_T[word], p_yuyp_zuzp)) for word in words])
            ll_r_tem = -np.log(ll_r+1e-8)
            nll_r = np.sum(ll_r_tem)
            ll_data += nll_r

        nll_data = ll_data/len(reviews)

        return(nll_data)

    # ==============================================================================
    #     EM-TRAINING
    # ==============================================================================
    nll_train = []
    nll_test = []
    tol = utils.TOLERANCE
    max_iter = utils.MAX_ITER
    max_patience = utils.MAX_PATIENCE
    patience = 0
    nll_old = np.infty
    i = 0
    while i < max_iter and patience <= max_patience:
        print(category + ' - Iteration ' + str(i+1))
        output_dict_zuzp, output_dict_yuyp = e_step(reviews_train, p_zu_u, p_zp_p, p_w_yuyp, alpha_user, alpha_prod, K, L, V)
        p_w_yuyp, p_zu_u, p_zp_p = m_step(output_dict_zuzp, output_dict_yuyp, b_w, w_up, w_pu, K, L, V)
        print('Computing neg-loglikelihood...')
        nll_iter_tr = compute_nll(reviews_train, p_w_yuyp, p_zu_u, p_zp_p, alpha_user, alpha_prod, K, L)
        nll_iter_te = compute_nll(reviews_test, p_w_yuyp, p_zu_u, p_zp_p, alpha_user, alpha_prod, K, L)

        if np.abs(nll_iter_tr - nll_old) < tol:
            patience +=1
        else:
            patience = 0   

        nll_old = nll_iter_tr
        nll_train.append(nll_iter_tr)
        nll_test.append(nll_iter_te)
        print(category + ' - Step %d, NLL__train: %0.3f, NLL__test: %0.3f' % (i+1, nll_iter_tr, nll_iter_te))
        
        i+=1
        

    out_filename = dir_results_c + 'p_w_yuyp.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(p_w_yuyp, outfile)
        outfile.close()

    out_filename = dir_results_c + 'p_zu_u.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(p_zu_u, outfile)
        outfile.close()

    out_filename = dir_results_c + 'p_zp_p.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(p_zp_p, outfile)
        outfile.close()

    out_filename = dir_results_c + 'nll_train.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(nll_train, outfile)
        outfile.close()
        
    out_filename = dir_results_c + 'nll_test.pkl'
    with open(out_filename, 'wb') as outfile:
        pickle.dump(nll_test, outfile)
        outfile.close()
