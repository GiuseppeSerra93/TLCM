# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Activation, Flatten, Concatenate, Dense, Dropout

import argparse
import utils


parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=utils.NUM_EPOCHS, help='Number of epochs')
parser.add_argument("--bs", type=int, default=utils.BATCH_SIZE, help='Batch size')
parser.add_argument("--lr", type=float, default=utils.LEARNING_RATE, help='Learning rate (default value: 2e-6)')
parser.add_argument("--gpu", type=int, help='GPU device ID')
parser.add_argument("--K", type=int, default=utils.K, help='Number of user latent classes')
parser.add_argument("--L", type=int, default=utils.L, help='Number of product latent classes')
parser.add_argument("--runs", type=int, default=utils.NUM_RUNS, help='Number of runs for each category')
args = parser.parse_args()


gpu = str(args.gpu)
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.bs
LR = args.lr
NUM_RUNS = args.runs
K = args.K
L = args.L

dir_preprocessed_data = utils.DIR_PREPROCESSED_DATA
dir_training_data = utils.DIR_TRAIN_TEST_DATA
dir_results = utils.DIR_RESULTS

categories = utils.CATEGORIES

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Hint: so the IDs match nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def build_model(lr):
    input_user = Input(shape=(int(np.sqrt(K)),int(np.sqrt(K)),1), name='input_user')
    input_product = Input(shape=(int(np.sqrt(L)),int(np.sqrt(L)),1), name='input_product')
    
    x_user = Conv2D(32, (3,3), padding='valid')(input_user)
    x_user = BatchNormalization(axis=-1)(x_user)
    x_user = Activation('relu')(x_user)
    x_user = Conv2D(32, (2,2), padding='valid')(x_user)
    x_user = BatchNormalization(axis=-1)(x_user)
    x_user = Activation('relu')(x_user)
    x_user = MaxPooling2D(pool_size=(1,1))(x_user)
    x_user = Flatten()(x_user)
    
    x_product = Conv2D(32, (3,3), padding='valid')(input_product)
    x_product = BatchNormalization(axis=-1)(x_product)
    x_product = Activation('relu')(x_product)
    x_product = Conv2D(32, (2,2), padding='valid')(x_product)
    x_product = BatchNormalization(axis=-1)(x_product)
    x_product = Activation('relu')(x_product)
    x_product = MaxPooling2D(pool_size=(1,1))(x_product)
    x_product = Flatten()(x_product)
    
    concat_x = Concatenate()([x_user, x_product])
    
    x_dense = Dense(128)(concat_x)
    x_dense = BatchNormalization()(x_dense)
    x_dense = Activation('relu')(x_dense)
    x_dense = Dropout(.2)(x_dense)
    
    x_dense = Dense(64)(x_dense)
    x_dense = BatchNormalization()(x_dense)
    x_dense = Activation('relu')(x_dense)
    x_dense = Dropout(.2)(x_dense)
    
    x_dense = Dense(32)(x_dense)
    x_dense = BatchNormalization()(x_dense)
    x_dense = Activation('relu')(x_dense)
    x_dense = Dropout(.3)(x_dense)
    
    x_dense = Dense(16)(x_dense)
    x_dense = BatchNormalization()(x_dense)
    x_dense = Activation('relu')(x_dense)
    x_dense = Dropout(.3)(x_dense)
    output = Dense(1)(x_dense)
    
    model = Model(inputs=[input_user,input_product], outputs=output)
    mse = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=mse)
    return model


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
    verbose=1,
    patience=20,
    mode='min',
    restore_best_weights=True)

for category in categories:
    tf.random.set_seed(93)
    dir_training_data_c = dir_training_data + category + '/'
    dir_rating_data = dir_training_data_c + 'rating_data_{0}_{1}/'.format(K, L)
    dir_results_c = dir_results + category + '/results_CNN_{0}_{1}/'.format(K, L)
    if not os.path.exists(dir_results_c):  # create directory if it does not exist
        print('\tCreate new output directory:', dir_results_c)
        os.makedirs(dir_results_c)

    y_train = pickle.load(open(dir_rating_data + 'y_train.pkl', 'rb'))
    y_test = pickle.load(open(dir_rating_data + 'y_test.pkl', 'rb'))
    y_val = pickle.load(open(dir_rating_data + 'y_val.pkl', 'rb'))       
    train_u = pickle.load(open(dir_rating_data + 'train_u.pkl', 'rb'))
    test_u = pickle.load(open(dir_rating_data + 'test_u.pkl', 'rb'))
    val_u = pickle.load(open(dir_rating_data + 'val_u.pkl', 'rb'))
    train_p = pickle.load(open(dir_rating_data + 'train_p.pkl', 'rb'))
    test_p = pickle.load(open(dir_rating_data + 'test_p.pkl', 'rb'))
    val_p = pickle.load(open(dir_rating_data + 'val_p.pkl', 'rb'))


    mse_test = []
    for i in range(NUM_RUNS):
        print('Run {0} - Category: {1}'.format(i+1, category))
        tf.keras.backend.clear_session()

        model = build_model(lr=LR)
        print(model.summary())

        trained = model.fit(x=[train_u, train_p], y=y_train,
                            batch_size=BATCH_SIZE,
                            epochs=NUM_EPOCHS,
                            validation_data=([val_u,val_p],y_val),
                            callbacks=[early_stopping],
                            verbose=1)

        mse_run = model.evaluate(x=[test_u, test_p], y=y_test,
                                  batch_size=BATCH_SIZE)
        
        mse_test.append(mse_run)
        
    with open(dir_results_c + 'mse_test.pkl', 'wb') as outfile:
        pickle.dump(mse_test, outfile)
        outfile.close()
        
    print(mse_test)