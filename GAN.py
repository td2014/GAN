#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:12:49 2017

@author: anthonydaniell
"""
# Start: Set up environment for reproduction of results
import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
#single thread
session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# End:  Set up environment for reproduction of results

#
from keras.layers import Dense, Input, concatenate, add, Dropout
from keras.models import Sequential, Model
from keras.datasets import mnist

import matplotlib.pyplot as plt

#
# Create input sequences
#

(x_train_mnist_orig, y_train_mnist_orig), (x_test_mnist_orig, y_test_mnist_orig) = mnist.load_data()
x_train_mnist = (x_train_mnist_orig-128.0)/255.0
x_test_mnist = (x_test_mnist_orig-128.0)/255.0

#
# Create models
#
common_input_dim=x_train_mnist.shape[1]*x_train_mnist.shape[2]
###model = Sequential(name='Full_model')
# Generator portion of model
main_input = Input(shape=(common_input_dim,), name='main_input')
x=Dense(128, name='Full_G_Dense_1',input_dim=common_input_dim, use_bias=False,)(main_input)
G_out=Dense(common_input_dim, name='Full_G_Dense_2', use_bias=False)(x)
# Add input for real data
auxiliary_input = Input(shape=(common_input_dim,), name='aux_input')
x = add([G_out, auxiliary_input], name='Full_Add')
# Discriminator portion (First layer is interface)
x = Dense(64, name='Full_D_Dense_1', use_bias=False, trainable=False, kernel_initializer='ones')(x)
x = Dense(64, name='Full_D_Dense_2', use_bias=True)(x)
x = Dropout(0.4, name='Full_D_Dropout_1')(x)
x = Dense(64, name='Full_D_Dense_3', use_bias=True)(x)
main_output_obj = Dense(1, name='main_output', activation='sigmoid')(x)
#
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output_obj])
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
print('Baseline:')
print(model.summary())

#
# Freeze G and first interface in D
#      

layer = model.get_layer(name='Full_G_Dense_1')
layer.trainable = False
layer = model.get_layer(name='Full_G_Dense_2')
layer.trainable = False
# in the model below, the weights of `layers` will not be updated during training
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
print('After freeze:')
model.summary()

#
# Train the discriminator
#
for eLoop in range(100):

    batch_size_loop=100
    # batch_size=2*batch_size_loop # 2x because we add generated and real each step
    x_train = []
    y_train = []
    x_aux=[]
    
    # Create a batch of data alternating between generated and real.
    for k in range(batch_size_loop):   
        
        # Create input for generator section.
        x_train_batch=np.random.uniform(low=-0.5,high=0.5,size=(common_input_dim))
        x_aux_batch=np.zeros([common_input_dim])
        y_train_batch=np.zeros([1]) # 0 means G model data
        
        # Update batch
        x_train.append(x_train_batch)
        y_train.append(y_train_batch)
        x_aux.append(x_aux_batch)
    
        # Now bring in data from mnist
        x_train_batch=np.zeros([common_input_dim])
        mnist_index=(k+eLoop*batch_size_loop)%x_train_mnist.shape[0]
        x_aux_batch=np.ndarray.flatten(x_train_mnist[mnist_index])
        y_train_batch=np.ones([1]) # 1 means real data
        
        # Update batch
        x_train.append(x_train_batch)
        y_train.append(y_train_batch)
        x_aux.append(x_aux_batch)
    
    
    # Convert list to arrays for input to model fitting
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_aux = np.asarray(x_aux)
    
    # Fit model to constructed batch
    model.fit({'main_input': x_train, 'aux_input': x_aux}, 
        {'main_output': y_train}, 
        epochs=10, batch_size=10)
    
#
# output predictions
#

prediction = model.predict({'main_input': x_train, 'aux_input': x_aux})

# Real data
###x_train=np.zeros([1,common_input_dim])
###mnist_index=5
###x_aux=np.ndarray.flatten(x_train_mnist[mnist_index])
###x_aux=np.expand_dims(x_aux,axis=0)
###prediction = model.predict({'main_input': x_train, 'aux_input': x_aux})

# Generated Data
###x_train=np.random.uniform(low=-0.5,high=0.5,size=(1,common_input_dim))
###x_aux=np.zeros([1,common_input_dim])
###prediction2 = model.predict({'main_input': x_train, 'aux_input': x_aux})

#
# Create a minimodel and look at output of concate layer
# I need to verify the inputs being seen by the discriminator.
#

#
# End of script
#
