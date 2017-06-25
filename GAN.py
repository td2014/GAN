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
from keras.layers import Dense, Input, concatenate
from keras.models import Sequential, Model
from keras.datasets import mnist

import matplotlib.pyplot as plt

#
# Create input sequences
#

(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

#
# Create models
#
common_input_dim=x_train_mnist.shape[1]*x_train_mnist.shape[2]
###model = Sequential(name='Full_model')
# Generator portion of model
main_input = Input(shape=(common_input_dim,), name='main_input')
x=Dense(128, name='Full_G_Dense_1',input_dim=common_input_dim, use_bias=False)(main_input)
G_out=Dense(common_input_dim, name='Full_G_Dense_2', use_bias=False)(x)
# Add input for real data
auxiliary_input = Input(shape=(common_input_dim,), name='aux_input')
x = concatenate([G_out, auxiliary_input], name='Full_Concatenate')
# Discriminator portion
x = Dense(64, name='Full_D_Dense_1', use_bias=False)(x)
###x = Dense(8, name='Full_D_Dense_2', use_bias=False)(x)
main_output_obj = Dense(1, name='main_output', activation='sigmoid', use_bias=False)(x)
#
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output_obj])
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
print('Baseline:')
print(model.summary())

#
# Just the G portion
#
###modelG = Sequential(name='modelG')
###modelG.add(Dense(200, name='G_Dense_1',input_dim=common_input_dim))
###modelG.add(Dense(common_input_dim, name='G_Dense_2'))
###modelG.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
###print(modelG.summary())

#
# Just the D portion
#
###modelD = Sequential(name='modelD')
###modelD.add(Dense(200, name='D_Dense_1',input_dim=common_input_dim))
###modelD.add(Dense(200, name='D_Dense_2'))
###modelD.add(Dense(1, name='D_Dense_3', activation='sigmoid'))
###model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
###modelD.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
###print(modelD.summary())

#
# Train
# 
##print('Training model...')
###model.fit(X_train, y_train, epochs=1)

###def generate_from_alternating_sources():
###    sourceGenModel=True
###    mnist_index=0
###    while 1:
###        # create Numpy arrays of input data
###        # if sourceGenModel==True
###        # Create from generative model portion
###        # else create from mnist_data
        
###        if sourceGenModel:
###            x=modelG.predict(np.random.uniform(size=(1,common_input_dim)), batch_size=1)
###            print('modelG.name = ', modelG.name)
###          #  x=np.expand_dims(np.random.uniform(size=common_input_dim),axis=0)
###            print('sourceGenModel = ', sourceGenModel)
###            print('x.shape = ', x.shape)
###            y=np.zeros([1]) # 0 means G model data
###            sourceGenModel=False # toggle
            
###        else:
            
###            x=np.ndarray.flatten(x_train[mnist_index])
###            x=np.expand_dims(x,axis=0)
###            mnist_index+=1
###            mnist_index=mnist_index%x_train.shape[0] # Loop to beginning if at end
###            print('sourceGenModel = ', sourceGenModel)
###            print('x.shape = ', x.shape)
###            print('mnist_index = ', mnist_index)
###            # Need to set output of G stage output to MNIST and Freeze G layers
            
###            #
###            y=np.ones([1]) # 1 means real data
###            sourceGenModel=True # toggle
        
###        print('y = ', y)
###        yield (x, y)
   

#
# Freeze G
#      

layer = model.get_layer(name='Full_G_Dense_1')
layer.trainable = False
layer = model.get_layer(name='Full_G_Dense_2')
layer.trainable = False
# in the model below, the weights of `layer` will not be updated during training
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
print('After freeze:')
model.summary()

# Train the discriminator
for k in range(10):   
        
    x_train=np.random.uniform(size=(1,common_input_dim))
    x_aux=np.zeros([1,common_input_dim])
    y_train=np.zeros([1]) # 0 means G model data
    
    model.fit({'main_input': x_train, 'aux_input': x_aux}, 
          {'main_output': y_train}, 
          epochs=1, batch_size=1)
    
#
#   Set output of G/input of D to mnist data  
#

    x_train=np.zeros([1,common_input_dim])
    mnist_index=k%x_train_mnist.shape[0]
    x_aux=np.ndarray.flatten(x_train_mnist[mnist_index])
    x_aux=np.expand_dims(x_aux,axis=0)
    #
    y_train=np.ones([1]) # 1 means G model data
    model.fit({'main_input': x_train, 'aux_input': x_aux}, 
          {'main_output': y_train}, 
          epochs=1, batch_size=1)

#
# output predictions
#

# Real data
x_train=np.zeros([1,common_input_dim])
mnist_index=998
x_aux=np.ndarray.flatten(x_train_mnist[mnist_index])
x_aux=np.expand_dims(x_aux,axis=0)
prediction = model.predict({'main_input': x_train, 'aux_input': x_aux})

# Generated Data
###x_train=np.random.uniform(size=(1,common_input_dim))
###x_train=np.random.uniform(size=(1,common_input_dim))
###x_aux=np.zeros([1,common_input_dim])
###prediction2 = model.predict({'main_input': x_train, 'aux_input': x_aux})

#
# Other diagnostics
#
###print()
###model.evaluate(X_train, y_train)

#
# End of script
#
