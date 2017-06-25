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
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist

#
# Create input sequences
#

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#
# Create models
#

model = Sequential()
# Generator portion of model
model.add(Dense(200, name='Full_G_Dense_1',input_dim=x_train.shape[1]*x_train.shape[2]))
model.add(Dense(x_train.shape[1]*x_train.shape[2], name='Full_G_Dense_2'))
# Discriminator portion
model.add(Dense(200, name='Full_D_Dense_1'))
model.add(Dense(200, name='Full_D_Dense_2'))
model.add(Dense(1, name='Full_D_Dense_3', activation='sigmoid'))
###model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

#
# Just the G portion
#
modelG = Sequential(name='modelG')
modelG.add(Dense(200, name='G_Dense_1',input_dim=x_train.shape[1]*x_train.shape[2]))
modelG.add(Dense(x_train.shape[1]*x_train.shape[2], name='G_Dense_2'))
modelG.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
print(modelG.summary())

#
# Just the D portion
#
modelD = Sequential()
modelD.add(Dense(200, name='D_Dense_1',input_dim=x_train.shape[1]*x_train.shape[2]))
modelD.add(Dense(200, name='D_Dense_2'))
modelD.add(Dense(1, name='D_Dense_3', activation='sigmoid'))
###model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
modelD.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
print(modelD.summary())

#
# Train
# 
##print('Training model...')
###model.fit(X_train, y_train, epochs=1)

def generate_from_alternating_sources():
    sourceGenModel=True
    mnist_index=0
    while 1:
        # create Numpy arrays of input data
        # if sourceGenModel==True
        # Create from generative model portion
        # else create from mnist_data
        
        if sourceGenModel:
            x=modelG.predict(np.random.uniform(size=(1,784)), batch_size=1)
            print('modelG.name = ', modelG.name)
            x=np.expand_dims(np.random.uniform(size=784),axis=0)
            print('sourceGenModel = ', sourceGenModel)
            print('x.shape = ', x.shape)
            y=np.zeros([1]) # 0 means G model data
            sourceGenModel=False # toggle
            
        else:
            
            x=np.ndarray.flatten(x_train[mnist_index])
            x=np.expand_dims(x,axis=0)
            mnist_index+=1
            mnist_index=mnist_index%x_train.shape[0] # Loop to beginning if at end
            print('sourceGenModel = ', sourceGenModel)
            print('x.shape = ', x.shape)
            print('mnist_index = ', mnist_index)
            y=np.ones([1]) # 1 means real data
            sourceGenModel=True # toggle
        
        print('y = ', y)
        yield (x, y)

#modelD.fit_generator(generate_from_alternating_sources(),
#        steps_per_epoch=1, epochs=1, max_q_size=1)

for z in generate_from_alternating_sources():
    print('calling generator:')
    print('return val = ', z)
    break

#
# output predictions
#
##print(model.get_weights()[0][0])
##predictions = model.predict(X_train)

#
# Other diagnostics
#
###print()
###model.evaluate(X_train, y_train)

#
# Ref values for one epoch:   [0.24794224, 0.66666669]
#

#
# End of script
#
