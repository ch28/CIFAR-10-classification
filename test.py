#-----------------------------------------------------------------------------------+
# Author:                                                                           |
# Time Stamp: Oct 14, 2022                                                          |
# Affiliation: Beijing University of Posts and Telecommunications                   |
# Email:                                                                            |
#-----------------------------------------------------------------------------------+
#                             *** Open Source Code ***                              |
#-----------------------------------------------------------------------------------+
import numpy as np

import tensorflow as tf
from keras.layers import Input, Dense, Activation, Layer, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
import scipy.io as sio 
import numpy as np
import math
import time
import keras.backend as K

from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from keras.optimizers import SGD

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.reset_default_graph()

#40%
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
session = tf.Session(config=config)


# training params
epochs = 15
batch_size = 20
lr = 1e-3
loss_mode = 'mse' # 'mae' or 'mse' or 'cross_entropy'

(x_train1,y_train1), (x_test1,y_test1) = cifar10.load_data()
 
x_train1 = x_train1.reshape(x_train1.shape[0],-1)/255.0
x_test1 = x_test1.reshape(x_test1.shape[0],-1)/255.0

x_train = np.zeros((10000, x_train1.shape[1]))
x_test = np.zeros((2000, x_test1.shape[1]))

y_train = np.concatenate((np.ones((5000,1)), np.zeros((5000,1))), axis = 0)
y_test = np.concatenate((np.ones((1000,1)), np.zeros((1000,1))), axis = 0)

print(x_train.shape)
print(y_train.shape)

index = 0
for i in range(y_train1.shape[0]):
    if y_train1[i][0] == 3:
        x_train[index] = x_train1[i]
        index = index + 1
print(index)
for i in range(y_train1.shape[0]):
    if y_train1[i][0] != 3:
        x_train[index] = x_train1[i]
        index = index + 1
    if index == 10000:
        break

index = 0
for i in range(y_test1.shape[0]):
    if y_test1[i][0] == 3:
        x_test[index] = x_test1[i]
        index = index + 1
print(index)
for i in range(y_test1.shape[0]):
    if y_test1[i][0] != 3:
        x_test[index] = x_test1[i]
        index = index + 1
    if index == 2000:
        break

# shuffle
idx = np.random.permutation(x_train.shape[0])
x_train = x_train[idx]
y_train = y_train[idx]

idx = np.random.permutation(x_test.shape[0])
x_test = x_test[idx]
y_test = y_test[idx]


def my_loss(y_true, y_pred):
    # cross entropy loss function
    y_true = 1.0 * y_true
    loss = -(y_true*tf.log(y_pred)/tf.log(2.0) + (1-y_true)*tf.log(1-y_pred)/tf.log(2.0))

    return loss

def NN_network(x):
    x = Dense(256)(x)
    x = Dropout(0.35)(x)
    x = Activation('tanh')(x)

    x = Dense(32)(x)
    x = Dropout(0.05)(x)
    x = Activation('tanh')(x)

    # x = Dense(8)(x)
    # x = Dropout(0.05)(x)
    # x = Activation('tanh')(x)

    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    return x

image_tensor = Input(shape=(3072,))

NN_net = Model(inputs=[image_tensor], outputs=[NN_network(image_tensor)])

if loss_mode == 'mae':
    NN_net.compile(optimizer=SGD(lr=lr), loss='mae')
elif loss_mode == 'mse':
    NN_net.compile(optimizer=SGD(lr=lr), loss='mae')
elif loss_mode == 'cross_entropy':
    NN_net.compile(optimizer=SGD(lr=lr), loss=my_loss)

print(NN_net.summary())

def generator(batch, x_train_down, x_train_up):
    i = 0 - 11
    while True:
        # print(i)
        i = i % int(x_train_down.shape[0]/batch)
        input_x = []
        input_x_up = []
        
        for row in range(0, batch):
            XX = x_train_down[i*batch+row]
            XX_up = x_train_up[i*batch+row]
            input_x.append(XX)
            input_x_up.append(XX_up)
            
        batch_x = np.asarray(input_x)
        batch_x_up = np.asarray(input_x_up)
        i = i + 1
        yield ([batch_x], batch_x_up)

def generator_val(batch, x_val_down, x_val_up):
    j = 0 - 11
    while True:
        # print(j)
        j = j % int(x_val_down.shape[0]/batch)
        input_x = []
        input_x_up = []
        
        for row in range(0, batch):
            XX = x_val_down[j*batch+row]
            XX_up = x_val_up[j*batch+row]
            input_x.append(XX)
            input_x_up.append(XX_up)
            
        batch_x = np.asarray(input_x)
        batch_x_up = np.asarray(input_x_up)
        j = j + 1
        yield ([batch_x], batch_x_up)
        

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))
        
file = 'PretrainNet'
file_encoder = 'PretrainNet_encoder'
file_decoder = 'PretrainNet_decoder'

path = 'result/TensorBoard_%s' %file

save_dir = os.path.join(os.getcwd(), 'result/')
model_name = '%s_model.h5' % file
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

history = LossHistory()

# callbacks = [checkpoint, history, TensorBoard(log_dir = path)]
callbacks = [history, TensorBoard(log_dir = path)]

NN_net.fit_generator(generator=generator(batch_size, x_train, y_train), 
                          steps_per_epoch=int(x_train.shape[0]/batch_size), 
                          epochs=epochs, 
                          validation_data=generator_val(batch_size, x_test, y_test),
                          validation_steps=int(x_test.shape[0]/batch_size),
                          callbacks=callbacks
                          )

outfile = 'result/%s_model.h5' % file
NN_net.save_weights(outfile)
#Testing data

# outfile = 'result/%s_model.h5' % file
NN_net.load_weights(outfile)

tStart = time.time()
y_hat = NN_net.predict([x_test], batch_size=batch_size)
tEnd = time.time()
print ("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))

y_hat = (np.sign(y_hat - 0.5) + 1)/2

print(1- np.sum(np.abs(y_hat - y_test))/y_test.shape[0])


