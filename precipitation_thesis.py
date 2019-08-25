#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:55:23 2019

@author: mingkuan   612060
"""

import keras.backend as K
import os
import numpy as np
import pylab as plt
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers.convolutional import Conv3D
#from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.backend import clip 

"""
os.chdir('/home/mingkuan/Desktop')
#os.chdir('/Users/mingkuanwu/Desktop')
i=0
with open('train.txt') as f:
    array = []
    for line in f:
        line = line.split(" ")
        array.append(line[:10201]) #40804 = 4*101*101

for i in array:
    tem = i[0].split(",")
    i[0] = tem[-1]

array = np.array(array)
array = array.astype('uint8')
array = array.reshape((10000,1,101,101))

data = array
data.shape

for i in range(1,15):
    with open('train.txt') as f:
        array_1 = []
        for line in f:
            line = line.split(" ")
            array_1.append(line[i*4*101*101:(i*4+1)*101*101])
    array_1 = np.array(array_1)
    array_1 = array_1.astype('uint8')    
    array_1 = array_1.reshape((10000,1,101,101))
    data = np.concatenate((data,array_1),axis=1)

data.shape
np.save('precipitation_height_1', data)
"""


'''
corr_vec[:20]
Out[11]: 
array([0.73919364, 0.67716772, 0.51100522, 0.46727919, 0.68365138,
       0.44796417, 0.60949635, 0.59427324, 0.64169548, 0.42225261,
       0.76154113, 0.56067588, 0.6394729 , 0.63988916, 0.64731068,
       0.62554742, 0.69201383, 0.54536371, 0.61678416, 0.52056691])

corr_vec[20:40]
Out[12]: 
array([0.30279804, 0.4303467 , 0.30921337, 0.33703039, 0.7297576 ,
       0.66401879, 0.66047476, 0.48397028, 0.53016545, 0.56780854,
       0.61436087, 0.6321898 , 0.54498433, 0.4650429 , 0.47228136,
       0.64216715, 0.57330789, 0.77321548, 0.56201957, 0.66461322])

corr_vec[40:60]
Out[13]: 
array([0.49287037, 0.57866577, 0.71029385, 0.89059671, 0.75960645,
       0.67485028, 0.75479932, 0.45991419, 0.72053796, 0.7950893 ,
       0.9482305 , 0.92554593, 0.77777245, 0.8791757 , 0.76461987,
       0.77880032, 0.66558019, 0.75563735, 0.77329565, 0.85731854])
'''






"""
Start loading data from here
"""
os.chdir('/home/mingkuan/Desktop')
data = np.load('precipitation_height_1.npy')
data.shape
data = data.reshape((10000,15,101,101,1))
data.dtype

which = 58 #np.random.randint(10000) # 21

fig = plt.figure(figsize=(10, 6))
for i in range(15):
    ax = fig.add_subplot(3,5,i+1)
    toplot = data[which][i, ::, ::, 0]
    plt.imshow(toplot)


for i in range(15):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 20:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=20, color='w')

    toplot = data[which, i, ::, ::, 0]
    plt.imshow(toplot)
 
i=0
fig = plt.figure(figsize=(10, 10)) 
toplot = data[which, i, ::, ::, 0]
plt.imshow(toplot)
plt.savefig('/home/mingkuan/Dropbox/thesis_slides/precipitation1-0.png', bbox_inches='tight')    
    
    
    
    
#  train set
X1_precipitation = data[:10000,:8,:,:,:]
X1_precipitation.shape

tem = np.zeros((10000,1,101,101,1))
x = tem.astype('uint8')
del tem
X2_precipitation = np.concatenate((x, data[:10000,8:14,:,:,:]), axis=1) 
X2_precipitation.shape
del x

y_precipitation = data[:10000,8:15,:,:,:]
y_precipitation.shape
    





"""
1-layer
"""
def define_models_1_precipitation(n_filter, filter_size):
    # define training encoder
    encoder_inputs = Input(shape=(None, 101, 101, 1))
    encoder_1 = ConvLSTM2D(filters = n_filter, kernel_size=filter_size, padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.0005), recurrent_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))
    encoder_outputs_1, encoder_state_h_1, encoder_state_c_1 = encoder_1(encoder_inputs)
    # define training decoder
    decoder_inputs = Input(shape=(None, 101, 101, 1))
    decoder_1 = ConvLSTM2D(filters=n_filter, kernel_size=filter_size, padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.0005), recurrent_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))
    decoder_outputs_1, _, _ = decoder_1([decoder_inputs, encoder_state_h_1, encoder_state_c_1])
    decoder_conv3d = Conv3D(filters=1, kernel_size=(1,1,16), padding='same', data_format='channels_last',
                            kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))
    decoder_outputs = decoder_conv3d(decoder_outputs_1)
    #clip(dec oder_outputs, 0, 255)
    
#    denselayer = Dense(1, activation='softmax')
#    decoder_outputs = denselayer(decoder_outputs)
    
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    #print(model.summary(line_length=250))
    
    # define inference encoder
    encoder_model = Model(encoder_inputs, [encoder_state_h_1, encoder_state_c_1])
    
    # define inference decoder
    decoder_state_input_h_1 = Input(shape=(101, 101, n_filter))
    decoder_state_input_c_1 = Input(shape=(101, 101, n_filter))

    decoder_output_1, decoder_state_h_1_new, decoder_state_c_1_new = decoder_1([decoder_inputs, decoder_state_input_h_1, decoder_state_input_c_1])
    decoder_output = decoder_conv3d(decoder_output_1)
    #clip(decoder_output, 0, 255)
    
#    decoder_output = denselayer(decoder_output)
    
    decoder_model = Model([decoder_inputs , decoder_state_input_h_1 , decoder_state_input_c_1],
                          [decoder_output, decoder_state_h_1_new, decoder_state_c_1_new])
    
    return model, encoder_model, decoder_model




"""
Fake lstm
"""
train_1_lstm, infenc_1_lstm, infdec_1_lstm = define_models_1_precipitation(n_filter=64, filter_size=3)

train_1_lstm.compile(loss='mse', optimizer='adam', metrics=['mae'])#, metrics=['mse'])

#train_2_moving_1.fit([X1_moving,X2_moving], y_moving, batch_size=8, validation_split=0.3, epochs=1)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
#cp = ModelCheckpoint('model-mnist-2layer.h5', verbose=1, save_best_only=True)
history_1_lstm = train_1_lstm.fit([X1_precipitation,X2_precipitation], y_precipitation, batch_size=8, 
                                          validation_split=0.25, epochs=2, callbacks=[es])




"""
test filter size = ?
"""
train_1_precipitation, infenc_1_precipitation, infdec_1_precipitation = define_models_1_precipitation(n_filter=96, filter_size=3)

train_1_precipitation.compile(loss='mse', optimizer='adam', metrics=['mae'])#, metrics=['mse'])

#train_2_moving_1.fit([X1_moving,X2_moving], y_moving, batch_size=8, validation_split=0.3, epochs=1)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
#cp = ModelCheckpoint('model-mnist-2layer.h5', verbose=1, save_best_only=True)
history_1_precipitation = train_1_precipitation.fit([X1_precipitation,X2_precipitation], y_precipitation, batch_size=8, 
                                          validation_split=0.25, epochs=30, callbacks=[es])








plt.plot(history_1_precipitation.history['loss'][:])
plt.plot(history_1_precipitation.history['val_loss'][:])
plt.title('1-layer convlstm model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



# generate target given source sequence
def predict_sequence_1(infenc, infdec, source, n_steps):
	# encode
	state_h_1, state_c_1 = infenc.predict(source)  # source_dim = ()
	#decoder_input = source[:,-1,:,:,:].reshape((1,1,64,64,1))
	decoder_input = np.repeat(0,101*101).reshape((1,1,101,101,1))
    #decoder_input = decoder_input.astype('float')
	# 123
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h_1, c_1 = infdec.predict([decoder_input, state_h_1, state_c_1])
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state_h_1, state_c_1 = h_1, c_1
		# update target sequence
		decoder_input = yhat
	return np.array(output)   
 








   
"""
2-layer 
"""    
def define_models_2_precipitation(n_filter, filter_size):
    # define training encoder
    encoder_inputs = Input(shape=(None, 101, 101, 1))
    encoder_1 = ConvLSTM2D(filters = n_filter, kernel_size=filter_size, padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.0005), recurrent_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))
    encoder_2 = ConvLSTM2D(filters = n_filter, kernel_size=filter_size, padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.0005), recurrent_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))
    encoder_outputs_1, encoder_state_h_1, encoder_state_c_1 = encoder_1(encoder_inputs)
    encoder_outputs_2, encoder_state_h_2, encoder_state_c_2 = encoder_2(encoder_outputs_1)
    # define training decoder
    decoder_inputs = Input(shape=(None, 101, 101, 1))
    decoder_1 = ConvLSTM2D(filters=n_filter, kernel_size=filter_size, padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.0005), recurrent_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))
    decoder_2 = ConvLSTM2D(filters=n_filter, kernel_size=filter_size, padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.0005), recurrent_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))
    decoder_outputs_1, _, _ = decoder_1([decoder_inputs, encoder_state_h_1, encoder_state_c_1])
    decoder_outputs_2, _, _ = decoder_2([decoder_outputs_1, encoder_state_h_2, encoder_state_c_2])
    decoder_conv3d = Conv3D(filters=1, kernel_size=(1,1,64), padding='same', data_format='channels_last',
                            kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))
    decoder_outputs = decoder_conv3d(decoder_outputs_2)
    #clip(dec oder_outputs, 0, 255)
    
#    denselayer = Dense(1, activation='softmax')
#    decoder_outputs = denselayer(decoder_outputs)
    
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    #print(model.summary(line_length=250))
    
    # define inference encoder
    encoder_model = Model(encoder_inputs, [encoder_state_h_1, encoder_state_c_1, encoder_state_h_2, encoder_state_c_2])
    
    # define inference decoder
    decoder_state_input_h_1 = Input(shape=(101,101,n_filter))
    decoder_state_input_c_1 = Input(shape=(101,101,n_filter))
    decoder_state_input_h_2 = Input(shape=(101,101,n_filter))
    decoder_state_input_c_2 = Input(shape=(101,101,n_filter))
    decoder_output_1, decoder_state_h_1_new, decoder_state_c_1_new = decoder_1([decoder_inputs, decoder_state_input_h_1, decoder_state_input_c_1])
    decoder_output_2, decoder_state_h_2_new, decoder_state_c_2_new = decoder_2([decoder_output_1, decoder_state_input_h_2, decoder_state_input_c_2])
    decoder_output = decoder_conv3d(decoder_output_2)
    #clip(decoder_output, 0, 255)
    
#    decoder_output = denselayer(decoder_output)
    
    decoder_model = Model([decoder_inputs , decoder_state_input_h_1 , decoder_state_input_c_1, decoder_state_input_h_2 , decoder_state_input_c_2],
                          [decoder_output, decoder_state_h_1_new, decoder_state_c_1_new, decoder_state_h_2_new, decoder_state_c_2_new])
    
    return model, encoder_model, decoder_model


train_2_precipitation, infenc_2_precipitation, infdec_2_precipitation = define_models_2_precipitation(n_filter=64, filter_size=3)

train_2_precipitation.compile(loss='mse', optimizer='adam', metrics=['mae'])

#train_2_precipitation.fit([X1_precipitation,X2_precipitation], y_precipitation, batch_size=8, validation_split=0.25, epochs=1)


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
history_2_precipitation = train_2_precipitation.fit([X1_precipitation,X2_precipitation],
                                                    y_precipitation, batch_size=8, validation_split=0.25, epochs=100, callbacks=[es])



plt.plot(history_2_precipitation.history['loss'][:])
plt.plot(history_2_precipitation.history['val_loss'][:])
plt.title('2-layer convlstm model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



# generate target given source sequence
def predict_sequence_2(infenc, infdec, source, n_steps):
	# encode
	state_h_1, state_c_1, state_h_2, state_c_2 = infenc.predict(source)  # source_dim = ()
	#decoder_input = source[:,-1,:,:,:].reshape((1,1,64,64,1))
	decoder_input = np.repeat(0,101*101).reshape((1,1,101,101,1))
    #decoder_input = decoder_input.astype('float')
	# 123
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h_1, c_1, h_2, c_2 = infdec.predict([decoder_input, state_h_1, state_c_1, state_h_2, state_c_2])
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state_h_1, state_c_1, state_h_2, state_c_2 = h_1, c_1, h_2, c_2
		# update target sequence
		decoder_input = yhat
	return np.array(output)


"""
Prediction on test set
"""
"""
which = 5

track = data[which,:10,:,:,:]
track.shape

history = track[np.newaxis, ::, ::, ::, ::]
history.shape

prediction = predict_sequence(infenc_2_precipitation, infdec_2_precipitation, history, 5)

prediction.shape

track = np.concatenate((track, prediction), axis=0)
track.shape

track2 = data[which][:15, :, :, ::] 
track2.shape

for i in range(15):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 10:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=20, color='w')

    toplot = track[i, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20, color='w')

    toplot = track2[i, ::, ::, 0]
    #if i >= 2:
    #    toplot = dt[which][i, ::, ::, 0]
        
    plt.imshow(toplot)
    plt.savefig('%i_animate.png' % (i + 1))
"""    





















"""
3-layer stacked ConvLSTM2D Encoder-Decoder
"""

def define_models_3_precipitation(n_filter, filter_size):
    # define training encoder
    encoder_inputs = Input(shape=(None, 101, 101, 1))
    encoder_1 = ConvLSTM2D(filters = n_filter, kernel_size=filter_size, padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001))
    encoder_2 = ConvLSTM2D(filters = 32, kernel_size=filter_size, padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001))
    encoder_3 = ConvLSTM2D(filters = 32, kernel_size=filter_size, padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001))
    encoder_outputs_1, encoder_state_h_1, encoder_state_c_1 = encoder_1(encoder_inputs)
    encoder_outputs_2, encoder_state_h_2, encoder_state_c_2 = encoder_2(encoder_outputs_1)
    encoder_outputs_3, encoder_state_h_3, encoder_state_c_3 = encoder_3(encoder_outputs_2)
    # define training decoder
    decoder_inputs = Input(shape=(None, 101, 101, 1))
    decoder_1 = ConvLSTM2D(filters=n_filter, kernel_size=filter_size, padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001))
    decoder_2 = ConvLSTM2D(filters=32, kernel_size=filter_size, padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001))
    decoder_3 = ConvLSTM2D(filters=32, kernel_size=filter_size, padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001))
    decoder_outputs_1, _, _ = decoder_1([decoder_inputs, encoder_state_h_1, encoder_state_c_1])
    decoder_outputs_2, _, _ = decoder_2([decoder_outputs_1, encoder_state_h_2, encoder_state_c_2])
    decoder_outputs_3, _, _ = decoder_3([decoder_outputs_2, encoder_state_h_3, encoder_state_c_3])
    decoder_conv3d = Conv3D(filters=1, kernel_size=(1,1,32), padding='same', data_format='channels_last',
                            kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))
    decoder_outputs = decoder_conv3d(decoder_outputs_3)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    #print(model.summary(line_length=250))
    # define inference encoder
    encoder_model = Model(encoder_inputs, 
                          [encoder_state_h_1, encoder_state_c_1, encoder_state_h_2, encoder_state_c_2, encoder_state_h_3, encoder_state_c_3])
    
    # define inference decoder
    decoder_state_input_h_1 = Input(shape=(101,101,n_filter))
    decoder_state_input_c_1 = Input(shape=(101,101,n_filter))
    decoder_state_input_h_2 = Input(shape=(101,101,32))
    decoder_state_input_c_2 = Input(shape=(101,101,32))
    decoder_state_input_h_3 = Input(shape=(101,101,32))
    decoder_state_input_c_3 = Input(shape=(101,101,32))
    decoder_output_1, decoder_state_h_1_new, decoder_state_c_1_new = decoder_1([decoder_inputs, decoder_state_input_h_1, decoder_state_input_c_1])
    decoder_output_2, decoder_state_h_2_new, decoder_state_c_2_new = decoder_2([decoder_output_1, decoder_state_input_h_2, decoder_state_input_c_2])
    decoder_output_3, decoder_state_h_3_new, decoder_state_c_3_new = decoder_3([decoder_output_2, decoder_state_input_h_3, decoder_state_input_c_3])
    decoder_output = decoder_conv3d(decoder_output_3)
    decoder_model = Model([decoder_inputs , decoder_state_input_h_1 , decoder_state_input_c_1, decoder_state_input_h_2 , decoder_state_input_c_2, 
                           decoder_state_input_h_3 , decoder_state_input_c_3],
                          [decoder_output, decoder_state_h_1_new, decoder_state_c_1_new, decoder_state_h_2_new, decoder_state_c_2_new, 
                           decoder_state_h_3_new, decoder_state_c_3_new])
    
    return model, encoder_model, decoder_model


train_3_precipitation, infenc_3_precipitation, infdec_3_precipitation = define_models_3_precipitation(n_filter=64, filter_size=3)

train_3_precipitation.compile(loss='mse', optimizer='adam', metrics=['mae'])

#train_3_moving.fit([X1_moving,X2_moving], y_moving, batch_size=8, validation_split=0.1, epochs=1)


#train_2.fit([X1,X2], y, batch_size=32, epochs=300, validation_split=0.1)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
history_3_precipitation = train_3_precipitation.fit([X1_precipitation, X2_precipitation],
                                                    y_precipitation, batch_size=8, validation_split=0.25, epochs=100, callbacks=[es])


plt.plot(history_3_precipitation.history['loss'])
plt.plot(history_3_precipitation.history['val_loss'])
plt.title('3-layer convlstm model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



# generate target given source sequence
def predict_sequence_3(infenc, infdec, source, n_steps):
	# encode
	state_h_1, state_c_1, state_h_2, state_c_2, state_h_3, state_c_3 = infenc.predict(source)  # source_dim = ()
#	decoder_input = source[:,-1,:,:,:].reshape((1,1,64,64,1))
	decoder_input = np.repeat(0,101*101).reshape((1,1,101,101,1))
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h_1, c_1, h_2, c_2, h_3, c_3 = infdec.predict([decoder_input, state_h_1, state_c_1, state_h_2, state_c_2, state_h_3, state_c_3])
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state_h_1, state_c_1, state_h_2, state_c_2, state_h_3, state_c_3 = h_1, c_1, h_2, c_2, h_3, c_3
		# update target sequence
		decoder_input = yhat
	return np.array(output)




















"""
Prediction on test set
"""
# which =21, 225, 5267, 3630

which =  np.random.randint(10000)
track = data[which,:10,:,:,:]
track.shape

history = track[np.newaxis, ::, ::, ::, ::]
history.shape

prediction_lstm = predict_sequence_1(infenc_1_lstm, infdec_1_lstm, history, 5)
prediction_1 = predict_sequence_1(infenc_1_precipitation, infdec_1_precipitation, history, 5)
prediction_2 = predict_sequence_2(infenc_2_precipitation, infdec_2_precipitation, history, 5)
prediction_3 = predict_sequence_3(infenc_3_precipitation, infdec_3_precipitation, history, 5)
prediction_1.shape

track = data[which,:,:,:,:]
track.shape


fig = plt.figure(figsize=(11,14))
for i in range(5):
    ax = fig.add_subplot(7,5,i+1)
    toplot = track[i, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(7,5,i+6)
    toplot = track[i+5, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(7,5,i+11)
    toplot = track[i+10, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(7,5,i+16)
    toplot = prediction_lstm[i, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(7,5,i+21)
    toplot = prediction_1[i, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(7,5,i+26)
    toplot = prediction_2[i, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(7,5,i+31)
    toplot = prediction_3[i, ::, ::, 0]
    plt.imshow(toplot)

plt.savefig('/home/mingkuan/Dropbox/ucalgary-thesis-master/precipitation_4.pdf', bbox_inches='tight')






which =  np.random.randint(10000)
track = data[which,:8,:,:,:]
track.shape

history = track[np.newaxis, ::, ::, ::, ::]
history.shape

prediction_lstm = predict_sequence_1(infenc_1_lstm, infdec_1_lstm, history, 7)
prediction_1 = predict_sequence_1(infenc_1_precipitation, infdec_1_precipitation, history, 7)
prediction_2 = predict_sequence_2(infenc_2_precipitation, infdec_2_precipitation, history, 7)
prediction_3 = predict_sequence_3(infenc_3_precipitation, infdec_3_precipitation, history, 7)
prediction_1.shape

track = data[which,:,:,:,:]
track.shape


fig = plt.figure(figsize=(22,22))
for i in range(7):
    ax = fig.add_subplot(6,7,i+1)
    toplot = track[i, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(6,7,i+8)
    toplot = track[i+7, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(6,7,i+15)
    toplot = prediction_lstm[i, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(6,7,i+22)
    toplot = prediction_1[i, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(6,7,i+29)
    toplot = prediction_2[i, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(6,7,i+36)
    toplot = prediction_3[i, ::, ::, 0]
    plt.imshow(toplot)

#plt.savefig('/home/mingkuan/Dropbox/ucalgary-thesis-master/precipitation_7output.pdf', bbox_inches='tight')











history_1_lstm.history['loss'][-1]
history_1_lstm.history['val_loss'][-1]
history_1_lstm.history['mean_absolute_error'][-1]
history_1_lstm.history['val_mean_absolute_error'][-1]


history_1_precipitation.history['loss'][-5]
history_1_precipitation.history['val_loss'][-5]
history_1_precipitation.history['mean_absolute_error'][-5]
history_1_precipitation.history['val_mean_absolute_error'][-5]

history_2_precipitation.history['loss'][-5]
history_2_precipitation.history['val_loss'][-5]
history_2_precipitation.history['mean_absolute_error'][-5]
history_2_precipitation.history['val_mean_absolute_error'][-5]

history_3_precipitation.history['loss'][-5]
history_3_precipitation.history['val_loss'][-5]
history_3_precipitation.history['mean_absolute_error'][-5]
history_3_precipitation.history['val_mean_absolute_error'][-5]

