#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:22:29 2019

@author: mingkuan
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
import math

'''
np.seterr(divide='ignore', invalid='ignore')

os.chdir('/home/mingkuan/Desktop')
data = np.load('precipitation_height_1.npy')
data.shape
data = data.reshape((10000,15,101,101,1))
data = data.astype(int)
data.dtype

x_old = data[:,0,:,:,0]
x_new = data[:,-1,:,:,0]

x_old.shape
x_new.shape

def cor(x,y):
    sum_xy=0
    sum_x=0
    sum_y=0
    for i in range(101):
        for j in range(101):
            sum_xy = sum_xy + x[i,j]*y[i,j]
            sum_x = sum_x + x[i,j]*x[i,j]
            sum_y = sum_y + y[i,j]*y[i,j]
    corr = sum_xy/(np.sqrt(sum_x)*np.sqrt(sum_y))    
    return corr

corr_vec = np.zeros(10000)
corr_vec.shape

for i in range(10000):
    corr_vec[i] = cor(x_old[i],x_new[i])

count = np.zeros(10000)
for i in range(10000):
    if (corr_vec[i] > 0.75 or math.isnan(corr_vec[i])==True):
        count[i] = 1
        
sum(count)       


drop_index = np.where(count==1)
drop_index = np.array(drop_index)
drop_index.shape
np.save('precipitation_drop_sample', drop_index)
'''


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

'''
os.chdir('/home/mingkuan/Desktop')
drop_index = np.load('precipitation_drop_sample.npy')
drop_index.shape
drop_index = list(drop_index.flatten())
drop_index[:10]

data = np.load('precipitation_height_1.npy')
data_new = data
data_new.shape
data_new = np.delete(data_new, drop_index, axis=0)
data_new = data_new[:,:,:,:,np.newaxis]
data_new.shape # (5485, 15, 101, 101, 1)

data = data_new

del data_new
del drop_index
'''




"""
Reverse order to double data
"""
'''
tem = np.zeros((5485,15,101,101,1))
tem = tem.astype('uint8')

for i in range(5485):
    for j in range(15):
        reverse = np.rot90(data[i,14-j,:,:,:])
        #reverse = data[i,14-j,:,:,:]
        tem[i,j,:,:,:] = reverse

# Check by plots
which = np.random.randint(5845) # 21,1895, 3288

fig = plt.figure(figsize=(8, 5))
for i in range(15):
    ax = fig.add_subplot(3,5,i+1)
    toplot = data[which][i, ::, ::, 0]
    plt.imshow(toplot)
    

fig = plt.figure(figsize=(8, 5))
for i in range(15):
    ax = fig.add_subplot(3,5,i+1)
    toplot = tem[which][i, ::, ::, 0]
    plt.imshow(toplot)        

data_new = np.concatenate((data, tem), axis=0)    
np.save('precipitation_reverse_order_double', data_new)    

data = np.load('precipitation_reverse_order_double.npy')
'''


"""
Rotate image to double the amount of images
"""
'''
data = data.reshape((5485,15,101,101))

tem = np.zeros((5485,15,101,101))
tem = tem.astype('uint8')
for i in range(5485):
    for j in range(15):
        rotate = np.rot90(data[i,j,:,:])
        tem[i,j,:,:] = rotate

del i,j,rotate 
        
tem = tem.reshape((5485,15,101,101,1))
tem.shape

data = data.reshape((5485,15,101,101,1))


which = 1895 #np.random.randint(10000) # 21,1895, 3288

fig = plt.figure(figsize=(10, 6))
for i in range(15):
    ax = fig.add_subplot(3,5,i+1)
    toplot = data[which][i, ::, ::, 0]
    plt.imshow(toplot)
    

fig = plt.figure(figsize=(10, 6))
for i in range(15):
    ax = fig.add_subplot(3,5,i+1)
    toplot = tem[which][i, ::, ::, 0]
    plt.imshow(toplot)
    
data_new = np.concatenate((data, tem), axis=0)    
    
np.save('precipitation_clean_rotate_double', data_new)    
'''


which = 1895 #np.random.randint(10000) # 21,1895, 3288

for i in range(15):
    fig = plt.figure(figsize=(10, 10))
    toplot = data[which][i, ::, ::, 0]
    plt.imshow(toplot)
    plt.savefig('precipitation1895-%i.png' % (i))

"""
Start here
"""

os.chdir('/home/mingkuan/Desktop')
#data = np.load('precipitation_clean_rotate_double.npy')
#data = np.load('precipitation_drop_sample.npy')
data = np.load('precipitation_reverse_order_double.npy')
data.shape

X1_precipitation = data[:,:8,:,:,:]
X1_precipitation.shape

tem = np.zeros((10970,1,101,101,1))
tem = tem.astype(int)

X2_precipitation = np.concatenate((tem, data[:,8:14,:,:,:]), axis=1)
X2_precipitation = X2_precipitation.astype('uint8')
X2_precipitation.shape


y_precipitation = data[:,8:15,:,:,:]
y_precipitation.shape
    
del tem


"""
1-layer
"""
def define_models_1_precipitation(n_filter, filter_size):
    # define training encoder
    encoder_inputs = Input(shape=(None, 101, 101, 1))
    encoder_1 = ConvLSTM2D(filters = n_filter, kernel_size=filter_size, activation='relu', padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.0005), recurrent_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))
    encoder_outputs_1, encoder_state_h_1, encoder_state_c_1 = encoder_1(encoder_inputs)
    # define training decoder
    decoder_inputs = Input(shape=(None, 101, 101, 1))
    decoder_1 = ConvLSTM2D(filters=n_filter, kernel_size=filter_size, activation='relu', padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.0005), recurrent_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))
    decoder_outputs_1, _, _ = decoder_1([decoder_inputs, encoder_state_h_1, encoder_state_c_1])
    decoder_conv3d = Conv3D(filters=1, kernel_size=(1,1,16), activation='relu', padding='same', data_format='channels_last',
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

history_1_lstm = train_1_lstm.fit([X1_precipitation,X2_precipitation], y_precipitation, batch_size=8, 
                                          validation_split=0.25, epochs=2)
'''
Prediction
'''
which = np.random.randint(5485)

track = data[which,:8,:,:,:]
track.shape

history = track[np.newaxis, ::, ::, ::, ::]
history.shape

prediction = predict_sequence_1(infenc_1_lstm, infdec_1_lstm, history, 7)

prediction.shape

track = np.concatenate((track, prediction), axis=0)
track.shape

track2 = data[which][::, ::, ::, ::] 
track2.shape


for i in range(15):
    fig = plt.figure(figsize=(5, 5))

    toplot = track[i, ::, ::, 0]
        
    plt.imshow(toplot)
    plt.savefig('precipitation_lstm_2743_-%i.png' % (i))












"""
1-layer ConvLSTM
"""
train_1_precipitation, infenc_1_precipitation, infdec_1_precipitation = define_models_1_precipitation(n_filter=64, filter_size=3)

train_1_precipitation.compile(loss='mse', optimizer='adam', metrics=['mae'])#, metrics=['mse'])

#train_2_moving_1.fit([X1_moving,X2_moving], y_moving, batch_size=8, validation_split=0.3, epochs=1)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
#cp = ModelCheckpoint('model-mnist-2layer.h5', verbose=1, save_best_only=True)
history_1_precipitation = train_1_precipitation.fit([X1_precipitation,X2_precipitation], y_precipitation, batch_size=8, 
                                          validation_split=0.25, epochs=100, callbacks=[es])



plt.plot(history_1_precipitation.history['loss'][:])
plt.plot(history_1_precipitation.history['val_loss'][:])
plt.title('1-layer convlstm model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

history_1_precipitation.history['loss'][-5:]
history_1_precipitation.history['val_loss'][-5:]
history_1_precipitation.history['mean_absolute_error'][-5:]
history_1_precipitation.history['val_mean_absolute_error'][-5:]



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
 
    

'''
prediction
'''


which = np.random.randint(5485)

track = data[which,:8,:,:,:]
track.shape

history = track[np.newaxis, ::, ::, ::, ::]
history.shape

prediction = predict_sequence_1(infenc_1_precipitation, infdec_1_precipitation, history, 7)

prediction.shape

track = np.concatenate((track, prediction), axis=0)
track.shape

track2 = data[which][::, ::, ::, ::] 
track2.shape


for i in range(15):
    fig = plt.figure(figsize=(5, 5))

    toplot = track2[i, ::, ::, 0]
        
    plt.imshow(toplot)
    plt.savefig('precipitation_truth_2743_-%i.png' % (i))


for i in range(15):
    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(111)

    toplot = track[i, ::, ::, 0]
        
    plt.imshow(toplot)
    plt.savefig('precipitation_1layer_2743_-%i.png' % (i))



fig = plt.figure(figsize=(10, 6))
for i in range(7):
    ax = fig.add_subplot(3,7,i+1)
    toplot = track2[i+1, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(3,7,i+8)
    toplot = track2[i+8, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(3,7,i+15)
    toplot = prediction[i, ::, ::, 0]
    plt.imshow(toplot)







   
"""
2-layer 
"""    
def define_models_2_precipitation(n_filter, filter_size):
    # define training encoder
    encoder_inputs = Input(shape=(None, 101, 101, 1))
    encoder_1 = ConvLSTM2D(filters = n_filter, kernel_size=filter_size, activation='relu', padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.0005), recurrent_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))
    encoder_2 = ConvLSTM2D(filters = n_filter, kernel_size=filter_size, activation='relu', padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.0005), recurrent_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))
    encoder_outputs_1, encoder_state_h_1, encoder_state_c_1 = encoder_1(encoder_inputs)
    encoder_outputs_2, encoder_state_h_2, encoder_state_c_2 = encoder_2(encoder_outputs_1)
    # define training decoder
    decoder_inputs = Input(shape=(None, 101, 101, 1))
    decoder_1 = ConvLSTM2D(filters=n_filter, kernel_size=filter_size, activation='relu', padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.0005), recurrent_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))
    decoder_2 = ConvLSTM2D(filters=n_filter, kernel_size=filter_size, activation='relu', padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.0005), recurrent_regularizer=l2(0.0005), bias_regularizer=l2(0.0005))
    decoder_outputs_1, _, _ = decoder_1([decoder_inputs, encoder_state_h_1, encoder_state_c_1])
    decoder_outputs_2, _, _ = decoder_2([decoder_outputs_1, encoder_state_h_2, encoder_state_c_2])
    decoder_conv3d = Conv3D(filters=1, kernel_size=(1,1,64), activation='relu', padding='same', data_format='channels_last',
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

history_2_precipitation = train_2_precipitation.fit([X1_precipitation,X2_precipitation],
                                                    y_precipitation, batch_size=8, validation_split=0.25, epochs=100, callbacks=[es])



plt.plot(history_2_precipitation.history['loss'][:])
plt.plot(history_2_precipitation.history['val_loss'][:])
plt.title('2-layer convlstm model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


history_2_precipitation.history['loss'][-5:]
history_2_precipitation.history['val_loss'][-5:]
history_2_precipitation.history['mean_absolute_error'][-5:]
history_2_precipitation.history['val_mean_absolute_error'][-5:]


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



'''
prediction
'''


which = np.random.randint(5485*2)

track = data[which,:8,:,:,:]
track.shape

history = track[np.newaxis, ::, ::, ::, ::]
history.shape

prediction = predict_sequence_2(infenc_2_precipitation, infdec_2_precipitation, history, 7)

prediction.shape

track = np.concatenate((track, prediction), axis=0)
track.shape

track2 = data[which][::, ::, ::, ::] 
track2.shape


for i in range(15):
    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(111)

    toplot = track[i, ::, ::, 0]
        
    plt.imshow(toplot)
    plt.savefig('precipitation_2layer_2743_-%i.png' % (i))


fig = plt.figure(figsize=(10, 6))
for i in range(7):
    ax = fig.add_subplot(3,7,i+1)
    toplot = track2[i+1, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(3,7,i+8)
    toplot = track2[i+8, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(3,7,i+15)
    toplot = prediction[i, ::, ::, 0]
    plt.imshow(toplot)












"""
3-layer stacked ConvLSTM2D Encoder-Decoder
"""

def define_models_3_precipitation(n_filter, filter_size):
    # define training encoder
    encoder_inputs = Input(shape=(None, 101, 101, 1))
    encoder_1 = ConvLSTM2D(filters = n_filter, kernel_size=filter_size, activation='relu', padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001))
    encoder_2 = ConvLSTM2D(filters = 32, kernel_size=filter_size, activation='relu', padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001))
    encoder_3 = ConvLSTM2D(filters = 32, kernel_size=filter_size, activation='relu', padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001))
    encoder_outputs_1, encoder_state_h_1, encoder_state_c_1 = encoder_1(encoder_inputs)
    encoder_outputs_2, encoder_state_h_2, encoder_state_c_2 = encoder_2(encoder_outputs_1)
    encoder_outputs_3, encoder_state_h_3, encoder_state_c_3 = encoder_3(encoder_outputs_2)
    # define training decoder
    decoder_inputs = Input(shape=(None, 101, 101, 1))
    decoder_1 = ConvLSTM2D(filters=n_filter, kernel_size=filter_size, activation='relu', padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001))
    decoder_2 = ConvLSTM2D(filters=32, kernel_size=filter_size, activation='relu', padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001))
    decoder_3 = ConvLSTM2D(filters=32, kernel_size=filter_size, activation='relu', padding='same', return_sequences=True, return_state=True,
                           kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001), bias_regularizer=l2(0.001))
    decoder_outputs_1, _, _ = decoder_1([decoder_inputs, encoder_state_h_1, encoder_state_c_1])
    decoder_outputs_2, _, _ = decoder_2([decoder_outputs_1, encoder_state_h_2, encoder_state_c_2])
    decoder_outputs_3, _, _ = decoder_3([decoder_outputs_2, encoder_state_h_3, encoder_state_c_3])
    decoder_conv3d = Conv3D(filters=1, kernel_size=(1,1,32), activation='relu', padding='same', data_format='channels_last',
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

history_3_precipitation = train_3_precipitation.fit([X1_precipitation, X2_precipitation],
                                                    y_precipitation, batch_size=8, validation_split=0.25, epochs=100, callbacks=[es])


plt.plot(history_3_precipitation.history['loss'])
plt.plot(history_3_precipitation.history['val_loss'])
plt.title('3-layer convlstm model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



history_3_precipitation.history['loss'][-5:]
history_3_precipitation.history['val_loss'][-5:]
history_3_precipitation.history['mean_absolute_error'][-5:]
history_3_precipitation.history['val_mean_absolute_error'][-5:]


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



del X1_precipitation
del X2_precipitation
del y_precipitation


'''
prediction
'''


which = 20#1895 # 21, 225, 5267, 3630, 
# 8028, 2171, 972, 2040, 6427, 3338

which = np.random.randint(5485*2)

track = data[which,:5,:,:,:]
track.shape

history = track[np.newaxis, ::, ::, ::, ::]
history.shape

prediction = predict_sequence_3(infenc_3_precipitation, infdec_3_precipitation, history, 10)

prediction.shape

track = np.concatenate((track, prediction), axis=0)
track.shape

track2 = data[which][::, ::, ::, ::] 
track2.shape





fig = plt.figure(figsize=(10, 6))
for i in range(10):
    ax = fig.add_subplot(3,10,i+1)
    toplot = track2[i, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(3,10,i+11)
    toplot = track2[i+5, ::, ::, 0]
    plt.imshow(toplot)
    
    ax = fig.add_subplot(3,10,i+21)
    toplot = prediction[i, ::, ::, 0]
    plt.imshow(toplot)


for i in range(15):
    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(111)

    toplot = track[i, ::, ::, 0]
        
    plt.imshow(toplot)
    plt.savefig('precipitation_3layer_2743_-%i.png' % (i))













"""
LSTM 2-layer 2048 nodes
"""
"""
def define_models_lstm(n_input, n_output, n_unit):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder_1 = LSTM(n_unit, return_sequences=True, return_state= True)
    encoder_outputs_1, state_h_1, state_c_1 = encoder_1(encoder_inputs)
    encoder_states_1 = [state_h_1, state_c_1]
    encoder_2 = LSTM(n_unit, return_state=True)
    encoder_outputs_2, state_h_2, state_c_2 = encoder_2(encoder_outputs_1)
    encoder_states_2 = [state_h_2, state_c_2]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm_1 = LSTM(n_unit, return_sequences=True, return_state=True)
    decoder_outputs_1,_,_ = decoder_lstm_1(decoder_inputs, initial_state=encoder_states_1)
    decoder_lstm_2 = LSTM(n_unit, return_sequences=True, return_state=True)
    decoder_outputs_2,_,_ = decoder_lstm_2(decoder_outputs_1, initial_state=encoder_states_2)
    decoder_dense = Dense(n_output, activation='relu')
    decoder_outputs = decoder_dense(decoder_outputs_2)
    
    model = Model([encoder_inputs,decoder_inputs], decoder_outputs)  # training model
    print(model.summary(line_length=250))
    
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states_1 + encoder_states_2)
    #print(encoder_model.summary())
    
    # define inference decoder
    decoder_state_input_h_1 = Input(shape=(n_unit,))
    decoder_state_input_c_1 = Input(shape=(n_unit,))
    decoder_state_inputs_1 = [decoder_state_input_h_1, decoder_state_input_c_1]
    decoder_state_input_h_2 = Input(shape=(n_unit,))
    decoder_state_input_c_2 = Input(shape=(n_unit,))
    decoder_state_inputs_2 = [decoder_state_input_h_2, decoder_state_input_c_2]    
    decoder_outputs_1, state_h_1, state_c_1 = decoder_lstm_1(decoder_inputs, initial_state = decoder_state_inputs_1)
#    decoder_states_1 = [state_h_1, state_c_1]
    decoder_outputs_2, state_h_2, state_c_2 = decoder_lstm_2(decoder_outputs_1, initial_state=decoder_state_inputs_2)
#    decoder_states_2 = [state_h_2, state_c_2]
    decoder_outputs = decoder_dense(decoder_outputs_2)
    
    decoder_model = Model([decoder_inputs] + decoder_state_inputs_1 + decoder_state_inputs_2, decoder_outputs)
    print(decoder_model.summary())
    return model, encoder_model, decoder_model   

# generate target given source sequence
def predict_sequence_lstm(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)   # array shape=(2, 1, 128)
	# start of sequence input
	target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)  # (1, 1, 51)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat = infdec.predict([target_seq] + state) # yhat.shape=(1,1,51),  h,c = (1,128)
		# store prediction
		output.append(yhat[0,0,:])    
		# update state
		#state = [h, c]
		# update target sequence
		target_seq = yhat
	return np.array(output)    

train_lstm, infenc_lstm, infdec_lstm = define_models_lstm(101*101, 101*101, 2048)  

X1_lstm = data[:,:10,:,:,:].reshape(5485,10,101*101)
X1_lstm.shape

tem = np.zeros((5485,1,101*101))
x = tem.astype(int)
x.shape
del tem
X2_lstm = np.concatenate((x, data[:,10:14,:,:,:].reshape(5485,4,101*101)), axis=1) 
X2_lstm.shape

y_lstm = data[:,10:15,:,:,:].reshape((5485,5,101*101))
y_lstm.shape

train_lstm.compile(loss='mse', optimizer='adam', metrics=['mae'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history_lstm = train_lstm.fit([X1_lstm, X2_lstm],y_lstm, batch_size=8, validation_split=0.25, epochs=1, callbacks=[es])
"""














"""
Prediction on test set , 10 predict 5
"""
# which = 10, 3288
"""
which =  np.random.randint(5485)
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

#plt.savefig('/home/mingkuan/Dropbox/ucalgary-thesis-master/precipitation_4.pdf', bbox_inches='tight')






8 predict 7
"""

# which = 1895, 1737
which = 20#1895 # 21, 225, 5267, 3630, 
# 8028, 252, 2171, 2040, 6427

which =  np.random.randint(5485)
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


fig = plt.figure(figsize=(9,9))
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











history_1_lstm.history['loss'][-1] #327.9122245820108
history_1_lstm.history['val_loss'][-1] #197.13978821334615
history_1_lstm.history['mean_absolute_error'][-1] #10.011888716651551
history_1_lstm.history['val_mean_absolute_error'][-1]#7.321181672655111


history_1_precipitation.history['loss'][-5]#271.51510785300167
history_1_precipitation.history['val_loss'][-5]#185.3573197790207
history_1_precipitation.history['mean_absolute_error'][-5]#8.835418028116168
history_1_precipitation.history['val_mean_absolute_error'][-5]#7.097247711771083

history_2_precipitation.history['loss'][-5]#195.3919241636477
history_2_precipitation.history['val_loss'][-5]#187.33512767202305
history_2_precipitation.history['mean_absolute_error'][-5]#7.422589238067338
history_2_precipitation.history['val_mean_absolute_error'][-5]#6.828466658689538

history_3_precipitation.history['loss'][-5]#203.0026483626195
history_3_precipitation.history['val_loss'][-5]#184.56706099885545
history_3_precipitation.history['mean_absolute_error'][-5]#7.500101906233625
history_3_precipitation.history['val_mean_absolute_error'][-5]#6.762557742894565















"""
Cpmpute metrics over time steps
"""

history_lstm = []
history_1 = []
history_2 = []
history_3 = []

for which in range(2500):
    track = data[which,:8,:,:,:]
    history = track[np.newaxis, ::, ::, ::, ::]
    prediction_lstm = predict_sequence_1(infenc_1_lstm, infdec_1_lstm, history, 7)
    prediction_1 = predict_sequence_1(infenc_1_precipitation, infdec_1_precipitation, history, 7)
    prediction_2 = predict_sequence_2(infenc_2_precipitation, infdec_2_precipitation, history, 7)
    prediction_3 = predict_sequence_3(infenc_3_precipitation, infdec_3_precipitation, history, 7)
    history_lstm.append(prediction_lstm)
    history_1.append(prediction_1)
    history_2.append(prediction_2)
    history_3.append(prediction_3)
    
history_lstm = np.array(history_lstm)
history_1 = np.array(history_1)
history_2 = np.array(history_2)
history_3 = np.array(history_3)

history_lstm = history_lstm.reshape((2500,7,101,101))  # for easy computation of corr, mse and mae
history_1 = history_1.reshape((2500,7,101,101))
history_2 = history_2.reshape((2500,7,101,101))
history_3 = history_3.reshape((2500,7,101,101))

history_lstm = history_lstm.astype(int) # SOLVE: __main__:7: RuntimeWarning: overflow encountered in ubyte_scalars
history_1 = history_1.astype(int)
history_2 = history_2.astype(int)
history_3 = history_3.astype(int)
                                  
np.save('history_lstm_7step', history_lstm)
np.save('history_1_7step', history_1)
np.save('history_2_7step', history_2)
np.save('history_3_7step', history_3)








"""
Save memory for 'float' dtype
Reload every thing from here
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
import math

os.chdir('/home/mingkuan/Desktop')
drop_index = np.load('precipitation_drop_sample.npy')
drop_index.shape
drop_index = list(drop_index.flatten())
drop_index[:10]

data = np.load('precipitation_height_1.npy')
data_new = data
data_new.shape
data_new = np.delete(data_new, drop_index, axis=0)
data_new = data_new[:,:,:,:,np.newaxis]
data_new.shape # (5485, 15, 101, 101, 1)

data = data_new[:3000,:,:,:,:]
data.shape
data.dtype
data = data.astype('float32')
del data_new

history_lstm = np.load('history_lstm.npy') # dtype = float32
history_1 = np.load('history_1.npy')
history_2 = np.load('history_2.npy')
history_3 = np.load('history_3.npy')

#history_lstm = history_lstm.astype(int) # SOLVE: __main__:7: RuntimeWarning: overflow encountered in ubyte_scalars
#history_1 = history_1.astype(int)
#history_2 = history_2.astype(int)
#history_3 = history_3.astype(int)
                                  


















np.seterr(divide='ignore', invalid='ignore')

def cor(x,y):
    sum_xy=0
    sum_x=0
    sum_y=0
    for i in range(101):
        for j in range(101):
            sum_xy = sum_xy + x[i,j]*y[i,j]
            sum_x = sum_x + x[i,j]*x[i,j]
            sum_y = sum_y + y[i,j]*y[i,j]
    corr = sum_xy/(np.sqrt(sum_x*sum_y))    
    return corr

def mse(x,y):
    error = 0
    for i in range(101):
        for j in range(101):
            error = error + np.square(x[i,j]-y[i,j])
    mse = error/(101*101)      
    return mse

def mae(x,y):
    error = 0
    for i in range(101):
        for j in range(101):
            error = error + np.abs(x[i,j]-y[i,j])
    mae = error/(101*101)      
    return mae



data = data.reshape((3000,15,101,101)) # for easy computation of corr, mse and mae
#data = data.astype(int) # SOLVE: __main__:7: RuntimeWarning: overflow encountered in ubyte_scalars
history_lstm = history_lstm.reshape((3000,5,101,101))  # for easy computation of corr, mse and mae
history_1 = history_1.reshape((3000,5,101,101))
history_2 = history_2.reshape((3000,5,101,101))
history_3 = history_3.reshape((3000,5,101,101))


cor_list_t1 = []
cor_list_t2 = []
cor_list_t3 = []
cor_list_t4 = []
cor_list_t5 = []

mse_list_t1 = []
mse_list_t2 = []
mse_list_t3 = []
mse_list_t4 = []
mse_list_t5 = []

mae_list_t1 = []
mae_list_t2 = []
mae_list_t3 = []
mae_list_t4 = []
mae_list_t5 = []


for i in range(3000):
    data_tem = data[i]
    prediction_lstm_tem = history_2[i] # change for different model
    
    cor_list_t1.append(cor(data_tem[10],prediction_lstm_tem[0]))
    mse_list_t1.append(mse(data_tem[10],prediction_lstm_tem[0]))
    mae_list_t1.append(mae(data_tem[10],prediction_lstm_tem[0]))   
    
    cor_list_t2.append(cor(data_tem[11],prediction_lstm_tem[1]))
    mse_list_t2.append(mse(data_tem[11],prediction_lstm_tem[1]))
    mae_list_t2.append(mae(data_tem[11],prediction_lstm_tem[1]))
    
    cor_list_t3.append(cor(data_tem[12],prediction_lstm_tem[2]))
    mse_list_t3.append(mse(data_tem[12],prediction_lstm_tem[2]))
    mae_list_t3.append(mae(data_tem[12],prediction_lstm_tem[2]))
    
    cor_list_t4.append(cor(data_tem[13],prediction_lstm_tem[3]))
    mse_list_t4.append(mse(data_tem[13],prediction_lstm_tem[3]))
    mae_list_t4.append(mae(data_tem[13],prediction_lstm_tem[3]))
    
    cor_list_t5.append(cor(data_tem[14],prediction_lstm_tem[4]))
    mse_list_t5.append(mse(data_tem[14],prediction_lstm_tem[4]))
    mae_list_t5.append(mae(data_tem[14],prediction_lstm_tem[4]))




mean_cor_lstm_t1 = np.nanmean(cor_list_t1)
mean_cor_lstm_t2 = np.nanmean(cor_list_t2)
mean_cor_lstm_t3 = np.nanmean(cor_list_t3)
mean_cor_lstm_t4 = np.nanmean(cor_list_t4)
mean_cor_lstm_t5 = np.nanmean(cor_list_t5)

cor_lstm = [mean_cor_lstm_t1, mean_cor_lstm_t2, mean_cor_lstm_t3, mean_cor_lstm_t4, mean_cor_lstm_t5 ]
cor_lstm

mean_cor_lstm_t1 = np.nanmean(mse_list_t1)
mean_cor_lstm_t2 = np.nanmean(mse_list_t2)
mean_cor_lstm_t3 = np.nanmean(mse_list_t3)
mean_cor_lstm_t4 = np.nanmean(mse_list_t4)
mean_cor_lstm_t5 = np.nanmean(mse_list_t5)

mse_lstm = [mean_cor_lstm_t1, mean_cor_lstm_t2, mean_cor_lstm_t3, mean_cor_lstm_t4, mean_cor_lstm_t5 ]
mse_lstm

mean_cor_lstm_t1 = np.nanmean(mae_list_t1)
mean_cor_lstm_t2 = np.nanmean(mae_list_t2)
mean_cor_lstm_t3 = np.nanmean(mae_list_t3)
mean_cor_lstm_t4 = np.nanmean(mae_list_t4)
mean_cor_lstm_t5 = np.nanmean(mae_list_t5)

mae_lstm = [mean_cor_lstm_t1, mean_cor_lstm_t2, mean_cor_lstm_t3, mean_cor_lstm_t4, mean_cor_lstm_t5 ]
mae_lstm



cor_lstm = [0.8964, 0.8279, 0.7743, 0.7254, 0.6931]
mse_lstm = [325.7047, 551.7955, 781.8535, 974.7883, 1139.3068]
mae_lstm = [8.9844, 12.2250, 15.3140, 17.6506, 19.6739]

cor_1 = [0.9004, 0.8479, 0.8075, 0.7745, 0.7605]
mse_1 = [317.6280, 509.6701, 704.6673, 879.5682, 1031.4036]
mae_1 = [8.7772, 11.5506, 14.1127, 16.1555, 17.9652]

cor_2 = [0.9031, 0.8564, 0.8210, 0.7870, 0.7683]
mse_2 = [301.1242, 503.1764, 669.7236, 834.9872, 970.3201]
mae_2 = [8.6331, 11.3365, 13.3976, 15.3696, 16.8739]

cor_3 = [0.91122, 0.8633, 0.8241, 0.7916, 0.7703]
mse_3 = [285.7048, 491.7956, 651.8535, 804.7884, 959.3068]
mae_3 = [8.2844, 11.0251, 13.1141, 15.0506, 16.6740]


# libraries
import matplotlib.pyplot as plt
import pandas as pd
 



fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(2,2,1)
df1=pd.DataFrame({'x': [1,2,3,4,5], 'y1': cor_lstm , 'y2': cor_1, 'y3': cor_2, 'y4': cor_3  })
# multiple line plot
plt.plot( 'x', 'y1', data=df1, marker='o', color='black', linewidth=2, label="FC-LSTM 2-layer")
plt.plot( 'x', 'y2', data=df1, marker='o', color='olive', linewidth=2, linestyle='dashed', label="ConvLSTM 1-layer")
plt.plot( 'x', 'y3', data=df1, marker='o', color='blue', linewidth=2, linestyle='dashed', label="ConvLSTM 2-layer")
plt.plot( 'x', 'y4', data=df1, marker='o', color='red', linewidth=2 ,linestyle='dashed', label="ConvLSTM 3-layer")
plt.xlabel('Time Steps')
plt.ylabel('Correlation between ground truth and prediction')
plt.legend()

ax = fig.add_subplot(2,2,2)
df2=pd.DataFrame({'x': range(1,6), 'y1': mse_lstm , 'y2': mse_1, 'y3': mse_2, 'y4': mse_3  })
# multiple line plot
plt.plot( 'x', 'y1', data=df2, marker='o', color='black', linewidth=2, label="FC-LSTM 2-layer")
plt.plot( 'x', 'y2', data=df2, marker='o', color='olive', linewidth=2, linestyle='dashed', label="ConvLSTM 1-layer")
plt.plot( 'x', 'y3', data=df2, marker='o', color='blue', linewidth=2, linestyle='dashed', label="ConvLSTM 2-layer")
plt.plot( 'x', 'y4', data=df2, marker='o', color='red', linewidth=2 ,linestyle='dashed', label="ConvLSTM 3-layer")
plt.xlabel('Time Steps')
plt.ylabel('MSE')
plt.legend()

ax = fig.add_subplot(2,2,3)
df3=pd.DataFrame({'x': range(1,6), 'y1': mae_lstm , 'y2': mae_1, 'y3': mae_2, 'y4': mae_3  })
# multiple line plot
plt.plot( 'x', 'y1', data=df3, marker='o', color='black', linewidth=2, label="FC-LSTM 2-layer")
plt.plot( 'x', 'y2', data=df3, marker='o', color='olive', linewidth=2, linestyle='dashed', label="ConvLSTM 1-layer")
plt.plot( 'x', 'y3', data=df3, marker='o', color='blue', linewidth=2, linestyle='dashed', label="ConvLSTM 2-layer")
plt.plot( 'x', 'y4', data=df3, marker='o', color='red', linewidth=2 ,linestyle='dashed', label="ConvLSTM 3-layer")
plt.xlabel('Time Steps')
plt.ylabel('MAE')
plt.legend()

plt.savefig('/home/mingkuan/Dropbox/ucalgary-thesis-master/precipitation_timestep.pdf', bbox_inches='tight')








