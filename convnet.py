# !usr/bin/python

from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
import numpy as np

def main():

	# Hyper-parameters:

	batch_size = 32 # in each iteration, we consider 32 training examples at once
	num_epochs = 1 # we iterate 200 times over the entire training set
	kernel_size = 3 # we will use 3x3 kernels throughout
	pool_size = 2 # we will use 2x2 pooling throughout
	conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
	conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
	drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
	drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
	hidden_size = 512 # the FC layer will have 512 neurons

	inp = Input(shape=(height, width, depth)) 

	# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
	conv_1 = Convolution2D(conv_depth_1, 
		(kernel_size, kernel_size), padding='same', 
		activation='relu')(inp)

	conv_2 = Convolution2D(conv_depth_1, 
		(kernel_size, kernel_size), padding='same', 
		activation='relu')(conv_1)

	pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)

	drop_1 = Dropout(drop_prob_1)(pool_1)

	# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
	conv_3 = Convolution2D(conv_depth_2, 
		(kernel_size, kernel_size), padding='same', 
		activation='relu')(drop_1)

	conv_4 = Convolution2D(conv_depth_2, 
		(kernel_size, kernel_size), padding='same', 
		activation='relu')(conv_3)

	pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)

	drop_2 = Dropout(drop_prob_1)(pool_2)

	# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
	flat = Flatten()(drop_2)
	hidden = Dense(hidden_size, activation='relu')(flat)
	drop_3 = Dropout(drop_prob_2)(hidden)

	out = Dense(num_classes, activation='softmax')(drop_3)

	model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

	model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
	              optimizer='adam', # using the Adam optimiser
	              metrics=['accuracy']) # reporting the accuracy
