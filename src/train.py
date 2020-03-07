from keras.layers import Dense, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv2D, MaxPooling3D, Conv3D, MaxPooling2D, Activation, Input
from keras import backend as K
from sequence_generator import DataGenerator
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

import os
import matplotlib.pyplot as plt

############## Globals #############
# Defines the global variables for #
# batch size, epochs and classes   #
####################################

BATCH = 18
EPOCH = 1
CLASSES = 3
SEQ_LEN = 8
TRAIN_DIR = os.path.abspath('../data/train')
VAL_DIR = os.path.abspath('../data/val')

########### End Globals ############

class CNN_LSTM():

	@staticmethod
	def build_model(ip_shape, nb_classes):
		def add_block(model, num_filters, kernel_dim, init, reg, padding='valid'):
			x = TimeDistributed(Conv2D(num_filters,
			                           kernel_size=kernel_dim,
			                           padding=padding,
			                           kernel_initializer=init,
			                           kernel_regularizer=l2(reg)))(model)
			x = TimeDistributed(BatchNormalization())(x)
			x = TimeDistributed(Activation('relu'))(x)
			return x

		initialiser = 'glorot_uniform'
		reg_lambda = 0.001

		input = Input(shape=ip_shape)
		# input2 = Input(shape=ip_shape)

		b1 = add_block(input, 32, 7, initialiser, reg_lambda, 'same')
		b2 = add_block(b1, 32, 3, initialiser, reg_lambda)
		pool1 = TimeDistributed(MaxPooling2D((2, 2), strides=2))(b2)

		b3 = add_block(pool1, 64, 3, initialiser, reg_lambda, 'same')
		b4 = add_block(b3, 64, 3, initialiser, reg_lambda, 'same')
		pool2 = TimeDistributed(MaxPooling2D((2, 2), strides=2))(b4)

		b5 = add_block(pool2, 64, 3, initialiser, reg_lambda, 'same')
		b6 = add_block(b5, 64, 3, initialiser, reg_lambda, 'same')
		pool3 = TimeDistributed(MaxPooling2D((2, 2), strides=2))(b6)

		b7 = add_block(pool3, 64, 3, initialiser, reg_lambda, 'same')
		b8 = add_block(b7, 64, 3, initialiser, reg_lambda, 'same')
		pool4 = TimeDistributed(MaxPooling2D((2, 2), strides=2))(b8)

		b9 = add_block(pool4, 64, 3, initialiser, reg_lambda, 'same')
		b10 = add_block(b9, 64, 3, initialiser, reg_lambda, 'same')
		pool5 = TimeDistributed(MaxPooling2D((2, 2), strides=2))(b10)

		# LSTM output head
		flat = TimeDistributed(Flatten())(pool5)

		lstm = LSTM(256, return_sequences=False, dropout=0.5)(flat)
		# gru = GRU(256, return_sequences=False,dropout=0.5)(flat)
		output = Dense(nb_classes, activation='softmax')(lstm)

		model = Model(inputs=input, outputs=output)

		return model

def train():
	if K.backend() == 'tensorflow':
		K.common.set_image_dim_ordering("th")

	if K.image_data_format() == 'channels_first':
		IP_SHAPE = (SEQ_LEN, 3, 224, 224)
	else:
		IP_SHAPE = (SEQ_LEN, 224, 224, 3)

	tr_data_gen = DataGenerator('../data', SEQ_LEN, 'train', BATCH, (224, 224))
	num_train = tr_data_gen.__len__()

	val_data_gen = DataGenerator('../data', SEQ_LEN, 'val', BATCH, (224, 224))
	num_val = val_data_gen.__len__()

	model = CNN_LSTM().build_model(IP_SHAPE, CLASSES)
	model.summary()
	# Now compile the network.
	optimizer = Adam(lr=1e-5, decay=1e-6)
	model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])

	history = model.fit_generator(generator=tr_data_gen,
	                              steps_per_epoch=num_train // BATCH,
	                              validation_data=val_data_gen,
	                              validation_steps=num_val,
	                              epochs=EPOCH, verbose=1, workers=4
	                              )

	score = model.evaluate()
	print("Test Loss: ", score[0])
	print("Test accuracy: ", score[1] * 100)
	plt.plot(history.history['acc'],
	         label='Train Accuracy', color='red')
	plt.plot(history.history['val_acc'],
	         label='Validation Accuracy')
	plt.title('Accuracy of training and validation')
	plt.legend()

train()
