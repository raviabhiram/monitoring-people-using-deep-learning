from keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical

import numpy as np
import pandas as pd

class DataGenerator(Sequence):
	__slots__ = 'data_dir', 'seq_length', 'type', 'batch_size', 'len', 'classes', 'files', 'target_size', 'x', 'y'

	def __init__(self, data_dir, seq_length, type, batch_size=32, target_size=(224, 224)):
		self.x = []
		self.y = []
		self.data_dir = data_dir
		self.seq_length = seq_length or 1
		self.type = type
		self.batch_size = batch_size
		self.target_size = target_size
		self.class_map = {}
		self.num_classes = 0

		## Process data needed for the ImageDataProcessor
		self.process_data_file()

	def __len__(self):
		'''
		Function to return the number of batches of data.
		'''
		return int(np.floor(len(self.x) // (self.batch_size * self.seq_length)))

	def __getitem__(self, idx):
		'''
		Function to return one batch of sequences of items.
		'''
		## Needs to be updated to return batches of image sequences.
		batch_x = self.x[idx * self.seq_length * self.batch_size:(idx + 1) * self.seq_length * self.batch_size]
		batch_y = self.y[idx * self.seq_length * self.batch_size:(idx + 1) * self.seq_length * self.batch_size]
		xs = []
		ys = []
		for i in range(self.batch_size):
			seq = []
			for j in range(self.seq_length):
				image = load_img(batch_x[i + j], target_size=self.target_size)
				img_arr = img_to_array(image)
				seq.append((img_arr / 255.).astype(np.float32))
			xs.append(seq)
			ys.append(batch_y[i + j])

		# xs = np.array(
		# 	[[np.stack(resize(imread(batch_x[i + j]), self.target_size)) for j in range(self.seq_length)] for i in
		# 	 range(self.batch_size)])
		# print(len(batch_y))
		# ys = np.array(
		# 	[batch_y[k] for k in range(0, self.batch_size * self.seq_length + 1, self.seq_length)])
		return np.array(xs), to_categorical(np.array(ys), len(self.class_map.keys()))

	# return np.array([resize(imread(file_name), self.target_size) for file_name in batch_x]), np.array(batch_y)

	def process_data_file(self):
		file_metadata = pd.read_csv(self.data_dir + '/data_file.csv')
		filtered = file_metadata.loc[file_metadata['type'] == self.type]

		for index, row in filtered.iterrows():
			self.x.append(self.data_dir + '/' + row['img_path'])
			if row['class'] not in self.class_map:
				self.class_map[row['class']] = self.num_classes
				self.num_classes += 1
			self.y.append(self.class_map[row['class']])
