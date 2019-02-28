# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 19:13:34 2019

@author: Daniel
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, Conv2D, Concatenate, AveragePooling2D, BatchNormalization
from keras.layers import MaxPool2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l1, l2, l1_l2
from keras_applications.imagenet_utils import _obtain_input_shape
from se_resnet import *
from se import squeeze_excite_block
from se_inception_resnet_v2 import inception_resnet_block, conv2d_bn
from keras import backend as K

"""
Data Augmentation Scripts
"""

# These scripts allow you to define a data augmentation pipeline.
# See transforms.py and train_image_generator_test.py for examples of usage.
import sys
sys.path.insert(0, '../input/data-augmentation')
import random
import numpy as np
from composition import Compose, OneOf
from transforms import *
#Make your own data augmentation function!
def aug_daniel(prob=0.66):
	return Compose([
		RandomRotate90(prob=0.5),
		Transpose(prob=0.5),
		Flip(prob=0.5),
		OneOf([
			CLAHE(clipLimit=2),
			IAASharpen(),
			IAAEmboss(),
			 OneOf([
				RandomContrast(),
				RandomBrightness(),
			]),
			Blur(),
			GaussNoise()
		], prob=0.5),
		HueSaturationValue(prob=0.5),
		ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, prob=0.5)
		], prob=prob)
		
def aug_alex(prob=0.5):
	return Compose([
			Flip()
			], prob=prob)   
			
def aug_victor(prob=0.9):
	return Compose([
		OneOf([
			CLAHE(clipLimit=2, prob=.6),
			IAASharpen(prob=.2),
			IAAEmboss(prob=.2)
		], prob=.9),
		OneOf([
			IAAAdditiveGaussianNoise(prob=.3),
			GaussNoise(prob=.7),
			], prob=.6),
		ToGray(prob=.25),
		#InvertImg(prob=.2),
		Remap(prob=.4),
		RandomRotate90(),
		Flip(),
		Transpose(),
		OneOf([
			MotionBlur(prob=.2),
			MedianBlur(blur_limit=3, prob=.3),
			Blur(blur_limit=3, prob=.5),
		], prob=.4),
		OneOf([
			RandomContrast(prob=.5),
			RandomBrightness(prob=.5),
		], prob=.5),
		ShiftScaleRotate(shift_limit=.0, scale_limit=0.0, rotate_limit=20, prob=.6),
		OneOf([
			Distort1(prob=.25),
			Distort2(prob=.25),
			#ElasticTransform(prob=.2),
			IAAPerspective(prob=.25),
			IAAPiecewiseAffine(prob=.25),
		], prob=.8),
		HueSaturationValue(prob=.5),
		ChannelShuffle(prob=.2)
	], prob=prob)

"""
Constants
"""
TRAIN_PATH = 'base_dir/train/'
VAL_PATH = 'base_dir/val/'
TEST_PATH = '../input/test'
SAMPLE_SIZE = 80000
IMAGE_SIZE = 96
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_VAL = 128

"""
Dataset
"""
print('Loading dataset...')

try:
	print(df)
except:
	df = pd.DataFrame({'path': glob(os.path.join('../input/histopathologic-cancer-detection/train/', '*.tif'))})
	df['id'] = df.path.map(lambda x: x.split('\\')[1].split(".")[0])
	labels = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')
	df_data = df.merge(labels, on='id')
	
	# Remove bad data
	df_data = df_data[df_data['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']
	df_data = df_data[df_data['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
	df_data.head(3)
	
	# Take a random sample of class 0 with size equal to num samples in class 1
	df_0 = df_data[df_data['label'] == 0].sample(SAMPLE_SIZE, random_state = 101)
	# Filter out class 1
	df_1 = df_data[df_data['label'] == 1].sample(SAMPLE_SIZE, random_state = 101)
	# Combine the dataframes
	df_data = shuffle(pd.concat([df_0, df_1], axis=0).reset_index(drop=True))
	
	# train_test_split # stratify=y creates a balanced validation set.
	y = df_data['label']
	df_train, df_val = train_test_split(df_data, test_size=0.10, random_state=101, stratify=y)
	
	for fold in [TRAIN_PATH, VAL_PATH]:
		for subf in ['0', '1']:
			folder = os.path.join(fold, subf)
			if not os.path.exists(folder):
				os.makedirs(folder)
			
	# Set the id as the index in df_data
	df_data.set_index('id', inplace=True)
	df_data.head()

	#print('Copying dataset...')
	##for image in df_train['id'].values:
	##	# the id in the csv file does not have the .tif extension therefore we add it here
	##	fname = image + '.tif'
	##	label = str(df_data.loc[image,'label']) # get the label for a certain image
	##	src = os.path.join('../input/histopathologic-cancer-detection/train', fname)
	##	dst = os.path.join(TRAIN_PATH, label, fname)
	##	shutil.copyfile(src, dst)
	##
	##for image in df_val['id'].values:
	##	fname = image + '.tif'
	##	label = str(df_data.loc[image,'label']) # get the label for a certain image
	##	src = os.path.join('../input/histopathologic-cancer-detection/train', fname)
	##	dst = os.path.join(VAL_PATH, label, fname)
	##	shutil.copyfile(src, dst)
	#

num_train_samples = len(df_train)
num_val_samples = len(df_val)

train_steps = np.ceil(num_train_samples / BATCH_SIZE_TRAIN)
val_steps = np.ceil(num_val_samples / BATCH_SIZE_VAL)


"""
TODO: Do some feature engineering and data augmentation here?
"""
augs = aug_daniel()
def preprocessing_with_aug(img):
	x = augs(image=img.astype(np.uint8))['image']
	
	#x = img
	if x.std() > 0:
		x = (x - x.mean()) / x.std()
		x = resize(x, img.shape)
#		x = np.pad(x, 32, mode='constant')
	else:
		x = img
#		x = np.pad(img, 32, mode='constant')
	return x
		
datagen = ImageDataGenerator(preprocessing_function=preprocessing_with_aug,
							 horizontal_flip=True,
							 vertical_flip=True,
							 zca_whitening=True)

train_gen = datagen.flow_from_directory(TRAIN_PATH,
										target_size=(IMAGE_SIZE, IMAGE_SIZE),
										batch_size=BATCH_SIZE_TRAIN,
										class_mode='binary', seed = 1)

val_gen = datagen.flow_from_directory(VAL_PATH,
										target_size=(IMAGE_SIZE, IMAGE_SIZE),
										batch_size=BATCH_SIZE_VAL,
										class_mode='binary')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(VAL_PATH,
										target_size=(IMAGE_SIZE, IMAGE_SIZE),
										batch_size=1,
										class_mode='binary',
										shuffle=True)

#Create model
print('Creating model...')
classes = 1
base_model = SEResNet50(include_top=False,
						weights='imagenet',
						input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),
						pooling="avg")


def base_model():
	#Total params: 2,386,689
	#Trainable params: 2,385,345
	
	kernel_size = (3,3)
	pool_size= (2,2)
	first_filters = 32
	second_filters = 64
	third_filters = 128
	
	dropout_conv = 0.3
	dropout_dense = 0.5
	regularizer = l1_l2(0.0001, 0.0001)
	
	model = Sequential()
	model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3), kernel_regularizer=regularizer))
	model.add(Conv2D(first_filters, kernel_size, use_bias=False, kernel_regularizer=regularizer))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(MaxPool2D(pool_size = pool_size)) 
	model.add(Dropout(dropout_conv))
	
	model.add(Conv2D(second_filters, kernel_size, use_bias=False, kernel_regularizer=regularizer))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(Conv2D(second_filters, kernel_size, use_bias=False, kernel_regularizer=regularizer))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(MaxPool2D(pool_size = pool_size))
	model.add(Dropout(dropout_conv))
	
	model.add(Conv2D(third_filters, kernel_size, use_bias=False, kernel_regularizer=regularizer))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(Conv2D(third_filters, kernel_size, use_bias=False, kernel_regularizer=regularizer))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(MaxPool2D(pool_size = pool_size))
	model.add(Dropout(dropout_conv))
	
	#model.add(GlobalAveragePooling2D())
	model.add(Flatten())
	model.add(Dense(256, use_bias=False, kernel_regularizer=regularizer))
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	model.add(Dropout(dropout_dense))
	model.add(Dense(1, activation = "sigmoid"))
	
	return model

def SEInceptionResnetCustom():
	kernel_size = (3,3)
	pool_size= (2,2)
	first_filters = 32
	second_filters = 64
	third_filters = 128
	
	dropout_conv = 0.3
	dropout_dense = 0.5
	regularizer = l1_l2(0.0001, 0.0001)
	
	# Determine proper input shape
	input_shape = _obtain_input_shape(
		(IMAGE_SIZE, IMAGE_SIZE, 3),
		default_size=96,
		min_size=96,
		data_format=K.image_data_format(),
		require_flatten=False)
	
	img_input = Input(shape=input_shape)
	
	# Stem block: 35 x 35 x 192
	x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid', activation='elu')
	x = conv2d_bn(x, 32, 3, padding='valid', activation='elu')
	x = conv2d_bn(x, 64, 3)
	x = MaxPooling2D(3, strides=2)(x)
	x = conv2d_bn(x, 80, 1, padding='valid', activation='elu')
	x = conv2d_bn(x, 192, 3, padding='valid', activation='elu')
	x = MaxPooling2D(3, strides=2)(x)

	# Mixed 5b (Inception-A block): 35 x 35 x 320
	branch_0 = conv2d_bn(x, 96, 1, activation='elu')
	branch_1 = conv2d_bn(x, 48, 1, activation='elu')
	branch_1 = conv2d_bn(branch_1, 64, 5, activation='elu')
	branch_2 = conv2d_bn(x, 64, 1, activation='elu')
	branch_2 = conv2d_bn(branch_2, 96, 3, activation='elu')
	branch_2 = conv2d_bn(branch_2, 96, 3, activation='elu')
	branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
	branch_pool = conv2d_bn(branch_pool, 64, 1, activation='elu')
	branches = [branch_0, branch_1, branch_2, branch_pool]
	channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
	x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

	# squeeze and excite block
	x = squeeze_excite_block(x)

	# 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
	for block_idx in range(1, 11):
		x = inception_resnet_block(x,
								   scale=0.17,
								   block_type='block35',
								   block_idx=block_idx)

	# Mixed 6a (Reduction-A block): 17 x 17 x 1088
	branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid', activation='elu')
	branch_1 = conv2d_bn(x, 256, 1, activation='elu')
	branch_1 = conv2d_bn(branch_1, 256, 3, activation='elu')
	branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid', activation='elu')
	branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
	branches = [branch_0, branch_1, branch_pool]
	x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

	# squeeze and excite block
	x = squeeze_excite_block(x)
	
#	model = Sequential()
#	model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3), kernel_regularizer=regularizer))
#	model.add(Conv2D(first_filters, kernel_size, use_bias=False, kernel_regularizer=regularizer))
#	model.add(BatchNormalization())
#	model.add(Activation("relu"))
#	model.add(MaxPool2D(pool_size = pool_size)) 
#	model.add(Dropout(dropout_conv))
#	
#	model.add(Conv2D(second_filters, kernel_size, use_bias=False, kernel_regularizer=regularizer))
#	model.add(BatchNormalization())
#	model.add(Activation("relu"))
#	model.add(Conv2D(second_filters, kernel_size, use_bias=False, kernel_regularizer=regularizer))
#	model.add(BatchNormalization())
#	model.add(Activation("relu"))
#	model.add(MaxPool2D(pool_size = pool_size))
#	model.add(Dropout(dropout_conv))
#	
#	model.add(Conv2D(third_filters, kernel_size, use_bias=False, kernel_regularizer=regularizer))
#	model.add(BatchNormalization())
#	model.add(Activation("relu"))
#	model.add(Conv2D(third_filters, kernel_size, use_bias=False, kernel_regularizer=regularizer))
#	model.add(BatchNormalization())
#	model.add(Activation("relu"))
#	model.add(MaxPool2D(pool_size = pool_size))
#	model.add(Dropout(dropout_conv))
#	
#	#model.add(GlobalAveragePooling2D())
#	model.add(Flatten())
#	model.add(Dense(256, use_bias=False, kernel_regularizer=regularizer))
#	model.add(BatchNormalization())
#	model.add(Activation("relu"))
#	model.add(Dropout(dropout_dense))
	
	# Final convolution block: 8 x 8 x 1536
	x = conv2d_bn(x, 816, 1, name='conv_7b')
	x = GlobalAveragePooling2D(name='avg_pool')(x)
	x = Dense(1, activation='sigmoid', name='predictions')(x)
	model = Model(inputs = img_input, outputs = x, name='se_inception_resnet_v2_custom')
	
	return model

model = SEInceptionResnetCustom()

# Compile the model
print('Compiling model...')
model.compile(Adam(0.01), loss = "binary_crossentropy", metrics=["accuracy"])
#model.load_weights('../input/weights/hcd_weights_SeRN_.h5')

callbacks = [
	EarlyStopping(monitor='val_loss', patience=6, verbose=1, restore_best_weights=True),
	ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1, factor=0.1),
	ModelCheckpoint('hcd_weights_base_SeIR_{epoch:02d}_{val_loss:.4f}.h5', monitor='val_loss', save_best_only=False)
]
model.summary()

print('Training...')
history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
					validation_data=val_gen,
					validation_steps=val_steps,
					epochs=10,
					callbacks=callbacks)
					
# Prediction
print('Predicting...')
y_pred_keras = model.predict_generator(test_gen, steps=len(test_gen), verbose=1)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_gen.classes, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
print('AUC: ', auc_keras)

# summarize history for accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure(3)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()