# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l1, l2, l1_l2

#"""
#Data Augmentation Scripts
#"""
#
## These scripts allow you to define a data augmentation pipeline.
## See transforms.py and train_image_generator_test.py for examples of usage.
## Originally derived from existing kaggle project: 
## Ask Daniel if you have questions about the below scripts
## copy our files into the working directory (make sure it has .py suffix)
#from shutil import copyfile
#copyfile(src = "../input/data-augmentation/augmentors.py", dst = "../working/augmentors.py") 
#copyfile(src = "../input/data-augmentation/composition.py", dst = "../working/composition.py") 
#copyfile(src = "../input/data-augmentation/functional.py", dst = "../working/functional.py") 
#copyfile(src = "../input/data-augmentation/transforms.py", dst = "../working/transforms.py") 
## import data augmentation pipeline "victor" - may be overkill, but we can modify it
##from transforms import aug_victor
##augs = aug_victor()
##Input: rgb image; Output: augmented version of image. May want to augment same input multiple times.
## augmented_img = augs(image=img)
#
#import random
#import numpy as np
#from composition import Compose, OneOf
#import functional as F
#from imgaug import augmenters as iaa
#from transforms import *
##Make your own data augmentation function!
#def aug_daniel(prob=0.5):
#    return Compose([
#            CLAHE(clipLimit=2, prob=1.0)
#            ], prob=prob)
#        
#def aug_alex(prob=0.5):
#    return Compose([
#            Flip()
#            ], prob=prob)
#            
#            
#def aug_victor(prob=0.9):
#    return Compose([
#        OneOf([
#            CLAHE(clipLimit=2, prob=.6),
#            IAASharpen(prob=.2),
#            IAAEmboss(prob=.2)
#        ], prob=.9),
#        OneOf([
#            IAAAdditiveGaussianNoise(prob=.3),
#            GaussNoise(prob=.7),
#            ], prob=.6),
#        ToGray(prob=.25),
#        #InvertImg(prob=.2),
#        Remap(prob=.4),
#        RandomRotate90(),
#        Flip(),
#        Transpose(),
#        OneOf([
#            MotionBlur(prob=.2),
#            MedianBlur(blur_limit=3, prob=.3),
#            Blur(blur_limit=3, prob=.5),
#        ], prob=.4),
#        OneOf([
#            RandomContrast(prob=.5),
#            RandomBrightness(prob=.5),
#        ], prob=.5),
#        ShiftScaleRotate(shift_limit=.0, scale_limit=0.0, rotate_limit=20, prob=.6),
#        OneOf([
#            Distort1(prob=.25),
#            Distort2(prob=.25),
#            #ElasticTransform(prob=.2),
#            IAAPerspective(prob=.25),
#            IAAPiecewiseAffine(prob=.25),
#        ], prob=.8),
#        HueSaturationValue(prob=.5),
#        ChannelShuffle(prob=.2)
#    ], prob=prob)
#
"""
Constants
"""
TRAIN_PATH = 'base_dir/train/'
VAL_PATH = 'base_dir/val/'
TEST_PATH = '../input/test'
SAMPLE_SIZE = 100
IMAGE_SIZE = 96
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 32

"""
Dataset
"""
print('Loading dataset...')

df = pd.DataFrame({'path': glob('../input/histopathologic-cancer-detection/train/*.tif')})
df['id'] = df.path.map(lambda x: x.split('\\')[1].split(".")[0])
labels = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')
df_data = df.merge(labels, on='id')

# Remove bad data
df_data = df_data[df_data['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']
df_data = df_data[df_data['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
df_data.head(3)

# Take a random sample of class 0 with size equal to num samples in class 1
df_0 = df_data[df_data['label'] == 0].sample(SAMPLE_SIZE, random_state=101)
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

print('Copying dataset...')
for image in df_train['id'].values:
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    label = str(df_data.loc[image,'label']) # get the label for a certain image
    src = os.path.join('../input/histopathologic-cancer-detection/train', fname)
    dst = os.path.join(TRAIN_PATH, label, fname)
    shutil.copyfile(src, dst)

for image in df_val['id'].values:
    fname = image + '.tif'
    label = str(df_data.loc[image,'label']) # get the label for a certain image
    src = os.path.join('../input/histopathologic-cancer-detection/train', fname)
    dst = os.path.join(VAL_PATH, label, fname)
    shutil.copyfile(src, dst)

num_train_samples = len(df_train)
num_val_samples = len(df_val)

train_steps = np.ceil(num_train_samples / BATCH_SIZE_TRAIN)
val_steps = np.ceil(num_val_samples / BATCH_SIZE_VAL)


"""
TODO: Do some feature engineering and data augmentation here?
"""
augs = aug_alex()
def preprocessing_with_aug(img):
    x = augs(image=img.astype(np.uint8))['image']
    if x.std() > 0:
        x = (x - x.mean()) / x.std()
        x = resize(x, img.shape)
        return x
    else:
        return img
        
datagen = ImageDataGenerator(preprocessing_function=preprocessing_with_aug,
                             horizontal_flip=True,
                             vertical_flip=True)

train_gen = datagen.flow_from_directory(TRAIN_PATH,
                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                        batch_size=BATCH_SIZE_TRAIN,
                                        class_mode='binary')

val_gen = datagen.flow_from_directory(VAL_PATH,
                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                        batch_size=BATCH_SIZE_VAL,
                                        class_mode='binary')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(VAL_PATH,
                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='binary',
                                        shuffle=False)

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

# Compile the model
print('Compiling model...')
model.compile(Adam(0.01), loss = "binary_crossentropy", metrics=["accuracy"])
#If we want to save and load individual weigth files, postfix them with validation accuracy.
#model_checkpoint = ModelCheckpoint('weights_{val_accuracy:.5f}.h5', monitor='val_loss', save_best_only=True)
#model.save_weights('cf_weights.h5')
#model_checkpoint = ModelCheckpoint('cf_weights.h5', monitor='val_loss', save_best_only=True)
#model.load_weights('cf_weights.h5')
#
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.1),
    model_checkpoint
]

print('Training...')
history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=2,
                    callbacks=callbacks)
                    
# Prediction
#print('Predicting...')
#y_pred_keras = model.predict_generator(test_gen, steps=len(test_gen), verbose=1)
#fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_gen.classes, y_pred_keras)
#auc_keras = auc(fpr_keras, tpr_keras)
#print('AUC: ', auc_keras)

# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.show()