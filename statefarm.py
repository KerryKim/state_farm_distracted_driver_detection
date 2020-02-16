#File I/O
import os
import glob
import numpy as np
import subprocess
from datetime import datetime
import argparse
import shutil
from shutil import copyfile
from sklearn.model_selection import KFold

# Image processing
import cv2
from scipy.ndimage import rotate
import scipy.misc

#Etc
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

#Keras Library
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#Define parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, default='vgg16', help='Model Architecture')
parser.add_argument('--weights', required=False, default='None')
parser.add_argument('--learning_rate', required=False, type=float, default=1e-4)
parser.add_argument('--semi_train', required=False, default=None)
parser.add_argument('--batch_size', required=False, type=int, default=8)
parser.add_argument('--random_split', required=False, type=int, default=1)
parser.add_argument('--data_augment', required=False, type=int, default=0)
args = parser.parse_args()

#Define classification model
def get_model():
    base_model = VGG16(include_top=False, weights=None, input_shape=(img_row_size, img_col_size,3))

    out = Flatten()(base_model.output)
    out = Dense(fc_size, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(fc_size, activation='relu')(out)
    out = Dropout(0.5)(out)
    output = Dense(10, activation='softmax')(out)
    model = Model(inputs=base_model.input, outputs=output)

    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#Make training data & validation data
def generate_split():
    #Make folders to save K fold data set(train/valid) // 이 함수를 사용하기전에 cd 디렉토리를 작업공간으로 바꿔놓는다.
    def _generate_temp_folder(root_path):
        os.mkdir(root_path)
        for i in range(n_class):
                os.mkdir('{}/c{}'.format(root_path,i))

    _generate_temp_folder(temp_train_fold)
    _generate_temp_folder(temp_valid_fold)

    train_samples =0
    valid_samples =0

    for label in labels:
        files = glob.glob('input/train/{}/*jpg'.format(label))
        for fl in files:
            if np.random.randint(nfolds)!= 1:
                copyfile(fl, 'input/temp_train_fold/{}/{}'.format(label,os.path.basename(fl)))
                train_samples += 1
            else:
                copyfile(fl, 'input/temp_valid_fold/{}/{}'.format(label,os.path.basename(fl)))
                valid_samples += 1

    print('# {} train samples | {} valid samples'.format(train_samples, valid_samples))
    return train_samples, valid_samples


if __name__ == "__main__":

    print('# Train Model')

    fc_size = 2048
    n_class = 10
    nfolds = 5
    batch_size = 8
    seed = 10
    img_row_size, img_col_size = 224, 224
    labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

    train_path = 'input/train'
    test_path = 'input/test'
    temp_train_fold = 'input/temp_train_fold'
    temp_valid_fold = 'input/temp_valid_fold'

    datagen = ImageDataGenerator()

    for fold in range(nfolds):
        model = get_model()
        train_samples, valid_samples = generate_split()

        train_generator = datagen.flow_from_directory(
            directory='input/temp_train_fold', target_size=(img_row_size, img_col_size), batch_size=8, class_mode='categorical', seed=2018)
        valid_generator = datagen.flow_from_directory(
            directory='input/temp_valid_fold', target_size=(img_row_size, img_col_size), batch_size=8, class_mode='categorical', seed=2018)

        weight_path = 'checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(500, 8)

        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0), ModelCheckpoint(weight_path, monitor='val_loss',
                    save_best_only=True, verbose=0)]

        model.fit_generator(train_generator, steps_per_epoch=train_samples/batch_size,
                            epochs=500, validation_data=valid_generator, validation_steps=valid_samples/batch_size,
                            shuffle=True, callbacks=callbacks, verbose=1)
