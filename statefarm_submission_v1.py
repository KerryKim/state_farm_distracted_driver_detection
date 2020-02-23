#File I/O
import os
import argparse
import numpy as np
import pandas as pd
import shutil
from shutil import copyfile
from datetime import datetime
from glob import glob

#Image processing
import cv2
from scipy.ndimage import rotate
import scipy.misc

#Keras Library
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#Etc
import warnings
warnings.filterwarnings("ignore")


#Define learning parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, default='vgg16', help='Model Architecture')
parser.add_argument('--weights', required=False, default='None')
parser.add_argument('--learning-rate', required=False, type=float, default=1e-4)
parser.add_argument('--semi-train', required=False, default=None)
parser.add_argument('--batch-size', required=False, type=int, default=2)
parser.add_argument('--random-split', required=False, type=int, default=1)
parser.add_argument('--data-augment', required=False, type=int, default=0)
args = parser.parse_args()


#Clear and Make train/valid, cache, subm folders
def _clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


#Define model
def get_model():
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(img_row_size, img_col_size,3))

    out = Flatten()(base_model.output)
    out = Dense(fc_size, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(fc_size, activation='relu')(out)
    out = Dropout(0.5)(out)
    output = Dense(10, activation='softmax')(out)
    model = Model(inputs=base_model.input, outputs=output)

    sgd = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#Make train data & valid data
def generate_split():
    #Make folders to save K fold data set(train/valid)
    def _generate_temp_folder(root_path):
        _clear_dir(root_path)
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
                copyfile(fl, 'input/temp_train_fold/{}/{}'.format(label, os.path.basename(fl)))
                train_samples += 1
            else:
                copyfile(fl, 'input/temp_valid_fold/{}/{}'.format(label, os.path.basename(fl)))
                valid_samples += 1

    print('# {} train samples | {} valid samples'.format(train_samples, valid_samples))
    return train_samples, valid_samples


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def crop_center(img, cropx, cropy):
    # 이미지 중간을 Crop하는 함수를 정의한다
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


def preprocess(image):
    # rotate
    rotate_angle = np.random.randint(40) - 20
    image = rotate(image, rotate_angle)

    # zoom
    width_zoom = int(img_row_size * (0.9 + 0.1 * (1 - np.random.random())))
    height_zoom = int(img_col_size * (0.9 + 0.1 * (1 - np.random.random())))
    final_image = np.zeros((height_zoom, width_zoom, 3))
    final_image[:,:,0] = crop_center(image[:,:,0], width_zoom, height_zoom)
    final_image[:,:,1] = crop_center(image[:,:,1], width_zoom, height_zoom)
    final_image[:,:,2] = crop_center(image[:,:,2], width_zoom, height_zoom)

    # resize
    image = cv2.resize(final_image, (img_row_size, img_col_size))
    return image


if __name__ == "__main__":
    print('# Train Model')

    #Define parameters
    fc_size = 2048
    n_class = 10
    nfolds = 5
    test_nfolds = 1
    seed = 10
    img_row_size, img_col_size = 224, 224
    labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    suffix = 'm{}.w{}.lr{}.s{}.nf{}.semi{}.b{}.row{}col{}.rsplit{}.augment{}.d{}'.format(args.model, args.weights, args.learning_rate, seed, nfolds, args.semi_train, args.batch_size, img_row_size, img_col_size, args.random_split, args.data_augment, datetime.now().strftime("%Y-%m-%d-%H-%M"))

    train_path = 'input/train'
    test_path = 'input/test'
    temp_train_fold = 'input/temp_train_fold'
    temp_valid_fold = 'input/temp_valid_fold'
    cache = 'cache/{}'.format(suffix)
    subm = 'subm/{}'.format(suffix)

    for path in [temp_train_fold, temp_valid_fold, cache, subm]:
        _clear_dir(path)

    datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess)

    #5-fold cross evaluation
    for fold in range(nfolds):
        model = get_model()
        train_samples, valid_samples = generate_split()

        train_generator = datagen.flow_from_directory(
            directory=temp_train_fold, target_size=(img_row_size, img_col_size), batch_size=args.batch_size, class_mode='categorical', seed=2018)
        valid_generator = datagen.flow_from_directory(
            directory=temp_valid_fold, target_size=(img_row_size, img_col_size), batch_size=args.batch_size, class_mode='categorical', seed=2018)
        test_generator = datagen.flow_from_directory(
            directory='input/test', target_size=(img_row_size, img_col_size), batch_size=1, class_mode=None, shuffle=False)

        test_id = [os.path.basename(fl) for fl in glob('{}/imgs/*.jpg'.format(test_path))]

        weight_path = 'cache/{}/weight.fold_{}.h5'.format(suffix, fold)

        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0), ModelCheckpoint(weight_path, monitor='val_loss',
                    save_best_only=True, verbose=0)]

        model.fit_generator(train_generator, steps_per_epoch=train_samples/args.batch_size,
                            epochs=5, validation_data=valid_generator, validation_steps=valid_samples/args.batch_size,
                            shuffle=True, callbacks=callbacks, verbose=1)

        #Predict test data
        for j in range(test_nfolds):
            preds = model.predict_generator(test_generator, steps=len(test_id), verbose=1)
            if j == 0:
                result = pd.DataFrame(preds, columns=labels)
            else:
                result += pd.DataFrame(preds, columns=labels)
        result /= test_nfolds
        result.loc[:, 'img'] = pd.Series(test_id, index=result.index)

        sub_file = 'subm/{}/f{}.csv'.format(suffix, fold)
        result.to_csv(sub_file, index=False)

        shutil.rmtree(temp_train_fold)
        shutil.rmtree(temp_valid_fold)

    #Ensemble 5-folds
    print('# Ensemble')

    ensemble = 0
    for fold in range(nfolds):
        ensemble += pd.read_csv('subm/{}/f{}.csv'.format(suffix, fold), index_col=-1).values * 1. / nfolds
    ensemble = pd.DataFrame(ensemble, columns=labels)
    ensemble.loc[:, 'img'] = pd.Series(test_id, index=ensemble.index)
    sub_file = 'subm/{}/ens.csv'.format(suffix)
    ensemble.to_csv(sub_file, index=False)