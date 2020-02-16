#File I/O
import os
import glob
import cv2
import numpy as np
from datetime import datetime
import argparse
import shutil
from shutil import copyfile

#Etc
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

#Keras Library
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#Define parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, default='vgg16', help='Model Architecture')
parser.add_argument('--weights', required=False, default='None')
parser.add_argument('--learning-rate', required=False, type=float, default=1e-4)
parser.add_argument('--semi-train', required=False, default=None)
parser.add_argument('--batch-size', required=False, type=int, default=8)
parser.add_argument('--random-split', required=False, type=int, default=1)
parser.add_argument('--data-augment', required=False, type=int, default=0)
args = parser.parse_args()
#Suffix에 사용하기 위해 위와 같은 정의를 사용합니다.
#명령 프롬프트에서 재현시 인자값만 변경하여 모델을 재현할 수 있도록 구성합니다.
#추후 다른 정의 없이 args.로 사용할 수 있는 것 같습니다.


#Clear and Make train/valid, cache, subm folders
def _clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


#Define classification model
def get_model():
    if args.weights == 'None':
        args.weights = None
    if args.model in ['vgg16']:
        base_model = VGG16(include_top=False, weights=args.weights, input_shape=(img_row_size, img_col_size,3))
    elif args.model in ['vgg19']:
        base_model = VGG19(include_top=False, weights=args.weights, input_shape=(img_row_size, img_col_size,3))
    elif args.model in ['resnet50']:
        base_model = ResNet50(include_top=False, weights=args.weights, input_shape=(img_row_size, img_col_size,3))
    else:
        print('# {} is not a valid value for "--model"'.format(args.model))
        exit()

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


#Make training data & validation data (Colorinput)
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

    #Clear & Make suffix folders
    for path in [temp_train_fold, temp_valid_fold, cache, subm]:
        _clear_dir(path)

    datagen = ImageDataGenerator(rescale = 1./255)

    #5-fold cross evaluation
    for fold in range(nfolds):
        model = get_model()
        train_samples, valid_samples = generate_split()

        #flow_from_directory는 하위 폴더 파일도 다 불러오는 듯함. test 폴더 안에 하위폴더가 있음. 아니면 test_genrator를 3개를 만들어야?
        train_generator = datagen.flow_from_directory(
            directory=temp_train_fold, target_size=(img_row_size, img_col_size), batch_size=args.batch_size, class_mode='categorical', seed=2018)
        valid_generator = datagen.flow_from_directory(
            directory=temp_valid_fold, target_size=(img_row_size, img_col_size), batch_size=args.batch_size, class_mode='categorical', seed=2018)
        test_generator = datagen.flow_from_directory(
            directory='input/test', target_size=(img_row_size, img_col_size), batch_size=1, class_mode=None, shuffle=False)

        test_id = glob.glob('{}/imgs/*.jpg'.format(test_path))

        weight_path = 'cache/weight.fold_{}.h5'.format(nfolds)

        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0), ModelCheckpoint(weight_path, monitor='val_loss',
                    save_best_only=True, verbose=0)]

        model.fit_generator(train_generator, steps_per_epoch=train_samples/args.batch_size,
                            epochs=3, validation_data=valid_generator, validation_steps=valid_samples/args.batch_size,
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
        #result.index는 c0~c9이다?
        #loc는 컬럼명을 추출하는 판다스함수, 시리즈는 저런 형태의 데이터 구조를 만들어줌
        #현재 런을 하면 모델을 5번 만들고 모델을 만들 때마다 3번씩 테스트 데이터로 예측함.
        sub_file = '../subm/{}/f{}.csv'.format(suffix, fold)
        result.to_csv(sub_file, index=False)

        shutil.rmtree(temp_train_fold)
        shutil.rmtree(temp_valid_fold)

    #Ensemble results in sub_file 1,2,3 (3 interation precdiction)
    print('# Ensemble')

    ensemble = 0
    for fold in range(nfolds):
        ensemble += pd.read_csv('subm/{}/f{}.csv'.format(suffix, fold), index_col=-1).values * 1. / nfolds
    ensemble = pd.DataFrame(ensemble, columns=labels)
    ensemble.loc[:, 'img'] = pd.Series(test_id, index=ensemble.index)
    sub_file = 'subm/{}/ens.csv'.format(suffix)
    ensemble.to_csv(sub_file, index=False)


