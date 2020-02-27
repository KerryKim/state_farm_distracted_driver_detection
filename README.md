# State Farm Distracted Driver Detection_Kaggle Competition

안녕하세요 이번에는 캐글 대회 중 State Farm에서 주최한 이미지 분류에 대한 코딩입니다. 총 22,424개의 학습 데이터가 10개의 클래스로 분류됩니다. 테스트 데이터는 총 79,726개 입니다.

본 포스트는 정권우님의 '머신러닝 탐구생활'과 마스터 코드를 바탕으로 작성하였습니다.

### 　
# 1. Prerequisites

https://www.kaggle.com/c/state-farm-distracted-driver-detection
캐글에 가입하고 데이터 세트를 다운받습니다.

### 　
# 2. Let's get it started!
###
### 1) 필요한 모듈을 모두 import 합니다. 
- warnings.filterwarnings("ignore")은 Future Warning을 제거하기 위해 선언해 줍니다.
```
#File I/O
import os
import argparse
import numpy as np
import pandas as pd
import shutil
from shutil import copyfile, copytree
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

```
### 　
### 2) 모델 구현 후 학습 파라미터를 변경하여 모델 재현을 할 수 있도록 ArumentParser() 함수를 사용합니다.
### 
```
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

```
### 　
### 3) 모델을 Run할 때 마다 Input, Cache, Subm 폴더에 남아있을 수 있는 파일들을 제거하고 폴더를 재생성하여 모델이 다시 학습할 수 있는 상태로 만들어 줍니다.
- os.path.exists() : 파일이 있는지 확인
- shutil.rmtree() : 폴더 삭제
- os.mkdir() : 파일 생성
### 
```
def _clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    
```
### 　
### 4) 학습에 사용할 모델을 선언해 줍니다. 
- weights='imagenet'으로 입력하면 이미지넷의 학습된 가중치를 가져옵니다. 
```
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
    
```
### 　
### 5) 비지도 학습을 하기 위해 Base model로 학습한 result 데이터의 정보를 이용합니다. 
```
#Make semi_train data
def semi_supervised():
    test_pred_fname = 'dataaug/Submission_v1_data augmentation_200223_origin.csv'
    test_pred = pd.read_csv(test_pred_fname)
    test_pred_probs = test_pred.iloc[:, :-1]
    test_pred_probs_max = np.max(test_pred_probs.values, axis=1)
```
###
- test_pred는 c0, c1, c2, ..., c9, img 열로 구성된 리스트 데이터입니다.
- test_pred_probs는 [:, :-1] 범위로 (모든 행)X(c0,...c9)으로 구성된 리스트 데이터입니다.
- test_pred_probs_max는 test_pred_probs의 데이터 중 각 행에서 최대값만 가져온 리스트 데이터입니다.
###
```
    for thr in range(1, 10):
        thr = thr / 10.
        count = sum(test_pred_probs_max > thr)
        print('# Thre : {} | count : {} ({}%)'.format(thr, count, 1. * count / len(test_pred_probs_max)))
```
###
- for thr in range에서는 count는 test_pred_probs_max에 있는 데이터 중에서 thr보다 크면 그 갯수를 모두 더합니다.
- len(test_pred_probs_max)는 test 이미지의 갯수가 됩니다. (총 이미지수)
###
```
    print('=' * 50)
    threshold = 0.90
    count = {}
    print('# Extracting data with threshold : {}'.format(threshold))

    copytree('input/train', 'input/semi_train')
```
###
- copyfile은 파일을 복사하는 것이고 디렉토리(폴더)를 생성하면서 복사하려면 copytree를 써야한다.
###
```
    for i, row in test_pred.iterrows():
        img = row['img']
        row = row.iloc[:-1]
        if np.max(row) > threshold:
            label = row.values.argmax()
            copyfile('input/test/imgs/{}'.format(img), 'input/semi_train/c{}/{}'.format(label, img))
            count[label] = count.get(label, 0) + 1

    print('# Added semi-supservised labels: \n{}'.format(count))
```
###
- .iterrows()를 사용하면 첫 번째 변수 i에는 행번호, 두번째 변수 row에는 그 행에 대한 값을 출력할 수 있습니다.
- img에는 한 행의 엑셀 img 이름으로 된 데이터입니다. for 문을 돌면서 첫 번째행, 두번째 행.. 이렇게 순차적으로 들어갑니다.
- row는 한 행의 엑셀 c0~c9의 확률값을 갖고 있습니다.
- label에는 c0~c9중 가장 큰 값에 있는 라벨값(행 index 열 label)을 가져옵니다.
- count는 갯수를 셉니다.
###
###
###
### 6) 모델에 입력가능한 train/valid data를 만들어 줍니다. 
```
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
        files = glob('input/semi_train/{}/*jpg'.format(label))
        for fl in files:
            if np.random.randint(nfolds)!= 1:
                copyfile(fl, 'input/temp_train_fold/{}/{}'.format(label, os.path.basename(fl)))
                train_samples += 1
            else:
                copyfile(fl, 'input/temp_valid_fold/{}/{}'.format(label, os.path.basename(fl)))
                valid_samples += 1

    print('# {} train samples | {} valid samples'.format(train_samples, valid_samples))
    return train_samples, valid_samples

```
###
###
### 7) Data augmentation을 하기 위해 img에 변위를 줍니다. 
- // 는 나누기에서 몫을 구할때 씁니다. 
- 또한 이미지의 세로의 길이 만큼 행을 만들어야 하고 가로의 길이만큼 열을 만들어야 하므로 가로, 세로의 크기를 반대로 써줘야 합니다.
- Data Augmentation시 이미지의 변화가 너무 크면 로컬 미니멈으로 빠지는 경향이 있어 학습이 되지 않습니다.
  학습이 되지 않는 경우 변위를 줄여서 안정적으로 학습할 수 있도록 설정해 줍니다.
- 특히 기존 코드에서 어파인 변환은 되지 않아 삭제했고 줌은 0.8에서 0.9로 변경했습니다.
```
def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def crop_center(img, cropx, cropy):
    # Crop
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

```
###
###
### 8) 실제로 main함수가 런되는 부분을 구현해 줍니다. 
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
    semi_train_path = 'input/semi_train'
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

        semi_supervised()
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

        shutil.rmtree(semi_train_path)
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
