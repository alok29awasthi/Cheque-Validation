import  keras
from sklearn.model_selection import  train_test_split

import PIL
import os, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from matplotlib import ticker

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam
from keras import backend as K 


ROWS = 190  
COLS = 142 
CHANNELS = 3

TRAIN_DIR="D:/VIT/TARP/SIgnature recognition/train/"
TEST_DIR="D:/VIT/TARP/SIgnature recognition/test/"
SIGNATURE_CLASSES = ['alok','neha','priya', 'rahil']

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def get_images(fish):
    fish_dir = TRAIN_DIR+'{}'.format(fish)
    images = [fish+'/'+im for im in os.listdir(fish_dir)]
    return images

def read_image(src):
    import os
    from scipy import misc
    import cv2
    import imutils
    filepath=src
    im=cv2.imread(filepath)
    import scipy.misc  as mc
    return imutils.resize(im,height=190)
##    return mc.imresize(im,(ROWS,COLS))

files = []
y_all = []

for fish in SIGNATURE_CLASSES:
    fish_files = get_images(fish)
    files.extend(fish_files)
    
    y_fish = np.tile(fish, len(fish_files))
    y_all.extend(y_fish)
    print("{0} photos of {1}".format(len(fish_files), fish))
    
y_all = np.array(y_all)
print(len(files))
print(len(y_all))


X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(files): 
    X_all[i] = read_image(TRAIN_DIR+im)
    if i%1000 == 0: print('Processed {} of {}'.format(i, len(files)))

print(X_all.shape)

y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)




from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 
                                                    test_size=0.3, random_state=23, 
                                                    stratify=y_all)



optimizer = RMSprop(lr=1e-4)
objective = 'categorical_crossentropy'
def center_normalize(x):
    return (x - K.mean(x)) / K.std(x)
print('1')
model = Sequential()
model.add(Activation(activation=center_normalize, input_shape=(ROWS, COLS, CHANNELS)))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(96, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(96, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 2, 2, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 2, 2, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(len(SIGNATURE_CLASSES)))
model.add(Activation('sigmoid'))
 
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss=root_mean_squared_error)


early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')        
       
model.fit(X_train, y_train, batch_size=64, nb_epoch=20,
              validation_split=0.1, verbose=1, shuffle=True, callbacks=[early_stopping])
preds = model.predict(X_valid, verbose=1)
print("Validation Log Loss: {}".format(log_loss(y_valid, preds)))
model.save('signature_model.h5')

test_files = [im for im in os.listdir(TEST_DIR)]
test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)
print(TEST_DIR)
print(test_files)
for i, im in enumerate(test_files):
    print(im)
    test[i] = read_image(TEST_DIR+im)
    
test_preds = model.predict(test, verbose=1)
submission = pd.DataFrame(test_preds, columns=SIGNATURE_CLASSES)
submission.insert(0, 'image', test_files)
submission.head()

submission.to_csv('D:/VIT/TARP/SIgnature recognition/signatureResults.csv',index=False)
