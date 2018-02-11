# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

X_train = []
X_val = []
X_test = []
for filename in ["VGG19", "Xception", "InceptionV3"]:
    bottleneck_features = np.load('bottleneck_features/Dog'+filename + 'Data.npz')
    train = bottleneck_features['train']
    valid = bottleneck_features['valid']
    test = bottleneck_features['test']
    X_train.append(train.reshape(train.shape[0], 1, 1, -1))
    X_val.append(valid.reshape(valid.shape[0], 1, 1, -1))
    X_test.append(test.reshape(test.shape[0], 1, 1, -1))

X_train = np.concatenate(X_train, axis = 3)
X_val = np.concatenate(X_val, axis = 3)
X_test = np.concatenate(X_test, axis = 3)

from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=X_train.shape[1:]))
model.add(Dropout(0.5))
model.add(Dense(133, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.ensemble.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(X_train, train_targets, 
          validation_data=(X_val, valid_targets),
          epochs=100, batch_size=10, callbacks=[checkpointer], verbose=2)
model.load_weights('saved_models/weights.best.ensemble.hdf5')
predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in X_test]

# report test accuracy
test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
