from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Read 
path = '/home/averyma/accent-classification/mfsc/us_uk_mfsc/'
train_data = np.load( path + 'train_data.npy')
train_label = np.load( path + 'train_label.npy')

dev_data = np.load( path + 'dev_data.npy')
dev_label = np.load( path + 'dev_label.npy')

# Training Parameters
batch_size = 128
num_classes = 2
epochs = 12

# input utterance frame dimensions
utt_rows, utt_cols = 40, 45
train_data = train_data.reshape(train_data.shape[0], utt_rows, utt_cols, 1)
dev_data = dev_data.reshape(dev_data.shape[0], utt_rows, utt_cols, 1)
input_shape = (utt_rows, utt_cols, 1)
train_data = train_data.astype('float32')
dev_data = dev_data.astype('float32')
print('train_data shape:', train_data.shape)
print(train_data.shape[0], 'train samples')
print(dev_data.shape[0], 'dev samples')

train_label = keras.utils.to_categorical(train_label, num_classes)
dev_label = keras.utils.to_categorical(dev_label, num_classes)

model = Sequential()
model.add(Conv2D(160, kernel_size=(8, 45),
                 activation='sigmoid',
                 input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(6,1)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_data, train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(dev_data, dev_label))

score = model.evaluate(dev_data, dev_label, verbose=0)
print('Dev loss:', score[0])
print('Dev accuracy:', score[1])
