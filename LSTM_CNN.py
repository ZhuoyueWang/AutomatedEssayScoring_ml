
from __future__ import print_function
import numpy as np
np.random.seed(4)  # for reproducibility

import tensorflow as tf
from keras.preprocessing import sequence

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
#from keras.datasets import imdb
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils.np_utils import to_categorical

import os, sys
# Embedding
max_features = 10000
maxlen = 400
embedding_size = 100

# Convolution
filter_length = 5
nb_filter = 64
pool_length = 4

# LSTM
lstm_output_size = 64

# Training
batch_size = 16
nb_epoch = 15
VALIDATION_SPLIT = 0.3

DATA_DIR = '../dataset/Processed/Keras_Processed/2_labels'

x_ept_dir = os.path.join(DATA_DIR, 'X_ept.npy')
y_ept_dir = os.path.join(DATA_DIR,'y_ept.npy')
x_ICNALE_dir = os.path.join(DATA_DIR,'X_ICNALE.npy')
y_ICNALE_dir = os.path.join(DATA_DIR,'y_ICNALE.npy')
x_kaggle_dir = os.path.join(DATA_DIR,'X_kaggle_dev.npy')
y_kaggle_dir = os.path.join(DATA_DIR,'y_kaggle_dev.npy')



with tf.device('/cpu:0'):

    print('Loading data...')
    x_kaggle = np.load(x_kaggle_dir)
    x_ICNALE = np.load(x_ICNALE_dir)
    x_ept = np.load(x_ept_dir)
    y_kaggle = np.load(y_kaggle_dir)
    y_ICNALE = np.load(y_ICNALE_dir)
    y_ept = np.load(y_ept_dir)

    print('Splitting data...')
# use the kaggle data for training model
    indices = np.arange(x_ICNALE.shape[0])
    np.random.shuffle(indices)
    data = x_ICNALE[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    labels = y_ICNALE[indices]

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]


    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    print('Building model...')

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam,
                  metrics=['accuracy'])

    csv_logger = CSVLogger('training.log')


# CheckPoint
filepath = "weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint, csv_logger]

history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(x_val, y_val), callbacks=callbacks_list)


# Save model to json and model parameters to h5
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
"""
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
"""

# Load accuracy history from History Callback
"""
epoch = history.epoch
acc = history.history['acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']

# Plotting training curves

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(epoch, acc,'b',label="training")
plt.plot(epoch, val_acc,'r',label="validation")
plt.xlabel('epoch')

plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1.05, 1), loc=10, borderaxespad=0.)
plt.figure(2)
plt.plot(epoch, loss,'b',label = "training")
plt.plot(epoch, val_loss, 'r', label = "validation")
plt.legend(bbox_to_anchor=(1.05, 1), loc=10, borderaxespad=0.)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()
"""
# Import model
"""
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
"""

# Evaluate model
"""
score, acc = model.evaluate(X_ept, y_ept, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

from sklearn.metrics import classification_report

yyy = loaded_model.predict_classes(X_test)

y_pred = to_categorical(yyy, nb_classes = None)
print(classification_report(y_test, y_pred))

"""
