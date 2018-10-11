from __future__ import print_function
import numpy as np
from scipy import stats
np.random.seed(4)  # for reproducibility
import os, sys
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + '/home/yanan/Desktop/NewsNet/glove.6B'
TEXT_DATA_DIR = BASE_DIR + '/home/yanan/Desktop/AES/dataset/ICNALE/ICNALE'
EPT_DATA_DIR = BASE_DIR + '/home/yanan/Desktop/AES/dataset/EPT'

VALIDATION_SPLIT = 0.2
# Embedding
MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 400
embedding_size = 128


# Training
batch_size = 16
nb_epoch = 80

with tf.device('/cpu:0'):


    print('Processing ICNALE text dataset')

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path) and name[-8:-6] != 'XX':
            label_id = name[-9:-5]
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                texts.append(f.read())
                f.close()
                labels.append(label_id)
    print('Found %s texts in the ICNALE dataset.' % len(texts))



    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # word_index will contain the dictionary
    # keys are all the tokens appeared in the texts file
    # values are the corresponding indices

    word_index = tokenizer.word_index
    print('Found %s unique tokens in the ICNALE dataset.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


    #Some processing on the labels from string to int
    label_int = []
    for label in labels:
        if label == 'A2_0':
            label_int.append(0)
        elif label == 'B1_1':
            label_int.append(0)
        elif label == 'B1_2':
            label_int.append(1)
        elif label == 'B2_0':
            label_int.append(1)



    labels = to_categorical(np.asarray(label_int))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    np.save('X_ICNALE.npy', data)
    np.save('y_ICNALE.npy', labels)
    print('Processing EPT text dataset')

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(EPT_DATA_DIR)):
        path = os.path.join(EPT_DATA_DIR, name)
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            if sys.version_info < (3,):
                f = open(fpath)
            else:
                f = open(fpath, encoding='latin-1')
            texts.append(f.read())
            f.close()
            labels.append(label_id)
    print('Found %s texts in the EPT dataset.' % len(texts))



    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # word_index will contain the dictionary
    # keys are all the tokens appeared in the texts file
    # values are the corresponding indices

    word_index = tokenizer.word_index
    print('Found %s unique tokens in the EPT dataset.' % len(word_index))

    # data contains text in the form of tokenized sequences
    # labels contains the corresonding label for the ept dataset
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = np.asarray(labels)
    labels = to_categorical(np.asarray(labels))

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]


    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    np.save('X_ept.npy', data)
    np.save('y_ept.npy', labels)

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)





# Preparing the kaggle dataset
# Validation set doesnt have scores
    essay_id = []
    essay_set = []
    essay = []
    print('Preparaing the Kaggle dataset: Valiadtion set')
    with open('/home/yanan/Desktop/AES/dataset/Kaggle/valid_set.tsv','rbU') as f:
        reader = csv.reader(f, delimiter = "\t")
        reader.next()     # get ride of headers
        for row in reader:
            essay_id.append(row[0])
            essay_set.append(row[1])
            essay.append(row[2])
    print('Found %i essays in the Kaggle Validation set.' % len(essay))
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(essay)
    sequences = tokenizer.texts_to_sequences(essay)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    np.save('X_kaggle_val.npy', data)


    essay_id = []
    essay_set = []
    essay = []
    score = []
    print('Preparaing the Kaggle dataset: Dev set')
    with open('/home/yanan/Desktop/AES/dataset/Kaggle/training_set_rel3.tsv','rbU') as f:
        reader = csv.reader(f, delimiter = "\t")
        reader.next()     # get ride of headers
        for row in reader:
            essay_id.append(int(row[0]))
            essay_set.append(int(row[1]))
            essay.append(row[2])
            if row[1] == '2':
                score.append(int(row[9])+int(row[6]))
            else:
                score.append(int(row[6]))
    print('Found %i essays in the Kaggle Validation set.' % len(essay))
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(essay)
    sequences = tokenizer.texts_to_sequences(essay)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    score_list = [[],[],[],[],[],[],[],[]]
    for id in range(8):
        for i in range(len(score)):
            if essay_set[i] == id+1:
                score_list[id].append(score[i])

    labels = [[],[],[],[],[],[],[],[]]
    for id in range(8):
        labels[id] = score_binning(score_list[id])

# Contatenate the labels
    i = 0
    while i < len(labels)-1:
        labels[i+1] = np.concatenate((labels[i], labels[i+1]), axis=0)
        i=i+1

    label = labels[7]



    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    label = label[indices]
    label = to_categorical(label)
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    np.save('X_kaggle_dev.npy', data)
    np.save('y_kaggle_dev.npy', label)

    x_train = data[:-nb_validation_samples]
    y_train = label[:-nb_validation_samples]


    x_val = data[-nb_validation_samples:]
    y_val = label[-nb_validation_samples:]


def score_binning(score_list):
    zscores = stats.zscore(score_list)
    bins = [np.percentile(zscores, i) for i in [25,50,75]]
    labels = np.digitize(zscores, bins)
    return labels
