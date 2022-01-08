import argparse
import multiprocessing
import operator
import os

import librosa.display
import numpy as np
from numpy.core.fromnumeric import repeat
import pandas as pd
import scipy.io as sio
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scikeras.wrappers import KerasClassifier
import tensorflow.keras as keras
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report


from deep_audio import Math
import time

# %%
parser = argparse.ArgumentParser(description='Arguments algo')
parser.add_argument('-f', action='store', dest='feature', required=False, help='Extrator de características',
                    default='lpc')
parser.add_argument('-c', nargs='+', type=int, action='store', dest='coeffs', required=False, help='Coeficientes',
                    default=[512])
parser.add_argument('-a', nargs='+', type=int, action='store', dest='augmentation', required=False, help='Augmentation',
                    default=[10])
parser.add_argument('-n', nargs='+', type=float, action='store', dest='noise', required=False, help='Noise',
                    default=np.arange(0, 1.01, 0.25).tolist())
parser.add_argument('-s', nargs='+', type=int, action='store', dest='segment', required=False, help='Segment ime',
                    default=list(range(1, 5)))
parser.add_argument('-t', type=bool, action='store', dest='trim', required=False, help='Use trim dataset',
                    default=False)
parser.add_argument('-m', type=str, action='store', dest='model', required=False, help='Model algorithm',
                    default='svm')

args = parser.parse_args()
now = int(time.time())


def output_dir():
    t = ''
    if args.trim:
        t = '_trim'
    if args.noise != [0]:
        t += '_noise'
    if args.augmentation != [0]:
        t += f'_aug{args.augmentation[0]}'

    return t


def build_perceptron(output, shape, dense1=512, dense2=256, dense3=128, learning_rate=0.0001):
    model = keras.Sequential()

    model.add(keras.layers.Flatten(
        input_shape=shape[1:]))

    model.add(keras.layers.Dense(dense1, activation='relu'))
    model.add(keras.layers.Dense(dense2, activation='relu'))
    model.add(keras.layers.Dense(dense3, activation='relu'))
    model.add(keras.layers.Dense(output, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_cnn(output, shape, resizing=(32, 32), conv2d1=32, conv2d2=64, dropout1=0.25, dropout2=0.5, dense=128, learning_rate=0.0001):
    from tensorflow.keras.layers.experimental import preprocessing

    input_shape = shape[1:]

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(preprocessing.Resizing(resizing[0], resizing[1]))
    model.add(keras.layers.Conv2D(conv2d1, 3, activation='relu'))
    model.add(keras.layers.Conv2D(conv2d2, 3, activation='relu'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(dropout1))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(dense, activation='relu'))
    model.add(keras.layers.Dropout(dropout2))
    model.add(keras.layers.Dense(output,  activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_lstm(output, shape, lstm=64, dropout1=0.2, dense1=64, dense2=32, dropout2=0.4, dense3=24, dropout3=0.4, learning_rate=0.0001):

    input_shape = shape[1:]

    model = keras.Sequential()
    model.add(keras.layers.LSTM(lstm, input_shape=input_shape))
    model.add(keras.layers.Dropout(dropout1))
    model.add(keras.layers.Dense(dense1, activation='relu'))
    model.add(keras.layers.Dense(dense2, activation='relu'))
    model.add(keras.layers.Dropout(dropout2))
    model.add(keras.layers.Dense(dense3, activation='relu'))
    model.add(keras.layers.Dropout(dropout3))
    model.add(keras.layers.Dense(output, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


params = {
    'segment_time': args.segment,
    'noise': args.noise,
    'coeffs': args.coeffs,
    'augmentation': args.augmentation,
    'feature': args.feature,
    'trim': args.trim,
    'model': args.model,
    'output_file': f'{args.feature}_{args.model}_{now}.csv',
    'output_dir': f'results/{args.model}/{args.feature}/{output_dir()}'
}


datatable = {
    'feature': [],
    'segment_time': [],
    'noise_percentage': [],
    'feature_coeff': [],
    'augmentation': [],
    'f1_micro': [],
    'f1_macro': [],
    'representation_size': [],
    'train_size': [],
    'test_size': [],
    'algorithm': []
}

fs = 24000

if False:
    segment = 2
    coeff = 512
    augmentation = 2
    noise = 0

if __name__ == '__main__':
    print(params)

    for segment in params['segment_time']:
        print('[SEG] ', segment)
        folder_suffix = '_trim' if args.trim == True else ''
        folder_name = f'datasets/ifgaudioData{segment}seg{folder_suffix}.mat'
        mat_audios = sio.loadmat(folder_name)
        mat_audios = mat_audios['ifgaudioData']

        data_audio = mat_audios['Data'][0, 0]
        labels_audio = mat_audios['Labels'][0, 0]

        # remove row that only contains zero
        rows = np.all(data_audio == 0, axis=1)
        data_audio = data_audio[~rows]
        labels_audio = labels_audio[~rows]

        labels = []

        for label in labels_audio[:, 0]:
            labels.append(label[0, 0])

        labels_audio = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(data_audio,
                                                            labels_audio,
                                                            stratify=labels_audio,
                                                            test_size=0.2,
                                                            random_state=42)

        for augmentation in params['augmentation']:
            print('[AUG] ', augmentation)
            X_train_aug = np.array(X_train.tolist() * augmentation)
            y_train_aug = np.array(y_train.tolist() * augmentation)

            for noise in params['noise']:
                print('[NOISE] ', noise)
                sinal = X_train[0, :]
                numAmostras = sinal.size

                if augmentation != 0:
                    if noise != 0:
                        SNR = Math.mag2db(1/noise)
                        potSinal = Math.rms(sinal) ** 2
                        potRuido = 1 / potSinal
                        ruidoAditivo = np.random.randn(
                            numAmostras) * np.std(sinal) / Math.db2mag(SNR)
                    else:
                        ruidoAditivo = np.zeros(numAmostras)

                    for audio_segment in range(X_train_aug.shape[0]):
                        X_train_aug[audio_segment] = X_train_aug[audio_segment] + ruidoAditivo

                    X_train_aug = np.concatenate(
                        (X_train, X_train_aug), axis=0)
                    y_train_aug = np.concatenate(
                        (y_train, y_train_aug), axis=0)
                else:
                    X_train_aug = X_train
                    y_train_aug = y_train

                for coeff in params['coeffs']:
                    print(f'[{params["feature"].upper()}] ', coeff)
                    X_train_rep = None
                    X_test_rep = None
                    rep_shape = None

                    if params['feature'] == 'mfcc':
                        rep_size = librosa.feature.mfcc(
                            X_train_aug[0], fs, n_mfcc=coeff)
                        rep_shape = rep_size.shape

                        X_train_rep = np.zeros(
                            (X_train_aug.shape[0], rep_shape[0], rep_shape[1]))
                        X_test_rep = np.zeros(
                            (X_test.shape[0], rep_shape[0], rep_shape[1]))

                        for i in range(X_train_aug.shape[0]):
                            rep = librosa.feature.mfcc(
                                X_train_aug[i], fs, n_mfcc=coeff)
                            X_train_rep[i] = rep

                        for i in range(X_test.shape[0]):
                            rep = librosa.feature.mfcc(
                                X_test[i], fs, n_mfcc=coeff)
                            X_test_rep[i] = rep

                    if params['feature'] == 'stft':
                        rep_size = np.abs(librosa.stft(
                            X_train_aug[0], n_fft=coeff))
                        rep_shape = rep_size.shape
                        X_train_rep = np.zeros(
                            (X_train_aug.shape[0], rep_shape[0], rep_shape[1]))
                        X_test_rep = np.zeros(
                            (X_test.shape[0], rep_shape[0], rep_shape[1]))

                        for i in range(X_train_aug.shape[0]):
                            rep = np.abs(librosa.stft(
                                X_train_aug[i], n_fft=coeff))
                            X_train_rep[i] = rep

                        for i in range(X_test.shape[0]):
                            rep = np.abs(librosa.stft(X_test[i], n_fft=coeff))
                            X_test_rep[i] = rep

                    if params['feature'] == 'lpc':
                        rep_size = librosa.lpc(X_train_aug[0], order=coeff)
                        rep_shape = rep_size.shape
                        X_train_rep = np.zeros(
                            (X_train_aug.shape[0], rep_size.size))
                        X_test_rep = np.zeros((X_test.shape[0], rep_size.size))

                        for i in range(X_train_aug.shape[0]):
                            rep = librosa.lpc(X_train_aug[i], order=coeff)
                            X_train_rep[i] = rep

                        for i in range(X_test.shape[0]):
                            rep = librosa.lpc(X_test[i], order=coeff)
                            X_test_rep[i] = rep

                    if params['feature'] == 'fft':
                        rep_size = np.abs(np.fft.fft(X_train_aug[0], n=coeff))
                        rep_shape = rep_size.shape
                        X_train_rep = np.zeros(
                            (X_train_aug.shape[0], rep_shape[0]))
                        X_test_rep = np.zeros((X_test.shape[0], rep_shape[0]))

                        for i in range(X_train_aug.shape[0]):
                            rep = np.abs(np.fft.fft(X_train_aug[i], n=coeff))
                            X_train_rep[i] = rep

                        for i in range(X_test.shape[0]):
                            rep = np.abs(np.fft.fft(X_test[i], n=coeff))
                            X_test_rep[i] = rep

                    if params['model'] == 'cnn':
                        X_train_rep = X_train_rep[..., np.newaxis]
                        X_test_rep = X_test_rep[..., np.newaxis]

                    se = StandardScaler()
                    le = LabelEncoder()

                    X_train_rep = se.fit_transform(
                        X_train_rep.reshape(-1, X_train_rep.shape[-1])).reshape(X_train_rep.shape)
                    X_test_rep = se.transform(
                        X_test_rep.reshape(-1, X_test_rep.shape[-1])).reshape(X_test_rep.shape)

                    y_train_rep = le.fit_transform(y_train_aug)
                    y_test_rep = le.transform(y_test)

                    # TRAIN
                    print('[TRAIN] starting')

                    unique_labels = len(np.unique(y_train_rep).tolist())

                    if params['model'] == 'mlp':
                        # clf = build_perceptron(unique_labels, X_train_rep.shape)
                        clf = KerasClassifier(model=build_perceptron, output=unique_labels,
                                              shape=X_train_rep.shape, epochs=2000, batch_size=128, verbose=0)
                    if params['model'] == 'cnn':
                        # clf = build_cnn(unique_labels, X_train_rep.shape)
                        clf = KerasClassifier(model=build_cnn, output=unique_labels,
                                              shape=X_train_rep.shape, epochs=2000, batch_size=128, verbose=0)
                    if params['model'] == 'lstm':
                        # clf = build_lstm(unique_labels, X_train_rep.shape)
                        clf = KerasClassifier(model=build_lstm, output=unique_labels,
                                              shape=X_train_rep.shape, epochs=2000, batch_size=128, verbose=0)

                    scoring = ['precision_macro',
                               'recall_macro', 'f1_macro', 'f1_micro']
                    scores = cross_validate(clf, X_train_rep, y_train_rep,
                                            scoring=scoring, cv=3,
                                            return_estimator=True,
                                            return_train_score=True)

                    # for train_index, valid_index in strat_kfold.split(X_train_rep, y_train_rep):
                    #     X_train_fold, X_valid_fold = X_train_rep[train_index], X_train_rep[valid_index]
                    #     y_train_fold, y_valid_fold = y_train_rep[train_index], y_train_rep[valid_index]

                    #     clf.fit(X_train_fold, y_train_fold, validation_data=(X_valid_fold, y_valid_fold), epochs=2000, batch_size=128, verbose=3)
                    #     y_pred = clf.predict(X_valid_fold)
                    #     print(classification_report(y_valid_fold, y_pred))

                    print(scores)
                    # PREDICT
                    print('[PREDICT] starting')
                    idx_estimator, _ = max(
                        enumerate(scores['test_f1_micro']), key=operator.itemgetter(1))
                    estimator = scores['estimator'][idx_estimator]
                    y_hat = estimator.predict(X_test_rep)
                    print('[PREDICT] done')

                    fmacro = f1_score(
                        y_test_rep, y_hat, average='macro', labels=np.unique(y_test_rep))
                    fmicro = f1_score(
                        y_test_rep, y_hat, average='micro', labels=np.unique(y_test_rep))

                    print('[DATA] saving')
                    datatable['feature'].append(params['feature'])
                    datatable['feature_coeff'].append(coeff)
                    datatable['segment_time'].append(segment)
                    datatable['noise_percentage'].append(noise)
                    datatable['augmentation'].append(augmentation)
                    datatable['f1_micro'].append(fmicro)
                    datatable['f1_macro'].append(fmacro)
                    datatable['train_size'].append(X_train.shape)
                    datatable['test_size'].append(X_test.shape)
                    datatable['representation_size'].append(rep_shape)
                    datatable['algorithm'].append(args.model)

                    df = pd.DataFrame(datatable)

                    if not os.path.exists(params['output_dir']):
                        os.makedirs(params['output_dir'])

                    df.to_csv(params['output_dir'] + '/' +
                              params['output_file'], index=False)

                    del X_train_rep, X_test_rep, y_train_rep, y_test_rep
