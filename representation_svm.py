import argparse
import operator

import librosa.display
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from deep_audio import Math

# %%
parser = argparse.ArgumentParser(description='Arguments algo')
parser.add_argument('-f', action='store', dest='feature', required=False, help='Extrator de características',
                    default='lpc')
parser.add_argument('-c', nargs='+', type=int, action='store', dest='coeffs', required=False, help='Coeficientes',
                    default=[512])
parser.add_argument('-a', nargs='+', type=int, action='store', dest='augmentation', required=False, help='Augmentation',
                    default=[1, 2, 5, 10, 15])

args = parser.parse_args()

params = {
    # 'segment_time': list(range(1, 5)),
    'segment_time': [2],
    # 'noise': np.arange(0, 1.01, 0.5).tolist(),
    'noise': [0],
    # 'coeffs': args.coeffs,
    'coeffs': [1024, 2048],
    'augmentation': [1, 5, 10],
    # 'augmentation': args.augmentation,
    # 'feature': args.feature
    'feature': 'lpc'
}

print(params)

datatable = {
    'feature': [],
    'segment_time': [],
    'noise_percentage': [],
    'feature_coeff': [],
    'augmentation': [],
    'f1_micro': [],
    'f1_macro': []
}

fs = 24000

if False:
    segment = 2
    coeff = 512
    augmentation = 2
    noise = 0

for segment in params['segment_time']:
    mat_audios = sio.loadmat(f'datasets/ifgaudioData{segment}seg.mat')
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

            if noise != 0:
                SNR = Math.mag2db(noise)
                potSinal = Math.rms(sinal) ** 2
                potRuido = 1 / potSinal
                ruidoAditivo = np.random.randn(numAmostras) * np.std(sinal) / Math.db2mag(SNR)
            else:
                ruidoAditivo = np.zeros(numAmostras)

            for audio_segment in range(X_train_aug.shape[0]):
                X_train_aug[audio_segment] = X_train_aug[audio_segment] + ruidoAditivo

            X_train_aug = np.concatenate((X_train, X_train_aug), axis=0)
            y_train_aug = np.concatenate((y_train, y_train_aug), axis=0)

            for coeff in params['coeffs']:
                print(f'[{params["feature"].upper()}] ', coeff)
                X_train_rep = None
                X_test_rep = None

                if params['feature'] == 'mfcc':
                    rep_size = librosa.feature.mfcc(X_train_aug[0], fs, n_mfcc=coeff)
                    X_train_rep = np.zeros((X_train_aug.shape[0], rep_size.size))
                    X_test_rep = np.zeros((X_test.shape[0], rep_size.size))

                    for i in range(X_train_aug.shape[0]):
                        rep = librosa.feature.mfcc(X_train_aug[i], fs, n_mfcc=coeff)
                        X_train_rep[i] = rep.flatten()

                    for i in range(X_test.shape[0]):
                        rep = librosa.feature.mfcc(X_test[i], fs, n_mfcc=coeff)
                        X_test_rep[i] = rep.flatten()

                if params['feature'] == 'stft':
                    rep_size = np.abs(librosa.stft(X_train_aug[0], n_fft=coeff))
                    X_train_rep = np.zeros((X_train_aug.shape[0], rep_size.size))
                    X_test_rep = np.zeros((X_test.shape[0], rep_size.size))

                    for i in range(X_train_aug.shape[0]):
                        rep = np.abs(librosa.stft(X_train_aug[i], n_fft=coeff))
                        X_train_rep[i] = rep.flatten()

                    for i in range(X_test.shape[0]):
                        rep = np.abs(librosa.stft(X_test[i], n_fft=coeff))
                        X_test_rep[i] = rep.flatten()

                if params['feature'] == 'lpc':
                    rep_size = librosa.lpc(X_train_aug[0], order=coeff)
                    X_train_rep = np.zeros((X_train_aug.shape[0], rep_size.size))
                    X_test_rep = np.zeros((X_test.shape[0], rep_size.size))

                    for i in range(X_train_aug.shape[0]):
                        rep = librosa.lpc(X_train_aug[i], order=coeff)
                        X_train_rep[i] = rep.flatten()

                    for i in range(X_test.shape[0]):
                        rep = librosa.lpc(X_test[i], order=coeff)
                        X_test_rep[i] = rep.flatten()

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
                clf = SVC(kernel='linear', C=10)
                scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'f1_micro']
                scores = cross_validate(clf, X_train_rep, y_train_rep,
                                        scoring=scoring, cv=5,
                                        return_estimator=True,
                                        return_train_score=True)

                # PREDICT
                print('[PREDICT] starting')
                idx_estimator, _ = max(enumerate(scores['test_f1_micro']), key=operator.itemgetter(1))
                estimator = scores['estimator'][idx_estimator]
                y_hat = estimator.predict(X_test_rep)
                fmacro = f1_score(y_test_rep, y_hat, average='macro')
                fmicro = f1_score(y_test_rep, y_hat, average='micro')

                print('[DATA] saving')
                datatable['feature'].append(params['feature'])
                datatable['feature_coeff'].append(coeff)
                datatable['segment_time'].append(segment)
                datatable['noise_percentage'].append(noise)
                datatable['augmentation'].append(augmentation)
                datatable['f1_micro'].append(fmicro)
                datatable['f1_macro'].append(fmacro)

df = pd.DataFrame(datatable)
df.to_csv(f'{params["feature"]}_svm.csv')
