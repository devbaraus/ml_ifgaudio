import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.io import wavfile
from deep_audio import Math

base_folder = 'test_audios'

def plot_audio(signal, fs, title='', filename='', save_audio=True, show=False):
    time = np.linspace(0, len(signal) / fs, num=len(signal))
    plt.plot(time,signal)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('Segundos')
    plt.title(title)
    plt.savefig(f'{base_folder}/{filename}.png')
    if save_audio:
        wavfile.write(f'{base_folder}/{filename}.wav', fs, signal)
    if show:
        plt.show()
    plt.close()

def plot_audio_snr(signal, fs, noise, title='', filename='', save_audio=True, show=False):
    time = np.linspace(0, len(signal) / fs, num=len(signal))
    plt.plot(time,signal)
    plt.plot(time,noise)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('Segundos')
    plt.title(title)
    plt.savefig(f'{base_folder}/{filename}.png')
    if save_audio:
        wavfile.write(f'{base_folder}/{filename}.wav', fs, signal + noise)
    if show:
        plt.show()
    plt.close()

# ORIGINAL
signal, fs = librosa.load('./base_portuguese/12/p834d7ae3f5a64a37b9990fc7a8622cc1_s00_a01.wav', sr=24000)
plot_audio(signal, fs, 'Áudio Original', 'original')

# TRIM
signal, fs = librosa.load('../ifgaudio_tcc/representacoes/teste_4/base_portuguese_trim/12/p834d7ae3f5a64a37b9990fc7a8622cc1_s00_a01.wav', sr=24000)
plot_audio(signal, fs, 'Áudio Trim', 'trim')

for pot in list(range(10, 101, 10)):
    noise = pot / 100
    SNR = Math.mag2db(1/noise)
    potSinal = Math.rms(signal) ** 2
    potRuido = 1 / potSinal
    ruidoAditivo = np.random.randn(len(signal)) * np.std(signal) / Math.db2mag(SNR)

    plot_audio_snr(signal, fs, ruidoAditivo, f'Áudio Ruido {pot}%', f'noise_{pot}')