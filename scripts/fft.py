import numpy as np

from scipy.fftpack import fft, fftfreq
import math

def get_spectrum(sig, Fs, lowFreq = 1, highFreq = 35):
    N=len(sig)
    T=1.0/Fs
    sig = sig - np.mean(sig)
    #Для лучшей скорости расчета спектр будем рассчитывать по массиву, длина которого равна ближайшей степени двойки. Лишние отсчеты забьются нулями
    nextPowerOfTwo = int(math.ceil(math.log(N,2)))
    NFFT = int(math.pow(2,nextPowerOfTwo))
    #Преобразование Фурье
    yf = fft(sig, NFFT)
    yf = 2.0/N*yf[1:NFFT//2]
    #Берем модуль (абсолютное значение) преобразования
    yfAbs = np.abs(yf)
    #Рассчитываем частоты
    freqs = fftfreq(NFFT, T)
    freqs = freqs[1:NFFT//2]
    #Выбираем из спектра те значения, которые соответствуют интересующему нас диапазону частот
    idx = np.where((freqs > lowFreq) * (freqs < highFreq))
    return yfAbs[idx], freqs[idx]


def fft_for_sample(sample_calss, freq, lowFreq = 1, highFreq = 35):
    sample_class_fft = []

    for sample in sample_calss:
        new_sample = 0
        for idx, s in enumerate(range(sample.shape[0])):
            if idx == 0:
                new_sample = get_spectrum(sample[s],freq, lowFreq=lowFreq, highFreq=highFreq)[0]
            else:
                new_sample = np.vstack((new_sample, get_spectrum(sample[s],freq, lowFreq=lowFreq, highFreq=highFreq)[0]))

        sample_class_fft.append(new_sample)

    return sample_class_fft
