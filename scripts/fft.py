import numpy as np

def fft_for_sample(sample_calss):

    sample_class_fft = []

    for sample in sample_calss:
        new_sample = 0
        for idx, s in enumerate(range(sample.shape[0])):
            if idx == 0:
                new_sample = np.fft.fft(sample[s])
            else:
                new_sample = np.vstack((new_sample, np.fft.fft(sample[s])))

        sample_class_fft.append(new_sample)

    return sample_class_fft
