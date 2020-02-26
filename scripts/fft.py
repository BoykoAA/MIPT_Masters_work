import numpy as np

def fft_for_sample(sample_calss, first_n_elements=None):
    sample_class_fft = []

    for sample in sample_calss:
        new_sample = 0
        for idx, s in enumerate(range(sample.shape[0])):
            if idx == 0:
                new_sample = np.fft.fft(sample[s])[:first_n_elements]
            else:
                new_sample = np.vstack((new_sample, np.fft.fft(sample[s])[:first_n_elements]))

        sample_class_fft.append(new_sample)

    return sample_class_fft
