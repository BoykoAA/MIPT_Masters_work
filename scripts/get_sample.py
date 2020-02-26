def get_sample(matrix, sample_size=550, step=200):
    sample = []

    beg_sample = 0
    end_sample = 0

    for idx, i in (enumerate(range(0, matrix.shape[1], step))):

        if idx == 0:
            sample.append(matrix[:, beg_sample:sample_size])
            beg_sample += step
            end_sample = sample_size + step
        else:
            sample.append(matrix[:, beg_sample:end_sample])
            beg_sample += step
            end_sample = beg_sample + sample_size

    return sample


def create_strings_for_dataset(samples_fft):
    strings = []
    for i in range(len(samples_fft)):
        new_string = []
        for n in range(samples_fft[i].shape[1]):
            new_string.extend(samples_fft[i][:,n])
        strings.append(new_string)

    return strings
