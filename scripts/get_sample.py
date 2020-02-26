def get_sample(matrix, sample_size=1500):
    sample = []
    step = 50
    sample_size = 150

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
