import numpy as np

def matrix_gen(size=(128, 20000), classes=3):

    '''
    The matrix generated in this way simulates an EEG signal,
    with standard parameters 40 seconds of the signal will be mapped.
    '''

    s, c = size
    matrix = np.zeros((s, 0))

    for cl in range(classes):
        one_class_size = c // classes
        m = np.random.uniform(10**-(cl+1), 20**-(cl+1), (s, one_class_size))
        matrix = np.hstack((matrix, m))

    return matrix

def get_ICA(size=(128, 128)):

    '''
    Generates an ICA matrix
    consisting of random values
    '''

    s, c = size
    ICA = np.random.uniform(10**(-1), 20**(-1), (s,c))
    return ICA
