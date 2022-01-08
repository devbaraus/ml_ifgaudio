import numpy as np


def mag2db(value):
    return 20 * np.log10(value)


def db2mag(value):
    return 10 ** (value / 20)


def rms(value):
    return np.sqrt(np.mean(value ** 2))
