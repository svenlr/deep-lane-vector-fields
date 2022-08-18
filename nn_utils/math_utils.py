import numpy as np


def gaussian(x, mu, sigma):
    """ evaluate gaussian function for parameter x """
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - mu) ** 2 / (sigma ** 2))


def rbf_activations(x, mus, sigmas):
    """ normalized RBF activations for all values in image. Gaussian are parameterized by the mus / sigmas arrays """
    x = np.array(x)
    x = np.expand_dims(x, -1)
    activations = gaussian(x, np.array(mus), np.array(sigmas))
    if np.all(activations == 0):
        ret = np.zeros(shape=activations.shape, dtype=np.float)
        ret[np.argmin((x - mus) * (x - mus))] = 1
        return ret
    if activations.ndim == 1:
        activations /= np.sum(activations)
    else:
        sums = np.sum(activations, axis=-1)
        sums[sums == 0] = 1
        activations /= np.expand_dims(sums, -1)
    return activations


def downsample_1d_np(values, num_points):
    values = np.interp(np.linspace(0, len(values) - 1, num=num_points, endpoint=True),
                       np.arange(len(values)), values)
    return values
