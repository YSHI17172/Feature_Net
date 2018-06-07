import numpy as np


def normalized(a, axis=-1, order=2):
    """ Function to normalize numpy eigenvector arrays (source StackOverflow)
    :param a: non-normalized eigenvector array
    :param axis: axis of a along which to compute the vector norms
    :param order: order of the norm ,see linalg.norm documentation
    :return: normalized a eigenvector array
    """

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
