import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE

    """
    y1 = np.array(y_pred)
    y2 = np.array(y_true)
    m = np.shape(y1)
    n = np.shape(y2)
    if m != n:
        return None
    e = 0

    for i in range(m[0]):
        e += (y1[i]-y2[i])**2
    e = e/m
    return e[0]
