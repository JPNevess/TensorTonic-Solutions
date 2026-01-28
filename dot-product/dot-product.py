import numpy as np

def dot_product(x, y):
    xa = np.array(x)
    ya = np.array(y)
    if xa.shape!= ya.shape:
        raise ValueError("Input arrays must have the same lengt")
    elif xa.ndim!=1 or xa.ndim!=1:
        raise ValueError("Input needs to be 1D")
    s = np.dot(xa,ya)
    return float(s)