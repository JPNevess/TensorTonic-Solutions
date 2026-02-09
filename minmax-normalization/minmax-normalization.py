def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    xa = np.array(X)
    m = np.shape(xa)
    mi = xa.min(axis=axis, keepdims=True)
    ma = xa.max(axis=axis, keepdims=True)
    diff = ma - mi  
    diff = np.where(diff == 0, eps, diff)
    return (xa-mi)/diff