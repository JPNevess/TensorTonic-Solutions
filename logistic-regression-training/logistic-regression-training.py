import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    for i in range(steps):
        linear_model = np.dot(X, w) + b
        p = _sigmoid(linear_model)
        error = p - y
        
        dw = (1 / N) * np.dot(X.T, error)
        db = (1 / N) * np.sum(error)
        w -= lr * dw
        b -= lr * db
    return (w,b)
       
