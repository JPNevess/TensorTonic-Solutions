import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    linhas = len(A)
    colunas = len(A[0])
    a_t = []
    for i in range(colunas):
        b = []
        for c in range(linhas):
            b.append(A[c][i])
        a_t.append(b)
    at_np = np.array(a_t)   
    return at_np
