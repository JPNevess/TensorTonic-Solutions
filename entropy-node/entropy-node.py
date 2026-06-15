import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    list = []
    for i in y:
        if i not in list:
            list.append(i)
    dic = {}
    for classe in list:
        dic[classe] = y.count(classe)
    p = []
    for i in dic:
        p.append(dic[i]/len(y))
    H = 0
    for i in p:
        H += float(i*np.log2(i))
    return float(-H)