import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len == None:
        max_len = 0
        for i in range(len(seqs)):
            if len(seqs[i])>max_len:
                max_len = len(seqs[i])
        for i in seqs:
            if len(i)!=max_len:
                for c in range(max_len-len(i)):
                    i.append(pad_value)         
    else:
        for i in seqs:
            if len(i)<max_len:
                for c in range(max_len-len(i)):
                    i.append(pad_value)
            elif len(i) >max_len:
                for c in range(len(i)-max_len):
                    i.pop()
    seqs_np = np.array(seqs)
    return seqs_np