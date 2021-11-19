import numpy as np
import torch

# Remove selected entries in an array
# E.g. compress_array(np.array([i*np.ones((2, 2)) for i in range(5)]), np.array([True, False, True, False, False])) => array([[[0., 0.], [0., 0.]], [[2., 2.], [2., 2.]]])
def compress_array(x, mask):
    return x[mask, ...]

# Coverts an input 2D numpy array into a 3D one-hot matrix (only one entry True along axis=0) 
# The one-hot matrix is created only using the values in labels array
# E.g. matrix_one_hot(np.array([[3]]), np.array([1, 2, 3])) => array([[[False]], [[False]], [[True]]])
def matrix_one_hot(x, labels):
    return (x == labels[:, None, None])

# Removes duplicates in an array while excluding certain entries
# E.g. remove_duplicate_entries(np.array([1, 1, 3, 'a', 'b']), exclude_entries=['a']) => np.array(['1', '3', 'b'])
def remove_duplicate_entries(x, exclude_entries=None):
    vals = np.sort(np.unique(x))
    if exclude_entries is None:
        return vals
    else:
        mask = np.in1d(vals, exclude_entries)
        return vals[~mask]

# Returns the bottom-left and top-right corners of the bounding box of a 2D boolean-valued input array 
# E.g. bounding_box(np.ones((4,4))) => ((0, 0), (3, 3))
def bounding_box(x):
    pos = np.where(x)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return (xmin, ymin), (xmax, ymax)

x = bounding_box(np.ones((4,4)))
print(np.array(x))
print(torch.tensor(np.array(x)).flip(0))