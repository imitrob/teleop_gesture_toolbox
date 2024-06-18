import numpy as np
from scipy.signal import argrelextrema

def get_local_min_and_max(arr):
    return (argrelextrema(arr, np.greater),argrelextrema(arr, np.less))

def get_median_extreme(arr):
    l = len(arr)
    gr, ls = get_local_min_and_max(arr)
    extrems = (*gr,*ls)

    min(extrems)

'''
>>> # Start with zeros, then move to true value 0.05, finally go back to zeros
>>> test_data = [0.01,0.02,0.01,0.02,0.04,0.05,0.051,0.02,0.01,0.00,0.01]
>>> # const
>>> test_data = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
>>> # Start with the right value, then move out
>>> test_data = [0.05,0.051,0.049,0.052,0.02,0.01,0.001]

>>> a = np.array([0.,0.5,1.0,1.5,2.2,2.3,2.4,2.0,1.4,1.0,0.0,0.1,0.2,0.5,0.3])
>>> x = np.array(range(len(a)))
>>> get_local_min_and_max(a)
>>> import matplotlib.pyplot as plt
>>> plt.plot(test_data)
'''
def get_dist_by_extremes(y, bufferlen=100, fit_deg=6):
    # y = test_data
    # 1. Fit the data
    x = np.array(range(len(y)))
    try:
        polyvals = np.polyfit(x, y, deg=fit_deg)
    except: #LinAlgError and SystemError:
        return y[0]
    # 2. Gen. new fitted data
    x_ = np.linspace(0,len(x),bufferlen)
    y_ = [np.polyval(polyvals, i) for i in x_]
    ''' # possible plot
    plt.plot(x_,y_)
    plt.plot(y)
    '''
    # 3. Get extremes and pick the closest extreme to center
    extremes = np.hstack(get_local_min_and_max(np.array(y_)))[0]
    if len(extremes) == 0: return np.array(y).mean()
    pt = min(extremes-np.array(bufferlen//2), key=abs)+bufferlen//2

    dist = y[round(pt/100*len(x))]
    return dist