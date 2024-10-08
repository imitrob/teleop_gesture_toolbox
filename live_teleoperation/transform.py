import numpy as np

def transform_leap_to_scene(data, scale=1.0, start=[0.5, 0.0, 0.2]):
    x, y, z = data[0], data[1], data[2]
    
    x_ =  x/1000
    y_ = -z/1000
    z_ =  y/1000

    x__ = np.dot([x_,y_,z_], [0,-1, 0])*scale + start[0]
    y__ = np.dot([x_,y_,z_], [1, 0, 0])*scale + start[1]
    z__ = np.dot([x_,y_,z_], [0, 0, 1])*scale + start[2]

    data[0] = x__
    data[1] = y__
    data[2] = z__

    return data