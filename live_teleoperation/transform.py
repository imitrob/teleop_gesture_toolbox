import numpy as np

def transform_leap_to_scene(data, scale=1.0, start=[0.5, 0.0, 0.4]):
    x, y, z = data
    
    x =  x/1000
    y = -z/1000
    z =  y/1000

    x_ = np.dot([x,y,z], [0,-1, 0])*scale + start[0]
    y_ = np.dot([x,y,z], [1, 0, 0])*scale + start[1]
    z_ = np.dot([x,y,z], [0, 0, 1])*scale + start[2]

    return [x_, y_, z_]