import numpy as np

def transform_leap_to_scene(data, env, scale):
    x, y, z = data
    
    x = x/1000
    y = -z/1000
    z = y/1000

    x_ = np.dot([x,y,z], [0,-1, 0])*scale + env['start'].x
    y_ = np.dot([x,y,z], [1, 0, 0])*scale + env['start'].y
    z_ = np.dot([x,y,z], [0, 0, 1])*scale + env['start'].z

    return [x_, y_, z_]