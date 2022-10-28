
import numpy as np

''' The Leap Motion base to Robot base '''
def calibrate_leap_motion():
    ''' Calibration procedure
    Returns:
        transformation matrix 4x4
    - Note: Measured by ruler
    '''
    return np.array([[1, 0, 0, 1.07],
                     [0, 1, 0,-0.40],
                     [0, 0, 1, 0.01],
                     [0, 0, 0, 1   ]])

def leap_motion_to_world_link(p):
    T = calibrate_leap_motion()
    # TMP
    return [T[0,3], T[1,3], T[2,3]]

def get_closest_point_to_line(line_points, test_point):
    ''' Line points
    '''
    p1, p2 = line_points
    p3 = test_point

    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    dx, dy, dz = x2-x1, y2-y1, z2-z1
    det = dx*dx + dy*dy + dz*dz
    a = (dy*(y3-y1)+dx*(x3-x1)+dz*(z3-z1))/det
    return x1+a*dx, y1+a*dy, z1+a*dz

def get_id_of_closest_point_to_line(line_points, test_points, max_dist=np.inf):
    '''
    Embedding: line_points are start and end of last bone (distal) from pointing finger (2nd finger)
    test_points:
    '''
    distances_from_line = []
    for test_point in test_points:
        closest_point = get_closest_point_to_line(line_points, test_point)
        norm_distance = np.linalg.norm(np.array(closest_point)-np.array(test_point))
        distances_from_line.append(norm_distance)

    if np.min(distances_from_line) > max_dist: return None, np.min(distances_from_line)
    return np.argmin(distances_from_line), np.min(distances_from_line)

if __name__ == '__main__':
    # test 1
    line_points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    test_points = ((0.0, 0.0, 1.00000), (10.0, 0.0, 1.0), (10.0, 0.0, 2.0))
    closest_point = get_id_of_closest_point_to_line(line_points, test_points)
    closest_point

    # test 2
    line_points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    test_points = ((0.0, 0.0, 0.99999), (10.0, 0.0, 1.0), (10.0, 0.0, 2.0))
    closest_point = get_id_of_closest_point_to_line(line_points, test_points)
    closest_point

    # test 3
    line_points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    test_points = ((0.0, 0.0, 0.99999), (10.0, 0.0, 1.0), (10.0, 0.0, 2.0))
    closest_point = get_id_of_closest_point_to_line(line_points, test_points, max_dist=0.3)
    closest_point
