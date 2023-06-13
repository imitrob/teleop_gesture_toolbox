import sys; sys.path.append("../.."); import settings; settings.init()
import numpy as np
from os_and_utils import visualizer_lib

data = np.vstack([np.tile(np.linspace(0,1,10),(3,1)),np.ones((3,10))*0.1])

visualizer_lib.ScenePlot.my_plot([], [data], leap=False, promp_paths_variances=[data], legend=['Series 1', 'Series 2'])

'''
Parameters:
     promp_paths_variances (Float[n x 6]): n path points, 6 = (x,y,z,var_x,var_y,var_z)
'''
 for i in range(len(path[:,0])):
  if 0 > i-1: continue # skip first index
  if len(path[:,0]) <= i+1: continue # skip last index
  # vector now, subtract indices path[i + 1], path[i]
  v2 = np.array([path[i+1,0]-path[i,0], path[i+1,1]-path[i,1], path[i+1,2]-path[i,2]])
  # vector before, subtract indices path[i], path[i - 1]
  v1 = np.array([path[i,0]-path[i-1,0], path[i,1]-path[i-1,1], path[i,2]-path[i-1,2]])

  direction_path_vector = v1 + v2 # direction used is mean from v1 and v2
  variances = path[i,3:6] # pick variances

  ellipse_dim = ScenePlot.get_variabilities_dimensions(variances, direction_vector)
  patch = Ellipse((0,0), ellipse_dim[0], height=ellipse_dim[1], facecolor = 'b', alpha = .2)
  ax.add_patch(patch)

  normal_vector = direction_path_vector # direction path vector is normal vector of elipsoid cut
  pathpatch_2d_to_3d(patch, z = 0, normal = normal_vector)
  pathpatch_translate(patch, (path[i,0], path[i,1], path[i,2]))

def get_variabilities_dimensions(point_variability, path_normal):
    ''' Returns: Elipsoid and plance cut x,y radius
    '''
    def get_radius_elipse_with_normal(a,b, vx,vy):
        ''' Radius from direction vector, input is normal vector
        Parameters:
            a,b (Float): elipse dimensions
            normal (Float): normal vector
        Returns:
            r (Float): radius
        '''
        theta = np.arctan2(vy,vx)
        theta_dir = theta + np.pi/2

        e = np.sqrt(1-(b**2/a**2))
        r = a * np.sqrt(1-e**2 * (np.sin(theta_dir)**2))
        return r
    x = get_radius_elipse_with_normal(point_variability[0], point_variability[1], path_normal[0], path_normal[1])
    y = get_radius_elipse_with_normal(point_variability[0], point_variability[2], path_normal[0], path_normal[2])

    return x,y
