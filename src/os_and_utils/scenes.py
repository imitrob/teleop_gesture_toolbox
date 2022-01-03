import os, yaml
import settings
from os_and_utils.utils import merge_two_dicts, ordered_load
from os_and_utils.parse_yaml import ParseYAML

from geometry_msgs.msg import Vector3
import os_and_utils.move_lib as ml

class CustomPaths():
    def __init__(self):
        self.paths = []
    def __getitem__(self, item):
        return self.paths[item]
    def append(self, path):
        self.paths.append(path)
    def names(self):
        return [path.name for path in self.paths]

    @staticmethod
    def GenerateFromYAML(scenes, paths_folder=None, paths_file_catch_phrase='paths', poses_file_catch_phrase='poses'):
        ''' Generates All paths from YAML files

        Parameters:
            paths_folder (Str): folder to specify YAML files, if not specified the default paths.custom_settings_yaml folder is used
            paths_file_catch_phrase (Str): Searches for files with this substring (e.g. use 'paths' to load all names -> 'paths1.yaml', paths2.yaml)
                - If not specified then paths_file_catch_phrase='paths'
                - If specified as '', all files are loaded
            poses_file_catch_phrase (Str): Loads poses from YAML file with this substring
        Returns:
            paths (CustomPath()[]): The generated array of paths
        '''
        paths = CustomPaths()
        if not paths_folder:
            paths_folder = settings.paths.custom_settings_yaml

        files = os.listdir(paths_folder)

        poses_data_loaded = {}
        for f in files:
            if '.yaml' in f and poses_file_catch_phrase in f:
                with open(paths_folder+f, 'r') as stream:
                    poses_data_loaded = merge_two_dicts(poses_data_loaded, ordered_load(stream, yaml.SafeLoader))

        for f in files:
            if '.yaml' in f and paths_file_catch_phrase in f:
                with open(paths_folder+f, 'r') as stream:
                    data_loaded = ordered_load(stream, yaml.SafeLoader)
                    for key in data_loaded.keys():
                        pickedpath = {key: data_loaded[key]}
                        paths.append(CustomPath(scenes,pickedpath, poses_data_loaded))
        return paths


# TODO: Make NAME, ENV, Objects lowercase
class CustomPath():
    def __init__(self, scenes, data=None, poses_data=None):
        ''' Create your custom path
        '''
        assert data, "No Data!"
        assert len(data.keys()) == 1, "More input Paths!"
        key = list(data.keys())[0]
        path_data = data[key]
        self.name = key
        self.poses = []
        self.actions = []
        if not path_data:
            return

        poses = path_data['poses']
        self.n = len(path_data['poses'])
        self.scene = ParseYAML.parseScene(path_data, scenes)
        self.ENV = path_data['env']
        for pose_e in poses:
            pose = pose_e['pose']
            self.poses.append(ParseYAML.parsePose(pose, poses_data))

            self.actions.append(ParseYAML.parseAction(pose_e))


class CustomScenes():
    def __init__(self):
        self.scenes = []
    def __getitem__(self, item):
        return self.scenes[item]
    def append(self, scene):
        self.scenes.append(scene)
    def names(self):
        return [scene.name for scene in self.scenes]

    @staticmethod
    def GenerateFromYAML(scenes_folder=None, scenes_file_catch_phrase='scene', poses_file_catch_phrase='poses'):
        ''' Generates All scenes from YAML files

        Parameters:
            scenes_folder (Str): folder to specify YAML files, if not specified the default paths.custom_settings_yaml folder is used
            scenes_file_catch_phrase (Str): Searches for files with this substring (e.g. use 'scene' to load all names -> 'scene1.yaml', scene2.yaml)
                - If not specified then scenes_file_catch_phrase='scene'
                - If specified as '', all files are loaded
            poses_file_catch_phrase (Str): Loads poses from YAML file with this substring
        '''
        scenes = CustomScenes()
        if not scenes_folder:
            scenes_folder = settings.paths.custom_settings_yaml

        files = os.listdir(scenes_folder)

        poses_data_loaded = {}
        for f in files:
            if '.yaml' in f and poses_file_catch_phrase in f:
                with open(scenes_folder+f, 'r') as stream:
                    poses_data_loaded = merge_two_dicts(poses_data_loaded, ordered_load(stream, yaml.SafeLoader))

        for f in files:
            if '.yaml' in f and scenes_file_catch_phrase in f:
                with open(scenes_folder+f, 'r') as stream:
                    data_loaded = ordered_load(stream, yaml.SafeLoader)
                    for key in data_loaded.keys():
                        pickedscene = {key: data_loaded[key]}
                        scenes.append(CustomScene(pickedscene, poses_data_loaded))
        return scenes

    def make_scene(self, new_scene=''):
        ''' Prepare scene, add objects for obstacle or manipulation.
            scene (str):
        '''
        global scene
        scenes = self.names()
        if scene: # When scene is initialized
            # get id of current scene
            id = scenes.index(scene.name)
            # remove objects from current scene
            for i in range(0, len(self.scenes[id].object_names)):
                self.remove_object(name=self.scenes[id].object_names[i])
            if ml.md.attached:
                self.detach_item_moveit(name=ml.md.attached)
        # get id of new scene
        id = self.names().index(new_scene)

        for i in range(0, len(self.scenes[id].object_names)):
            obj_name = self.scenes[id].object_names[i] # object name
            size = extv(self.scenes[id].object_sizes[i])
            color = self.scenes[id].object_colors[i]
            scale = self.scenes[id].object_scales[i]
            shape = self.scenes[id].object_shapes[i]
            mass = self.scenes[id].object_masses[i]
            friction = self.scenes[id].object_frictions[i]
            inertia = self.scenes[id].object_inertia[i]
            inertiaTransformation = self.scenes[id].object_inertiaTransform[i]
            dynamic = self.scenes[id].object_dynamic[i]
            pub_info = self.scenes[id].object_pub_info[i]
            texture_file = self.scenes[id].object_texture_file[i]
            file = self.scenes[id].object_file[i]

            if shape:
                self.add_or_edit_object(name=obj_name, frame_id=settings.base_link, size=size, color=color, pose=settings.scenes[id].object_poses[i], shape=shape, mass=mass, friction=friction, inertia=inertia, inertiaTransformation=inertiaTransformation, dynamic=dynamic, pub_info=pub_info, texture_file=texture_file)
            elif file:
                if scale: size = [settings.scenes[id].object_scales[i], 0, 0]
                else: size = [0,0,0]
                self.add_or_edit_object(file=f"{settings.paths.home}/{settings.paths.ws_folder}/src/mirracle_gestures/include/models/{file}", size=size, color=color, mass=mass, friction=friction, inertia=inertia, inertiaTransformation=inertiaTransformation, dynamic=dynamic, pub_info=pub_info, texture_file=texture_file, name=obj_name, pose=settings.scenes[id].object_poses[i], frame_id=settings.base_link)
            else:
                self.add_or_edit_object(name=obj_name, frame_id=settings.base_link, size=size, color=color, pose=settings.scenes[id].object_poses[i], shape='cube', mass=mass, friction=friction, inertia=inertia, inertiaTransformation=inertiaTransformation, dynamic=dynamic, pub_info=pub_info, texture_file=texture_file)
        scene = self.scenes[id]
        if id == 0:
            scene = None

        print(f"[Scenes] Scene {new_scene} ready!")

class CustomScene():
    ''' Custom scenes with custom names is defined with custom objects with
        pose and size

    '''
    def __init__(self, data=None, poses_data=None):
        assert data, "No Data!"
        assert len(data.keys()) == 1, "More input Scenes!"

        key = list(data.keys())[0]
        scene_data = data[key]
        self.name = key
        self.object_names = []
        self.object_poses = []
        self.object_sizes = []
        self.object_scales = []
        self.object_colors = []
        self.object_shapes = []
        self.object_masses = []
        self.object_frictions = []
        self.object_inertia = []
        self.object_inertiaTransform = []
        self.mesh_trans_origin = []
        self.object_dynamic = []
        self.object_pub_info = []
        self.object_texture_file = []
        self.object_file = []
        if not scene_data:
            return

        objects = scene_data['Objects']
        self.object_names = objects.keys()
        self.mesh_trans_origin = [Vector3(0.,0.,0.)] * len(objects.keys())
        for object in self.object_names:
            pose_vec = objects[object]['pose']

            self.object_poses.append(ParseYAML.parsePose(pose_vec, poses_data))
            self.object_sizes.append(ParseYAML.parsePosition(objects[object], poses_data, key='size'))
            self.object_scales.append(ParseYAML.parseScale(objects[object]))
            self.object_colors.append(ParseYAML.parseColor(objects[object]))
            self.object_shapes.append(ParseYAML.parseShape(objects[object]))
            self.object_frictions.append(ParseYAML.parseFriction(objects[object]))
            self.object_masses.append(ParseYAML.parseMass(objects[object]))
            self.object_inertia.append(ParseYAML.parseInertia(objects[object]))
            self.object_inertiaTransform.append(ParseYAML.parseInertiaTransform(objects[object]))
            self.object_dynamic.append(ParseYAML.parseDynamic(objects[object]))
            self.object_pub_info.append(ParseYAML.parsePubInfo(objects[object]))
            self.object_texture_file.append(ParseYAML.parseTextureFile(objects[object]))
            self.object_file.append(ParseYAML.parseMeshFile(objects[object]))
            if 'mesh_trans_origin' in scene_data.keys():
                if 'axes' in scene_data.keys():
                    self.mesh_trans_origin = TransformWithAxes(scene_data['mesh_trans_origin'], scene_data['axes'])
                else:
                    self.mesh_trans_origin = scene_data['mesh_trans_origin']



def init():
    ### 5. Latest Data                      ###
    ###     - Generated Scenes from YAML    ###
    ###     - Generated Paths from YAML     ###
    ###     - Current scene info            ###
    ###########################################
    global paths, scenes, scene

    ## Objects for saved scenes and paths
    #paths
    #scenes
    scenes = CustomScenes.GenerateFromYAML()
    paths = CustomPaths.GenerateFromYAML(scenes)
    scene = None # current scene informations

def inSceneObj(self, point):
    ''' in the zone of a box with length l
        Compatible for: Pose, Point, [x,y,z]
        Cannot be in two zones at once, return id of first zone
    '''
    collisionObjs = []
    if not settings.scene:
        return False
    z = [False] * len(settings.scene.object_poses)
    assert settings.scene, "Scene not published yet"
    if isinstance(point, Pose):
        point = point.position
    if isinstance(point, Point):
        point = [point.x, point.y, point.z]
    for n, pose in enumerate(settings.scene.object_poses):
        zone_point = pose.position

        zone_point = self.PointAdd(zone_point, settings.scene.mesh_trans_origin[n])
        #print(n, ": \n",zone_point.z, "\n" ,settings.scene.object_sizes[n].z, "\n", point[2])
        if settings.scene.object_sizes[n].y > 0.0:
            if zone_point.x <= point[0] <= zone_point.x+settings.scene.object_sizes[n].x:
              if zone_point.y <= point[1] <= zone_point.y+settings.scene.object_sizes[n].y:
                if zone_point.z <= point[2] <= zone_point.z+settings.scene.object_sizes[n].z:
                    collisionObjs.append(settings.scene.object_names[n])

        else:
            if zone_point.x <= point[0] <= zone_point.x+settings.scene.object_sizes[n].x:
              if zone_point.y >= point[1] >= zone_point.y+settings.scene.object_sizes[n].y:
                if zone_point.z <= point[2] <= zone_point.z+settings.scene.object_sizes[n].z:
                    collisionObjs.append(settings.scene.object_names[n])
        '''
                else:
                    print("z")
              else:
                print("y")
            else:
              print("x")
        '''
    return collisionObjs


def TransformWithAxes(data_to_transform, transform_mat):
    '''
    Parameters:
        data_to_transform (Vector3()[]) or dict
        transform_mat (2D array): size 3x3
    Returns:
        data_transformed (Vector3()[])
    '''

    if isinstance(data_to_transform[0], dict) and 'x' in data_to_transform[0].keys():
        new_data = []
        for vec in data_to_transform:
            new_data.append(Vector3(vec['x'], vec['y'], vec['z']))
        data_to_transform = new_data

    data_transformed = []
    for i in range(0, len(data_to_transform)):
        orig = data_to_transform[i]
        orig_ = Vector3()
        orig_.x = np.dot(transform_mat[0],[orig.x, orig.y, orig.z])
        orig_.y = np.dot(transform_mat[1],[orig.x, orig.y, orig.z])
        orig_.z = np.dot(transform_mat[2],[orig.x, orig.y, orig.z])
        data_transformed.append(Vector3(orig_.x, orig_.y, orig_.z))
    return data_transformed
