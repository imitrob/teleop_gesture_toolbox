
import os
path = os.path.dirname(os.path.abspath(__file__))
package_path = "/".join(path.split("/")[:-1])

scenes_path = os.environ.get("SCENES_PATH", path+"/scene_makers/scenes")
