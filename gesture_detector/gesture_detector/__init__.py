
import os
path = os.path.dirname(os.path.abspath(__file__))
package_path = "/".join(path.split("/")[:-1])

def _env_dir(var, default):
    p = os.environ.get(var, default)
    return p if p.endswith("/") else p+"/"

saved_models_path = _env_dir("GESTURE_MODELS_PATH", package_path+"/saved_models/")
gesture_data_path = _env_dir("GESTURE_DATA_PATH", package_path+"/gesture_data/")
