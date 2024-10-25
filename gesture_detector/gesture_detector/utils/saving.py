import json
import time, os, pathlib
import os
import numpy as np

from gesture_detector.hand_processing import frame_lib

class Recording():
    def __init__(self):
        self.present = False
        self.recording_start = None # time [s]

        # Queue of requests
        self.recording_requests = []
        # Array of frames
        self.data = []

    def is_recording(self):
        return self.present

    def start(self):
        self.present = True
        self.recording_start = time.perf_counter()

    def stop(self):
        directory, recording_length = self.recording_requests.pop(0)
        self.recording_start = None
        self.present = False
        if self.recording_check():
            Recording.save_recording(directory, self.data)
        self.data = []

    def auto_handle(self, f):
        if self.is_recording():
            self.data.append(f)
            if (time.perf_counter() - self.recording_start) >= self.recording_requests[0][1]:
                # recording is over
                self.stop()
        else:
            if self.recording_requests:
                self.start()
                self.data.append(f)

    def recording_check(self):
        if len(self.data) < 10:
            print(f"WARNING: Only {len(self.data)} record samples, it won't be saved!!")
            return False
        for r in self.data:
            if len(r.r.get_learning_data()) < 57 and len(r.l.get_learning_data()) < 57:
                print(f"WARNING: Data is not valid, only {len(r.r.get_learning_data())} {len(r.l.get_learning_data())} observations, it won't be saved!!")
                return False
        return True

    @staticmethod
    def save_recording(directory, object_to_save):
        print("[Saving] Saving data")

        ext = '.npy'
        pathlib.Path(directory).mkdir(exist_ok=True)

        i=0
        while os.path.isfile(directory+"/"+str(i)+ext):
            i+=1
        file_abs_path = directory+"/"+str(i)
        JSONLoader.save(file_abs_path, object_to_save)
        
        print(f"[Saving] Gesture movement {directory} saved")



class FrameEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (frame_lib.Vector, 
                            frame_lib.Bone, 
                            frame_lib.Finger, 
                            frame_lib.Hand, 
                            frame_lib.Frame, 
                            frame_lib.LeapGesturesCircle, 
                            frame_lib.LeapGesturesSwipe, 
                            frame_lib.LeapGesturesKeytap, 
                            frame_lib.LeapGesturesScreentap, 
                            frame_lib.LeapGestures)):
            return obj.__dict__  
        if isinstance(obj, np.ndarray):
            return list(obj)

        return json.JSONEncoder.default(self, obj)

def decode_frame(dct):
    if 'x' in dct:  
        return frame_lib.Vector(**dct)   
    if 'width' in dct:  
        b = frame_lib.Bone()  
        b.import_from_json(**dct)
        return b
    if 'bones' in dct:  
        f = frame_lib.Finger()  
        f.import_from_json(**dct)
        return f
    if 'visible' in dct:  
        h = frame_lib.Hand()  
        h.import_from_json(**dct)
        return h
    if 'seq' in dct:  
        h = frame_lib.Frame()  
        h.import_from_json(**dct)
        return h
    if 'keytap' in dct:
        h = frame_lib.LeapGestures()
        h.import_from_json(**dct)
        return h
    if 'clockwise' in dct:
        h = frame_lib.LeapGesturesCircle()   
        h.import_from_json(**dct)
        return h
    if 'speed' in dct:
        h = frame_lib.LeapGesturesSwipe()
        h.import_from_json(**dct)
        return h
    if 'keytap_flag' in dct:
        h = frame_lib.LeapGesturesKeytap()   
        h.import_from_json(**dct)
        return h 
    if 'state' in dct:
        h = frame_lib.LeapGesturesScreentap()   
        h.import_from_json(**dct)
        return h 
    return dct


class JSONLoader():
    @staticmethod
    def save(path: str, frame):
        pathlib.Path(path).parent.mkdir(exist_ok=True)
        json_data = json.dumps(frame, cls=FrameEncoder)

        with open(f"{path}", "w") as outfile:
            outfile.write(json_data)

    @staticmethod
    def load(path_: str):
        with open(f"{path_}", 'r') as openfile:
            json_data = str(json.load(openfile))

        json_data = json_data.replace("'", '"')
        json_data = json_data.replace("False", "false")
        json_data = json_data.replace("True", "true")
        json_data = json_data.replace("nan", "null")

        frame_copy = json.loads(json_data, object_hook=decode_frame)

        return frame_copy