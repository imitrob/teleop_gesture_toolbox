import time, os, pathlib
import os
import numpy as np

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
        np.save(file_abs_path+".npy", object_to_save)
        
        print(f"[Saving] Gesture movement {directory} saved")
