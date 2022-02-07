import time
import os

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
        directory, save_method, recording_length = self.recording_requests.pop(0)
        self.recording_start = None
        self.present = False
        if self.recording_check():
            Recording.save_recording(directory, self.data, save=save_method)
        self.data = []

    def auto_handle(self, f):
        if self.is_recording():
            self.data.append(f)
            if (time.perf_counter() - self.recording_start) >= self.recording_requests[0][2]:
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
    def save_recording(directory, object_to_save, save='numpy'):
        print("[Saving] Saving data")

        if save=='numpy': ext = '.npy'
        elif save=='pickle': ext = '.pkl'
        else: raise Exception("Wrong save form, choose numpy or pickle!")

        if not os.path.isdir(directory):
            os.mkdir(directory)
        i=0
        while os.path.isfile(directory+"/"+str(i)+ext):
            i+=1
        file_abs_path = directory+"/"+str(i)

        if save == 'numpy':
            import numpy as np # import here due to compatibility issues
            np.save(file_abs_path+".npy", object_to_save)
        elif save == 'pickle':
            import pickle
            with open(file_abs_path+".pkl", 'wb') as output:
                pickle.dump(object_to_save, output, pickle.HIGHEST_PROTOCOL)

        print(f"[Saving] Gesture movement {directory} saved")
