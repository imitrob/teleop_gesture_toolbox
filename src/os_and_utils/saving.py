

def save_recording(directory, object_to_save, save='numpy'):
    print("[Saving] Saving data")
    if not isdir(directory):
        from os import mkdir
        mkdir(directory)
    i = 0
    while isfile(directory+"/"+str(i)+".pkl"):
        i+=1
    file_abs_path = directory+"/"+str(i)+".pkl"

    if save == 'numpy':
        import numpy as np
        np.save(file_abs_path+".npy", object_to_save)
    else: # 'pickle'
        import pickle
        with open(file_abs_path+".pkl", 'wb') as output:
            pickle.dump(object_to_save, output, pickle.HIGHEST_PROTOCOL)

    print("[Saving] Gesture movement ", directory," saved")
