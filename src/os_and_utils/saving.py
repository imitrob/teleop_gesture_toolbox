

def save_recording(directory, object_to_save, save='numpy'):
    print("[Saving] Saving data")
    if not isdir(directory):
        from os import mkdir
        mkdir(directory)
    i = 0
    while not isfile(f"{directory}/{str(i)}.pkl"):
        i+=1
    file_abs_path = f"{directory}/{str(i)}.pkl"

    if save == 'numpy':
        import numpy as np
        np.save(f"{file_abs_path}.npy", object_to_save)
    else: # 'pickle'
        import pickle
        with open(f"{file_abs_path}.pkl", 'wb') as output:
            pickle.dump(object_to_save, output, pickle.HIGHEST_PROTOCOL)



    print("[Saving] Gesture movement ", directory," saved")
