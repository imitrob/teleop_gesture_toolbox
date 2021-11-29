
import sys
import os

def get(data_path,seed,fixed_order=False,pc_valid=0,discardedFiles={},Gs,taskcl=0):
    '''
    Parameters:
        Gs (String[]): List of strings, where index is also ID (flag)
                       - Only append new values with new learn, do not delete
        taskcl (Int): Update number (default=0)
    '''
    data={}
    taskcl = []
    size = [1,x,x]

    # Create dict, for every {'gesture1': ['file1', 'file2'], ...}
    allFiles = {key: [] for key in Gs}
    for n,G in enumerate(Gs):
        for r in os.listdir(data_path+G):
            allFiles[G].append(r)

    # Discard previously loaded files
    for n,G in enumerate(allFiles.keys()):
        if G not in discardedFiles.keys(): continue
        allFiles[G] = [item for item in allFiles[G] if item not in discardedFiles[G]]


    data[0]={}
    data[0]['name']='all-static'
    data[0]['ncla']=7
    data[0][s]={'X': [],'Y': []}

    for n,G in enumerate(Gs):
        for r in os.listdir(data_path):
            with open(data_path+r,'rb') as input:
                dataset_files.append(i_file)
                HandAdvObjs.append(pickle.load(input, encoding="latin1"))
                Y.append(n)

    ### TEST ####
    recordings = 0
    recordingsLongerThan1_5secs = 0
    recordingsLessPointsThan10 = 0
    recordingsWithNoAngles = 0
    for n,rec in enumerate(X):
        recordings+=1
        i = 0
        while i < len(rec):
            i+=1
            if abs((rec[-1].r.pPose.header.stamp-rec[-i].r.pPose.header.stamp).to_sec()) > 1.5:
                recordingsLongerThan1_5secs+=1
        ## Pick i elements
        if i < 10:
            recordingsLessPointsThan10 +=1
            continue
        c = False
        for r in rec:
            if r.r.wrist_hand_angles_diff[1:3] == []:
                c = True
                recordingsWithNoAngles+=1
        if c: continue
    print(f"recordings {recordings}, recordingsLongerThan1_5secs {recordingsLongerThan1_5secs}, recordingsLessPointsThan10 {recordingsLessPointsThan10}, recordingsWithNoAngles {recordingsWithNoAngles}")
    #############



    if X == []:
        print(('[WARN*] No data was imported! Check parameters path folder (learn_path) and gesture names (Gs)'))
        return None


    for n, X_n in enumerate(X):
        row = []
        for m, X_nt in enumerate(X_n):
            X_nt_ = []
            X_nt_.extend(X_nt.r.get_angles_array())
            X_nt_.extend(X_nt.r.get_distances_array())
            if 'include_palm' in args:
                X_nt_.extend(X_nt.r.pRaw)
                if 'include_palm_and_finger' in args:
                    X_nt_.extend(X_nt.r.index_position)
            if len(X_nt_) != 0:
                row.append(np.array(X_nt_))
        X_.append(row)
        Y_.append(Y[n])
