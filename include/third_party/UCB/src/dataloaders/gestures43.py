import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
from itertools import accumulate

sys.path.append("/home/pierro/my_ws/src/mirracle_gestures/src/")
sys.path.append("/home/pierro/my_ws/src/mirracle_gestures/src/leapmotion")
from os_and_utils.loading import HandDataLoader, DatasetLoader
from os_and_utils.utils import GlobalPaths

#data_path = '/home/pierro/UCB/data'
#seed=0

########################################################################################################################

def get_n_testtrain(Ylen, ratio=0.3):
    n_test = int(Ylen * ratio)
    n_train = Ylen - n_test
    return n_train, n_test

'''
n_test, n_train = get_n_testtrain(90)
n_test, n_train
type(n_test), type(n_train)
'''
def get(data_path,seed,fixed_order=False,pc_valid=0):
    # id: n update
    _map = {
    0: 0,
    1: 0,
    2: 1
    }
    Gs = ['grab', 'kick', 'nothing']
    args = ['interpolate', 'discards']

    ###########################################################################################

    data={}
    taskcla=[]
    #size=[1,57,1] # from: size = [1,28,28], TODO: discard last Dimension somehow to size=[1,87]

    if 'static' in args:
        g_type = 'static'
        X, Y = DatasetLoader(args).load_static(GlobalPaths(change_working_directory=False).learn_path, Gs=Gs)
    else:
        g_type = 'dynamic'
        X, Y = DatasetLoader(args).load_dynamic(GlobalPaths(change_working_directory=False).learn_path, Gs=Gs)
    uniqY = list(set(Y))
    size = list(X[0:1].shape)

    n_train, n_test = get_n_testtrain(len(Y))

    data_splited = {'train': {'X': None, 'Y':None}, 'test': {'X':None, 'Y':None}}

    data_splited1 = torch.utils.data.random_split(X, [n_train, n_test], generator=torch.Generator().manual_seed(seed))
    data_splited['train']['X'] = data_splited1[0]
    data_splited['test']['X'] = data_splited1[1]
    data_splited2 = torch.utils.data.random_split(Y, [n_train, n_test], generator=torch.Generator().manual_seed(seed))
    data_splited['train']['Y'] = data_splited2[0]
    data_splited['test']['Y'] = data_splited2[1]

    n_updates = len(list(set([_map[k] for k in list(_map.keys())])))

    def make_m(_map):
        prev = -1
        m = []
        for k in list(_map.keys()):
            curr = _map[k]
            if prev != curr:
                m.append(k)
                prev = curr
        return m
    m = make_m(_map)
    tmp = [_map[k] for k in list(_map.keys())]
    _, counts = np.unique(tmp, return_counts=True)
    counts = list(counts)

    for n,ncla in enumerate(counts):
        data[n]={}
        data[n]['ncla']=ncla
        data[n]['name']=f"{data[n]['ncla']}-{g_type}-n-{n}"


    for s in ['train','test']:
        #loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False,drop_last=True)
        for n in range(n_updates):
            data[n][s]={'x': [],'y': []}
        for image,label in zip(data_splited[s]['X'], data_splited[s]['Y']):
            ud = _map[label]
            data[ud][s]['x'].append(image)
            data[ud][s]['y'].append(label-m[ud])


    # "Unify" and save
    for n in list(range(n_updates)):
        for s in ['train','test']:
            data[n][s]['x']=torch.tensor(data[n][s]['x'],dtype=torch.float32).view(-1,size[0],size[1],size[2])
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)

    # Validation
    for t in data.keys():
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'].clone()
        data[t]['valid']['y']=data[t]['train']['y'].clone()

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size

########################################################################################################################
