import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms

sys.path.append("/home/pierro/my_ws/src/teleop_gesture_toolbox/src/")
sys.path.append("/home/pierro/my_ws/src/teleop_gesture_toolbox/src/leapmotion")
from os_and_utils.loading import HandDataLoader, DatasetLoader

#data_path = '/home/pierro/UCB/data'
#seed=0

########################################################################################################################

def get(data_path,seed,fixed_order=False,pc_valid=0):
    data={}
    taskcla=[]
    size=[1,101,4] # from: size = [1,28,28], TODO: discard last Dimension somehow to size=[1,87]

    X, Y = DatasetLoader(['interpolate', 'discards']).load_dynamic(GlobalPaths().learn_path, Gs=['grab', 'kick', 'nothing'])

    n_test = int(len(Y)//(3.333))
    n_train = int(len(Y)-n_test)

    data_splited = {'train': {'X': None, 'Y':None}, 'test': {'X':None, 'Y':None}}

    data_splited1 = torch.utils.data.random_split(X, [n_train, n_test], generator=torch.Generator().manual_seed(seed))
    data_splited['train']['X'] = data_splited1[0]
    data_splited['test']['X'] = data_splited1[1]
    data_splited2 = torch.utils.data.random_split(Y, [n_train, n_test], generator=torch.Generator().manual_seed(seed))
    data_splited['train']['Y'] = data_splited2[0]
    data_splited['test']['Y'] = data_splited2[1]

    data[0]={}
    data[0]['name']='2-init-dynamic'
    data[0]['ncla']=2
    data[1]={}
    data[1]['name']='2-first-update-dynamic'
    data[1]['ncla']=1

    for s in ['train','test']:
        #loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False,drop_last=True)
        data[0][s]={'x': [],'y': []}
        data[1][s]={'x': [],'y': []}
        for image,label in zip(data_splited[s]['X'], data_splited[s]['Y']):
            if label < 2:
                data[0][s]['x'].append(image)
                data[0][s]['y'].append(label)
            elif label == 2:
                data[1][s]['x'].append(image)
                data[1][s]['y'].append(label-2)


    #len(data[0]['train']['x'])

    # "Unify" and save
    for n in [0,1]:
        for s in ['train','test']:
            #data[0]['train']['x']=torch.tensor(data[0]['train']['x']).view(-1,size[0],size[1])
            #torch.tensor(data[0]['train']['x']).shape

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
