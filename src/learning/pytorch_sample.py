#!/usr/bin/env python3
''' Run main_sample_thread.py with: gesture_detection_approach: pytorch
                                    in include/custom_settings/gesture_config.yaml
'''
import sys,os
import numpy as np
import torch
sys.path.append("/home/pierro/UCB/src")

class PyTorch_Sample():
    def __init__(self):
        # Args
        self.seed = 0
        self.device = 'cuda:0'
        self.experiment = 'gestures43'
        self.approach = 'ucb'
        self.data_path = '/home/pierro/UCB/data/'
        self.output = ''
        self.checkpoint_dir = '/home/pierro/UCB/checkpoints/'
        self.nepochs = 200
        self.sbatch = 64
        self.lr = 0.01
        self.nlayers = 1
        self.nhid = 1200
        self.parameter = ''

        self.samples = 10
        self.rho = -3
        self.sig1 = 0.0
        self.sig2 = 6.0
        self.pi = 0.25
        self.arch = 'mlp'
        self.resume = 'no'
        self.sti = 0

        self.checkpoint = '/home/pierro/UCB/checkpoints/gestures43_ucb'

        #############################################################

        # Seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    def sample(self, data):
        d = np.array(data)
        d = torch.tensor(d, dtype=torch.float32).view(-1,pts.inputsize[0],pts.inputsize[1],pts.inputsize[2])
        d = d.to(self.device)
        return self.appr.eval_single(0,d)

    def init(self):
        from dataloaders import gestures43 as dataloader
        from approaches import ucb as approach
        from networks import mlp_ucb as network

        data,taskcla,inputsize=dataloader.get(data_path=self.data_path, seed=self.seed)
        print('Input size =',inputsize,'\nTask info =',taskcla)
        self.num_tasks=len(taskcla)
        self.inputsize, self.taskcla = inputsize, taskcla

        self.model=network.Net(self).to(self.device)

        checkpoint = torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(self.sti)))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device=self.device)

        self.appr=approach.Appr(self.model,args=self)

'''
pts = PyTorch_Sample()
pts.init()

print(pts.sample( np.zeros((101,3)) ) )
'''
# ~14ms


#
