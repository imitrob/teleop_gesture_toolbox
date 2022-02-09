#!/usr/bin/env python3
''' Run main_sample_thread.py with: gesture_detection_approach: pytorch
                                    in include/custom_settings/gesture_config.yaml
'''
import sys,os
import numpy as np
import torch
import settings

class PyTorch_Sample():
    def __init__(self):
        # Args
        self.seed = 0
        self.device = 'cpu' #'cuda:0'
        self.experiment = ''
        self.approach = 'ucb'
        self.data_path = settings.paths.UCB_path+'data/'
        self.output = ''

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

        self.dataset_n_time_samples = 32
        self.dataset_n_observations = 3

        self.checkpoint = settings.paths.UCB_path+'checkpoints/'

        #############################################################

        # Seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    def sample(self, data):
        d = np.array(data.data)
        d.resize(data.layout.dim[0].size,data.layout.dim[1].size)
        d = torch.tensor(d, dtype=torch.float32).view(-1,self.inputsize[0],self.inputsize[1],self.inputsize[2])
        d = d.to(self.device)
        return self.appr.eval_single(0,d)

    def init(self, network_name='gestures_dynamic_ucb'):
        self.experiment = '_'.join(network_name.split("_")[:-1])

        # It needs to be imported here, because it depends on settings
        sys.path.append(settings.paths.UCB_path)
        from src.dataloaders import gestures_dynamic as dataloader
        from src.approaches import ucb as approach
        from src.networks import mlp_ucb as network

        data,taskcla,inputsize=dataloader.get(data_path=self.data_path, seed=self.seed, args=self)
        print('Input size =',inputsize,'\nTask info =',taskcla)
        self.num_tasks=len(taskcla)
        self.inputsize, self.taskcla = inputsize, taskcla

        self.model=network.Net(self).to(self.device)

        checkpoint = torch.load(os.path.join(self.checkpoint+network_name, 'model_{}.pth.tar'.format(self.sti)), map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device=self.device)

        self.appr=approach.Appr(self.model,args=self)

        print(f"[Sample thread] network is: {network_name}")

'''
pts = PyTorch_Sample()
pts.init()

print(pts.sample( np.zeros((101,3)) ) )
'''
# ~14ms

if __name__ == '__main__':
    print("Run main_sample_thread.py with static/dynamic parameter instead!")

#
