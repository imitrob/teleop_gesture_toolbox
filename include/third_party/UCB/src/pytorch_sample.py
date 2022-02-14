import sys,os,argparse
import numpy as np
import torch
import utils


class AA():
    def __init__(self):
        self.seed = 0
        self.device = 'cuda:0'
        self.experiment = 'gestures43'
        self.approach = 'ucb'
        self.data_path = '../data/'
        self.output = ''
        self.checkpoint_dir = '../checkpoints/'
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

args = AA()

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from dataloaders import gestures43 as dataloader
from approaches import ucb as approach
from networks import mlp_ucb as network


data,taskcla,inputsize=dataloader.get(data_path=args.data_path, seed=args.seed)
print('Input size =',inputsize,'\nTask info =',taskcla)
args.num_tasks=len(taskcla)
args.inputsize, args.taskcla = inputsize, taskcla

# Inits
print('Inits...')
model=network.Net(args).to(args.device)

checkpoint = torch.load(os.path.join(args.checkpoint, 'model_{}.pth.tar'.format(args.sti)))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device=args.device)

appr=approach.Appr(model,args=args)


xtest=data[0]['test']['x'].to(args.device)
ytest=data[0]['test']['y'].to(args.device)
for x_, y_ in zip(xtest, ytest):
    out, pred = appr.eval_single(0,x_)
    print(f'{pred==y_}, out {out}, pred {pred}, y_ {y_}')

xtest=data[1]['test']['x'].to(args.device)
ytest=data[1]['test']['y'].to(args.device)
for x_, y_ in zip(xtest, ytest):
    out, pred = appr.eval_single(1,x_)
    print(f'{pred==y_}, out {out}, pred {pred}, y_ {y_}')








#
