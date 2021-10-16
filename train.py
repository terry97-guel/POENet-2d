#%%
import argparse
import torch
import numpy as np
from dataloader import *
from model import Model
# from trainer import Trainer
from loss import *
import os
import random
from pathlib import Path
import wandb
import time

# fix random seeds for reproducibility
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def train_iteration(model, optimizer, input, label,loss_fn, regul_fn, args):
    output = model(input)
    loss = loss_fn(output,label)
    regularizer_loss = args.TwistNormCoefficient * regul_fn(model)
    total_loss = loss + regularizer_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return loss


def main(args):
    #set logger
    wandb.init(project = args.pname)
    delta_time = [0,0,0,0,0]

    #set device
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    #set model
    model = Model(args.n_joint, args.input_dim)
    model = model.to(device)
    model.train()

    #load weight when requested
    if os.path.isfile(args.resume_dir):
        model = torch.load(args.resume_dir)

    #set optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr= args.lr, weight_decay=args.wd)
    
    #declare loss function
    Loss = Get_Loss_Function()
    loss_fn = Loss.__getattribute__(args.loss_function)

    #regularizer function
    # regul_fn = get_regularizer(args)
    regul_fn = Twist_norm

    #assert path to save model
    pathname = args.save_dir
    Path(pathname).mkdir(parents=True, exist_ok=True)

    #set dataloader
    data_loader = ToyDataloader(args.data_path, args.n_workers, args.batch_size)
    data_loader = iter(data_loader)
    for iterate in range(args.iterations):
        #estimate eta
        time_start = time.time()
        
        if iterate > 5:
            avg_time = sum(delta_time)/len(delta_time)
            eta_time = (args.iterations - iterate) * avg_time
            h = int(eta_time //3600)
            m = int((eta_time %3600)//60)
            s = int((eta_time %60))
            print("iteration: {}, eta:{}:{}:{}".format(iterate,h,m,s))
        else: 
            print("iteration: {}".format(iterate))

        (input,label) = next(data_loader)
        input = input.to(device)
        label = label.to(device)
        train_loss = train_iteration(model, optimizer, input, label, loss_fn, regul_fn, args)
        print('TrainLoss: {}'.format(train_loss))
        time_end = time.time()
        delta_time[iterate%5] = time_end-time_start

        #save train_loss from last epoch
        wandb.log({'TrainLossScenc':train_loss},step = iterate)
    

    #save model 
    if (iterate+1) % args.save_period==0:
        filename = args.save_dir + '/checkpoint_{}'.format(iterate+1)
        print("saving... {}".format(filename))
        state = {
            'state_dict':model.state_dict(),
            'optimizer':optimizer.state_dict()
        }
        torch.save(state, filename)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description= 'parse for POENet')
    args.add_argument('--n_joint', default= 12, type=int,
                    help='number of joints')
    args.add_argument('--batch_size', default= 128, type=int,
                    help='batch_size')
    args.add_argument('--data_path', default= './data/2dim_log_spiral',type=str,
                    help='path to data')
    args.add_argument('--save_dir', default= './output/run1',type=str,
                    help='path to save model')
    args.add_argument('--resume_dir', default= './output/',type=str,
                    help='path to load model')
    args.add_argument('--device', default= '1',type=str,
                    help='device to use')
    args.add_argument('--n_workers', default= 2, type=int,
                    help='number of data loading workers')
    args.add_argument('--wd', default= 0.00, type=float,
                    help='weight_decay for model layer')
    args.add_argument('--lr', default= 0.01, type=float,
                    help='learning rate for model layer')
    # args.add_argument('--optim', default= 'adam',type=str,
    #                 help='optimizer option')
    args.add_argument('--loss_function', default= 'Pos_norm2',type=str,
                    help='get loss function')
    args.add_argument('--minimal_params', default=True)
    args.add_argument('--use_adjoint', default=False)
    args.add_argument('--input_dim', default= 2, type=int,
                    help='dimension of input')
    args.add_argument('--iterations', default= 300, type=int,
                    help='number of epoch to perform')
    args.add_argument('--early_stop', default= 50, type=int,
                    help='number of n_Scence to early stop')
    args.add_argument('--save_period', default= 100, type=int,
                    help='number of scenes after which model is saved')
    args.add_argument('--TwistNormCoefficient', default= 0.1, type=float,
                    help='Coefficient for TwistNorm')
    args.add_argument('--pname', default= 'POE',type=str,
                    help='Project name')
    
    args = args.parse_args()
    main(args)
#%%
