#%%
import torch.nn as nn
import torch
import torch.functional as F
from utils.pyart import *


class POELayer(nn.Module):
    def __init__(self, n_joint):
        super(POELayer, self).__init__()
        self.n_joint = n_joint
        self.twist = nn.Parameter(torch.Tensor(n_joint,6,1))
        self.init_p = nn.Parameter(torch.Tensor(1,3,1))
        self.init_rpy =  nn.Parameter(torch.Tensor(1,3,1))
        self.twist.data.uniform_(-1,1)
        self.init_p.data.uniform_(-1,1)
        self.init_rpy.data.uniform_(-1,1)

        # init_SE3 = pr2t(self.init_p,self.init_rpy)
        # self.init_SE3 = init_SE3
        # self.register_buffer('init_SE3',init_SE3)
    
    

    def forward(self, q_value):
        n_joint = self.n_joint
        batch_size = q_value.size()[0]
        device = q_value.device
        out = torch.tile(torch.eye(4),(batch_size,1,1)).to(device)

        for joint in range(n_joint):
            twist = self.twist[joint,:,0]
            out = out @ srodrigues(twist, q_value[:,joint])
        out =  out @ pr2t(self.init_p,self.init_rpy)
        return out

class q_layer(nn.Module):
    def __init__(self,n_joint,inputdim):
        super(q_layer, self).__init__()
        self.layer1 = nn.Linear(inputdim,16)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        self.layer2 = nn.Linear(16,32)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        self.layer3 = nn.Linear(32,64)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        self.layer4 = nn.Linear(64,128)
        torch.nn.init.xavier_uniform_(self.layer4.weight)
        self.layer5 = nn.Linear(128,256)
        torch.nn.init.xavier_uniform_(self.layer5.weight)
        self.layer6 = nn.Linear(256,512)
        torch.nn.init.xavier_uniform_(self.layer6.weight)

        self.layer11 = nn.Linear(512,512)
        torch.nn.init.xavier_uniform_(self.layer11.weight)
        self.layer12 = nn.Linear(512,512)
        torch.nn.init.xavier_uniform_(self.layer12.weight)
        self.layer13 = nn.Linear(512,512)
        torch.nn.init.xavier_uniform_(self.layer13.weight)
        self.layer14 = nn.Linear(512,256)
        torch.nn.init.xavier_uniform_(self.layer14.weight)
        
        self.layer7 = nn.Linear(256,128)
        torch.nn.init.xavier_uniform_(self.layer7.weight)
        self.layer8 = nn.Linear(128,64)
        torch.nn.init.xavier_uniform_(self.layer8.weight)
        self.layer9 = nn.Linear(64,32)
        torch.nn.init.xavier_uniform_(self.layer9.weight)
        self.layer10 = nn.Linear(32,n_joint)
        torch.nn.init.xavier_uniform_(self.layer10.weight)

    def forward(self, motor_control):
        out = self.layer1(motor_control)
        out = nn.LeakyReLU()(out)

        out = self.layer2(out)
        out = nn.LeakyReLU()(out)

        out = self.layer3(out)
        out = nn.LeakyReLU()(out)

        out = self.layer4(out)
        out = nn.LeakyReLU()(out)

        out = self.layer5(out)
        out = nn.LeakyReLU()(out)

        out = self.layer6(out)
        out = nn.LeakyReLU()(out)

        out = self.layer11(out)
        out = nn.LeakyReLU()(out)
        out = self.layer12(out)
        out = nn.LeakyReLU()(out)
        out = self.layer13(out)
        out = nn.LeakyReLU()(out)
        out = self.layer14(out)
        out = nn.LeakyReLU()(out)
        
        out = self.layer7(out)
        out = nn.LeakyReLU()(out)

        out = self.layer8(out)
        out = nn.LeakyReLU()(out)

        out = self.layer9(out)
        out = nn.LeakyReLU()(out)

        out = self.layer10(out)
        out = nn.LeakyReLU()(out)
        
        q_value = out
        return q_value

class Model(nn.Module):
    def __init__(self, n_joint, inputdim):
        super(Model,self).__init__()
        self.q_layer = q_layer(n_joint, inputdim)
        self.poe_layer = POELayer(n_joint)

    def forward(self, motor_control):
        out = self.q_layer(motor_control)
        SE3 = self.poe_layer(out)

        return SE3




#%%