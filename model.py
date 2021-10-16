#%%
import torch.nn as nn
import torch
import torch.functional as F
from utils.pyart import *


class POELayer(nn.Module):
    def __init__(self, n_joint):
        super(POELayer, self).__init__()
        self.twist = nn.Parameter(torch.Tensor(n_joint,6,1))
        self.init_p = torch.Tensor(1,3,1)
        self.init_rpy = torch.Tensor(1,3,1)
        self.twist.data.uniform_(-1,1)
        self.init_p.data.uniform_(-1,1)
        self.init_rpy.data.uniform_(-1,1)

        init_SE3 = pr2t(self.init_p,self.init_rpy)
        self.register_buffer('init_SE3',init_SE3)

    def forward(self, q_value):
        out = POE(self.twist, q_value) @ self.init_SE3
        return out

class q_layer(nn.Module):
    def __init__(self,n_joint,inputdim):
        super(q_layer, self).__init__()
        self.layer1 = nn.Linear(inputdim,16)
        self.layer2 = nn.Linear(16,32)
        self.layer3 = nn.Linear(32,64)
        self.layer6 = nn.Linear(64,128)
        self.layer8 = nn.Linear(128,512)
        self.layer10 = nn.Linear(512,512)
        self.layer9 = nn.Linear(512,128)
        self.layer7 = nn.Linear(128,64)
        self.layer4 = nn.Linear(64,32)
        self.layer5 = nn.Linear(32,n_joint)

    def forward(self, motor_control):
        out = self.layer1(motor_control)
        out = nn.Tanh()(out)
        out = self.layer2(out)
        out = nn.Tanh()(out)
        out = self.layer3(out)
        out = nn.Tanh()(out)
        out = self.layer6(out)
        out = nn.Tanh()(out)
        out = self.layer8(out)
        out = nn.Tanh()(out)
        out = self.layer10(out)
        out = nn.Tanh()(out)
        out = self.layer9(out)
        out = nn.Tanh()(out)
        out = self.layer7(out)
        out = nn.Tanh()(out)
        out = self.layer4(out)
        out = nn.Tanh()(out)
        out = self.layer5(out)
        q_value = nn.Tanh()(out)
        
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