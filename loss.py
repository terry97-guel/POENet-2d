#%%
import torch

class Get_Loss_Function():
    def Pos_norm2(self,output, label):
        nBatch = len(label)
        output = output[:,0:3,3]
        loss = torch.nn.MSELoss()(output,label)

        return loss
    
    # def se3_norm2(output, label):
        #Fill me

def get_regularizer(args):
    #ADD CODE: use args to determine which regularizer to use & coefficient
    regul_fn = Twist_norm

    return regul_fn

def Twist_norm(model):
    loss = torch.norm(model.poe_layer.twist,dim=0)
    device = loss.device
    loss = loss - torch.ones(6,1,dtype=torch.float).to(device)
    loss = torch.norm(loss)

    return loss