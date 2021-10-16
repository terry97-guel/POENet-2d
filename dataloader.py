#%%
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import random
import os 

def getRandomDataloader(data_path, n_workers, batch):
    Scenario_ls = os.listdir(data_path)
    data_path = data_path + '/' + random.choice(Scenario_ls)
    dataloader = RandomDataloader(data_path, n_workers, batch)

    return dataloader

class RandomDataloader(DataLoader):
    def __init__(self,data_path, n_workers,batch):
        self.dataset = RandomDataset(data_path)
        super().__init__(self.dataset, batch_size=batch, shuffle=True, num_workers=n_workers)

class RandomDataset(Dataset):
    def __init__(self,data_path,):
        rawdata = np.loadtxt(data_path)
        self.label = torch.Tensor(rawdata[:,:3])
        self.input = torch.Tensor(rawdata[:,3:])

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self,idx):
        return self.input[idx], self.label[idx]
    

class ToyDataset(Dataset):
    def __init__(self,data_path):
        Scenario_ls = os.listdir(data_path)
        self.label = torch.tensor([])
        self.input = torch.tensor([])

        for scenario in Scenario_ls:
            file_path = data_path + '/' + scenario
            rawdata = np.loadtxt(file_path)
            self.label = torch.cat((self.label,torch.Tensor(rawdata[:,:3])),0)
            self.input = torch.cat((self.input,torch.Tensor(rawdata[:,3:])),0)
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self,idx):
        return self.input[idx], self.label[idx]

class ToyDataloader(DataLoader):
    def __init__(self,data_path, n_workers,batch):
        self.dataset = ToyDataset(data_path)
        super().__init__(self.dataset, batch_size=batch, shuffle=True, num_workers=n_workers)
  