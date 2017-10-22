#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:35:31 2017

@author: ayooshmac
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.style.use('ggplot')
from torchnet import meter

import pickle as pkl 
from custom_model import *
from loaders import *


import torch 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import nn
from torchvision import transforms



def preprocess(df):
    df['bare_nuclei'].replace({'?': np.nan}, inplace = True)
    df.dropna(inplace=True)
    df["bare_nuclei"] = df["bare_nuclei"].astype(int)
    df.drop(["id"], axis = 1, inplace=True)
    df["class"] = df["class"].map({2:0, 4:1})
    return df



load = loaders("data/data.csv", preprocess)



tran = transforms.Compose([transforms.ToTensor()])

a = open("data/datasets", "rb")
datasets = pkl.load(a)

train, test, valid = datasets[0], datasets[1], datasets[2]

trainloader, testloader, validloader = load.get_loaders([0.6, 0.2, 0.2], tran)
trainloader, testloader, validloader = get_dataloaders(datasets, tran, batch_size = 30)

D_in, H, D_out = trainloader.dataset.shape[1] - 1, 30 , 2



model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax()
)

lr = 0.001
loss_fn = nn.CrossEntropyLoss()
wd = 0.1
optimizer = optim.Adam(model.parameters(), lr, weight_decay=wd)

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0, 2/float(12))
        m.bias.data.normal_(0, 2/float(12))


a = custom_model(model, loss_fn)
a.model.apply(init_weights)
a.train(trainloader, testloader, validloader, optimizer, 50, plot = True)
a.plot(a.get_logs())

accuracy, auc, cm = a.metrics_val(testloader)
print ("Train Accuracy", a.metrics_val(trainloader)[0])
print ("Test Accuracy: ", accuracy)
print("AUROC:", auc, "\nConfusion Matrix\n", cm)





