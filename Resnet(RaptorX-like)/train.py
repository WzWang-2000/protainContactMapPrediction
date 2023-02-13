import numpy as np
import h5py
import torch
import torch.nn as nn
import random
from ResNet import ResNet,ContactM
import time
import matplotlib.pyplot as plt
import seaborn
valid = [x.strip() for x in open('C:/Users/wwz2000/study1/valid_list.txt')]
data = h5py.File('C:/Users/wwz2000/study1/esm2_3B_targetEmbed.h5', 'r')
xyz = h5py.File('C:/Users/wwz2000/study1/xyz.h5', 'r')

finalnet = ResNet()
finalnet.cuda()

optimizer = torch.optim.Adam(finalnet.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss(reduction='mean')

def loadprotein(pdb):
    gap = xyz[pdb]['gap'][:]
    coor = xyz[pdb]['xyz'][np.where(gap > 0)[0]]  # [L, 4, 3], 其中L是序列长度，4代表四个原子，顺序是CA， C， N和CB
    X1D = data[pdb]['token_embeds'][0, np.where(gap > 0)[0]]
    X2D = data[pdb]['feature_2D'][0, :, np.where(gap > 0)[0]][:, :, np.where(gap > 0)[0]]
    print(X1D.shape, X2D.shape)
    if X1D.shape[0] > 256:
        start = random.randint(0, X1D.shape[0] - 256)
        end = start + 256
        X1D = X1D[start:end, :]
        X2D = X2D[:, start:end, start:end]
        coor = coor[start:end, :, :]
    return X1D,X2D,coor

def train_func(n,valid_data):
    epochloss=[]
    for epoch in range(n):
        print('EPOCH:', epoch )
        random.shuffle(valid)
        starttime = time.time()
        i = 1
        allloss=[]
        for pdb in valid_data:
            X1D,X2D,coor=loadprotein(pdb)
            X1D = torch.from_numpy(X1D.astype(np.float32))
            X2D = torch.from_numpy(X2D.astype(np.float32))
            # print(X1D.shape)
            gpu_X1D = X1D.cuda()
            gpu_X2D = X2D.cuda()
            predict = finalnet(gpu_X1D, gpu_X2D)
            fact = ContactM(coor)
            gpu_fact = fact.cuda()
            loss = loss_func(predict, gpu_fact)
            allloss.append(loss.item())
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(i)
            i = i + 1
        endtime = time.time()
        print(endtime - starttime)
        lossmean=sum(allloss)/len(allloss)
        epochloss.append(lossmean)
        print(lossmean)
        torch.save(finalnet, f'C:/Users/wwz2000/study1/eachResNet{epoch}.pkl')  # 保存整个网络
        state = {'net':finalnet.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch,'meanloss':lossmean}
        torch.save(state, f'C:/Users/wwz2000/study1/ResNetpara{epoch}.pth')

    return epochloss

train_func(10,valid)