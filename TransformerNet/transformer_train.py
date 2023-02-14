import numpy as np
import h5py
import torch
import torch.nn as nn
import random
import math
import time

random.seed(1)
def activation_func(activation, inplace=False):
    '''
    Activation functions
    '''
    if activation is None: return None
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=inplace)],
        ['elu', nn.ELU(inplace=inplace)],
        ['leaky_relu', nn.LeakyReLU(inplace=inplace)],
        ['selu', nn.SELU(inplace=inplace)],
        ['none', nn.Identity()],
    ])[activation]

def normalization_func(input_size, normalization, n_dim):
    '''
    Normalization functions
    '''
    assert input_size in ['1D', '2D'], 'input_size: 1D or 2D.'
    if input_size=='1D':
        return  nn.ModuleDict([
            ['batch', nn.BatchNorm1d(n_dim)],
            ['instance', nn.InstanceNorm1d(n_dim)],
            ['layer', nn.LayerNorm(n_dim)],
            ['none', nn.Identity()],
        ])[normalization]

    elif input_size=='2D':
        return  nn.ModuleDict([
            ['batch', nn.BatchNorm2d(n_dim)],
            ['instance', nn.InstanceNorm2d(n_dim)],
            ['none', nn.Identity()]
        ])[normalization]

class Conv1dAuto(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size
        self.padding = (self.dilation[0] * (self.kernel_size[0]-1) // 2, )

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size and dilation
        self.padding =  (self.dilation[0]*(self.kernel_size[0]-1)//2,
                            self.dilation[1]*(self.kernel_size[1]-1)//2)
class ResBlock2d(nn.Module):
    def __init__(self, n_input=553, kernel_size=5, dilation=2,
                 dropout=0.0, activation='elu', normalization='batch', bias=False, *args, **kwargs):
        super().__init__()
        self.block2d = nn.Sequential(
            Conv2dAuto(n_input, n_input, kernel_size=kernel_size, dilation=dilation, bias=bias),
            normalization_func('2D', normalization, n_input),
            activation_func(activation),
            nn.Dropout2d(p=dropout),
            Conv2dAuto(n_input, n_input, kernel_size=kernel_size, dilation=dilation, bias=bias),
            normalization_func('2D', normalization, n_input),
        )
        self.activate = activation_func(activation)

    def forward(self, x2d):
        residual = x2d
        x2d = self.block2d(x2d)
        x2d += residual
        x2d = self.activate(x2d)
        return x2d

def ContactM(coor):
    x=torch.tensor(coor[:,3,0].reshape(-1,1))
    y=torch.tensor(coor[:,3,1].reshape(-1,1))
    z=torch.tensor(coor[:,3,2].reshape(-1,1))
    distance_m=torch.sqrt((x-x.T)**2+(y-y.T)**2+(z-z.T)**2)
    l=coor.shape[0]
    noncon_m=(distance_m>=8).long()
    return noncon_m.reshape(1,noncon_m.shape[0],-1)


class MSAMultiHeadAttention(nn.Module):
    def __init__(self,n_input,n_head,dropout=0.0,bias=False,*args,**kwargs):
        super().__init__()
        self.toK=nn.Linear(n_input,n_input,bias=False)
        self.toQ=nn.Linear(n_input,n_input,bias=False)
        self.toV=nn.Linear(n_input,n_input,bias=False)
        self.Attention=nn.MultiheadAttention(n_input,n_head,dropout=dropout,bias=False)
    def forward(self,X):
        K=self.toK(X)
        Q=self.toQ(X)
        V=self.toV(X)
        Attn=self.Attention(Q,K,V,need_weights=True)
        return Attn


class SeqAttentionBlock(nn.Module):
    def __init__(self,n_input,n_head,ffn_n_hiddens,dropout=0.0,bias=False,*args,**kwargs):
        super().__init__()
        self.MultiHeadAttention=MSAMultiHeadAttention(n_input,n_head)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(n_input)
        self.ffn=nn.Sequential(
                                nn.Linear(n_input,ffn_n_hiddens),
                                nn.ReLU(),
                                nn.Linear(ffn_n_hiddens,n_input)
                                )
        self.ln2 = nn.LayerNorm(n_input)
    def forward(self,x1d):
        residual=x1d
        x1d=self.MultiHeadAttention(x1d)[0]
        x1d+=residual
        x1d=self.ln1(x1d)
        residual=x1d
        x1d=self.ffn(x1d)
        x1d+=residual
        x1d=self.ln2(x1d)
        return x1d


class Weight1dto2dMultiHeadAttention(nn.Module):
    def __init__(self, n_input, n_heads):
        super().__init__()
        self.nheads = n_heads
        self.d_k = n_input // n_heads
        self.toK = nn.Linear(n_input, n_input)
        self.toQ = nn.Linear(n_input, n_input)
        self.toV = nn.Linear(n_input, n_input)
        self.proj = nn.Linear(n_input, n_input)

    def forward(self, x1d, Attn2d=None):
        query = self.toQ(x1d).reshape(-1, self.nheads, self.d_k).transpose(0, 1)
        key = self.toK(x1d).reshape(-1, self.nheads, self.d_k).transpose(0, 1)
        value = self.toV(x1d).reshape(-1, self.nheads, self.d_k).transpose(0, 1)
        AttnScore = torch.matmul(query, key.transpose(-2, -1))

        AttnScore = AttnScore / math.sqrt(self.d_k)
        attn_W = AttnScore.softmax(dim=-1)
        x1d = torch.matmul(attn_W, value)
        x1d = (
            x1d.transpose(0, 1).reshape(-1, self.nheads * self.d_k)
        )
        x1d = self.proj(x1d)
        if Attn2d == None:
            return x1d, AttnScore
        else:
            AttnScore = torch.cat((Attn2d, AttnScore), dim=0)
            return x1d, AttnScore

class AttentionResNet(nn.Module):

    def __init__(self, n_input1d=256, n_head=8, n_attentionblock=6, n_attn1dto2dblock=8,
                 kernel_size2d=5, dilation2d=2, n_ResNet1D_block=4, n_ResNet2D_block=4,
                 dropout=0.0, activation='elu', normalization='batch', bias=False, **kwargs):
        super().__init__()
        n_input2d = 41 + n_head * n_attn1dto2dblock
        self.linear = nn.Linear(2560, 256)
        self.proj_1D = nn.Conv1d(n_input1d, n_input1d, kernel_size=1, bias=True)

        #self.ResNet1D_blocks = nn.ModuleList(
        #    [
        #        ResBlock1d(
        #            n_input1d, n_input1d, kernel_size=3, dilation=1, dropout=0,
        #            activation=activation, normalization=normalization,
        #            bias=bias) for _ in range(n_ResNet1D_block)
        #    ]
        #)

        self.attention1d = nn.ModuleList(
            [
                SeqAttentionBlock(n_input1d, n_head, n_input1d) for _ in range(n_attentionblock)
            ]
        )

        self.ResNet2D_blocks = nn.ModuleList(
            [
                ResBlock2d(n_input2d, kernel_size=3, dilation=1, dropout=0.0,
                           activation='elu', normalization='batch', bias=False) for _ in range(n_ResNet2D_block)
            ]
        )
        self.Attn1dto2dblock1 = Weight1dto2dMultiHeadAttention(n_input1d, n_head)
        self.n_Attn1dto2dblocks = nn.ModuleList(
            [
                Weight1dto2dMultiHeadAttention(n_input1d, n_head) for _ in range(n_attn1dto2dblock - 1)
            ]
        )
        self.activate = activation_func(activation)
        self.finalconv2d = nn.Sequential(
            nn.ReLU(),
            Conv2dAuto(n_input2d, 2, kernel_size=kernel_size2d, dilation=dilation2d, bias=bias)
        )
        self.Dropout=nn.Dropout(p=0.1)
        self.Norm=nn.InstanceNorm2d(n_input2d)
    def forward(self, x1d, x2d):

        x1d = self.linear(x1d)
        # x1d = x1d.permute(1, 0)
        # print(x1d.shape)
        # x1d = self.proj_1D(x1d)
        for block in self.attention1d:
            x1d = block(x1d)
        x1d = self.activate(x1d)
        # x1d=x1d.unsqueeze(1)
        # x1d=x1d.repeat(1,x1d.shape[1],1)
        # x1d = torch.cat((x1d, torch.transpose(x1d, 1, 2)), 0)
        x1d, AttnW = self.Attn1dto2dblock1(x1d)
        for block in self.n_Attn1dto2dblocks:
            x1d, AttnW = block(x1d, AttnW)
        AttnW=self.activate(AttnW)
        x = torch.cat((AttnW, x2d), dim=0)
        x = x.reshape(1, x.shape[0], x.shape[1], -1)
        x = self.Norm(x)
        for block in self.ResNet2D_blocks:
            x = block(x)
        x = self.finalconv2d(x)
        x=self.Dropout(x)
        return x


train = [x.strip() for x in open('C:/Users/wwz2000/study1/train_list.txt')]
data = h5py.File('C:/Users/wwz2000/study1/esm2_3B_targetEmbed.h5', 'r')
xyz = h5py.File('C:/Users/wwz2000/study1/xyz.h5', 'r')


finalnet=AttentionResNet()
finalnet.cuda()
optimizer = torch.optim.Adam(finalnet.parameters(), lr=0.0005)
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
        random.shuffle(valid_data)
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
        print(lossmean)
        epochloss.append(lossmean)
        torch.save(finalnet, f'C:/Users/wwz2000/study1/eachAttnNet{epoch}.pkl')  # 保存整个网络
        state = {'net':finalnet.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch,'meanloss':lossmean}
        torch.save(state, f'C:/Users/wwz2000/study1/AttnNetpara{epoch}.pth')
    return epochloss


import matplotlib.pyplot as plt
epochloss=train_func(10,train)
print(epochloss)
x=np.arange(0,10)
plt.plot(x,epochloss)
plt.show()
