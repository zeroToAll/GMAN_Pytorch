import torch.nn as nn
import torch
from myModule import STEmbedding,STAttBlock,transformAttention
from utils import FC
import torch.nn.functional as F


class GMAN(nn.Module):
    '''
    GMAN
    X：       [batch_size, P, N]
    TE：      [batch_size, P + Q, 2] (time-of-day, day-of-week)
    SE：      [N, K * d]
    P：       number of history steps
    Q：       number of prediction steps
    T：       one day is divided into T steps
    L：       number of STAtt blocks in the encoder/decoder
    K：       number of attention heads
    d：       dimension of each attention head outputs
    return：  [batch_size, Q, N]
    '''
    def __init__(self,T,K,d,SE_dim,STE_Q_dim,STE_P_dim,x_dim,P,Q,L,batch_size):
        #STE_Q_dim, STE_P_dim, X_dim, k, d, batch_size

        super(GMAN, self).__init__()



        self.T = T
        self.SE_dim = SE_dim
        self.K = K
        self.d = d

        self.D = K*d

        self.P = P
        self.Q = Q

        self.x_dim = x_dim
        self.batch_size = batch_size

        self.STE_Q_dim = STE_Q_dim
        self.STE_P_dim = STE_P_dim


        self.input_mapping = FC(self.x_dim,[self.D,self.D],[F.relu,None])

        self.STE_ = STEmbedding(self.T,self.D,self.SE_dim)

        encoder_list = []
        for _ in range(L):
            STABlock = STAttBlock(self.K,self.d,self.D,self.batch_size)
            encoder_list.append(STABlock)

        self.encoder_module_list = nn.ModuleList(encoder_list)

        self.transformAtt = transformAttention(self.STE_Q_dim,self.STE_P_dim,self.D,self.K,self.b,self.batch_size)

        decoder_list = []
        for _ in range(L):
            STABlock = STAttBlock(self.K,self.d,self.D,self.batch_size)
            decoder_list.append(STABlock)

        self.decoder_module_list = nn.ModuleList(decoder_list)

        self.out_mapping = FC(self.D,[self.D,1],[F.relu,None])




    def forward(self,X,TE,SE):
        ##### TE是day-of-week,time-of-day的one-hot 拼接后的结果

        X = self.input_mapping(X)

        STE = self.STE_(SE,TE)

        STE_P = STE[:,:self.P]  ####输入前P步的时空嵌入
        STE_Q = STE[:,self.P:]  ####输出后Q步的时空嵌入

        ##### encoder ######
        for i, layer in enumerate(self.encoder_module_list):
            X = layer(X,STE_P)

        X = self.transformAtt(X,STE_P,STE_Q)

        ### decoder #######
        for i, layer in enumerate(self.decoder_module_list):
            X = layer(X,STE_Q)

        X = self.out_mapping(X)

        return torch.squeeze(X,dim=3)



