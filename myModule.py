import torch
import torch.nn as nn
from utils import FC
import torch.nn.functional as F
from utils import FC
#####数据的输入各维度和原始论文保持一致，在模型里面进行变换
class STEmbedding(nn.Module):

    def __init__(self,T,D,SE_dim):
        '''
        spatio-temporal embedding
        SE:     [N, D]          D是监测器的关系图用node2vec嵌入的维度，N是多少个监测点
        TE:     [batch_size, P + Q, T+7]
        T:      num of time steps in one day
        D:      output dims
        retrun: [batch_size, P + Q, N, D]
        '''
        super(STEmbedding, self).__init__()

        self.T = T
        self.D = D



        self.SE_dim = SE_dim

        self.SE_mapping = FC(self.SE_dim, [self.D, self.D], [F.relu, None])
        self.TE_embedding = FC(self.T+7,[self.D,self.D],activations=[F.relu,None])

    def forward(self, SE, TE_onehot):



        #### spatial embedding,在FC的时候做了Permute所以这里不用，保持
        #SE = SE.permute(0,3,1,2)
        SE = self.SE_mapping(SE)

        #### temporal embedding
        #TE = TE_onehot.permute(0,3,1,2)
        TE = self.TE_embedding(TE_onehot) #输出是(Batch,in_channel,in_h,in_w)

        return TE+SE


class spatialAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention head outputs
    return: [batch_size, num_step, N, D]
    '''
    def __init__(self,K,d,x_dim,batch_size):
        super(spatialAttention, self).__init__()
        self.D = K*d
        self.x_dim = x_dim
        self.K = K
        self.d = d
        self.batch_size = batch_size

        self.query_mapping = FC(self.x_dim,output_channels=self.D,activations=F.relu)
        self.key_mapping = FC(self.x_dim,self.D,F.relu)
        self.value_mapping = FC(self.x_dim,self.D,F.relu)

        self.out_mapping = FC(self.D,[self.D,self.D],[F.relu,None])


    def forward(self, X, STE):
        X = torch.cat((X,STE),dim=-1)
        query = self.query_mapping(X).permute(0,2,3,1)
        key = self.key_mapping(X).permute(0,2,3,1)
        value = self.value_mapping(X).permute(0,2,3,1) #切分之前 [batch_size,num_step,N,K*d]


        query = torch.cat(torch.split(query,self.d,dim=-1),dim=0) #切分之后 [K*batch_size,num_step,N,d]
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        key = key.permute((0,1,3,2))
        #query: [K*batch_size,num_step,N,d]
        #key:   [K*batch_size,num_step,d,N]
        #value: [K*batch_size,num_step,N,d]
        attention = torch.matmul(query,key) #[K*batch_size,num_step,N,N]
        attention /= (self.d**0.5)
        attention = F.softmax(attention,dim=-1)
        X = torch.matmul(attention,value)

        X = torch.cat(torch.split(X,self.batch_size,dim=0),dim=-1)

        X = self.out_mapping(X) #输出是(Batch,in_channel,in_h,in_w)

        return X


class temporalAttention(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    def __init__(self,K,d,x_dim,batch_size):
        super(temporalAttention, self).__init__()
        self.D = K * d
        self.x_dim = x_dim #x_dim是原始X的dim与STE的dim的拼接
        self.K = K
        self.d = d
        self.batch_size = batch_size

        #x_dim是原始X的dim与STE的dim的拼接
        self.query_mapping = FC(self.x_dim, output_channels=self.D, activations=F.relu)
        self.key_mapping = FC(self.x_dim, self.D, F.relu)
        self.value_mapping = FC(self.x_dim, self.D, F.relu)

        self.out_mapping = FC(self.D, [self.D, self.D], [F.relu, None])

    def forward(self, X,STE):
        X = torch.cat((X, STE), dim=-1)
        query = self.query_mapping(X).permute(0, 2, 3, 1)
        key = self.key_mapping(X).permute(0, 2, 3, 1)
        value = self.value_mapping(X).permute(0, 2, 3, 1)  # 切分之前 [batch_size,num_step,N,K*d]


        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)  # 切分之后 [K*batch_size,num_step,N,d]
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)

        query = query.permute(0,2,1,3)
        key = key.permute(0,2,3,1)
        value = value.permute(0,2,1,3)
        ###调整后的个维度如下
        # query: [K * batch_size, N, num_step, d]
        # key:   [K * batch_size, N, d, num_step]
        # value: [K * batch_size, N, num_step, d]

        attention = torch.matmul(query,key) #[K * batch_size, N, num_step, num_step]
        attention /= (self.d**0.5)

        #### 输入的信息不回暴露后面的预测信息，这里不做mask了
        attention = torch.softmax(attention,dim=-1)

        X = torch.matmul(attention,value)
        X = X.permute(0,2,1,3)

        X = torch.cat(torch.split(X,self.batch_size,dim=0),dim=-1)

        X = self.out_mapping(X) #(Batch,in_channel,in_h,in_w)

        return X

class gateFusion(nn.Module):
    '''
    gated fusion
    HS:     [batch_size, num_step, N, D]
    HT:     [batch_size, num_step, N, D]
    D:      output dims
    return: [batch_size, num_step, N, D]
    '''
    def __init__(self,XS_dim,XT_dim,D):
        super(gateFusion, self).__init__()

        self.XS_dim = XS_dim
        self.XT_dim = XT_dim
        self.D = D
        self.XS_mapping = FC(self.XS_dim,D,activations=None)
        self.XT_mapping = FC(self.XT_dim,D,activations=None)
        self.out_mapping = FC(self.D,[self.D,self.D],activations=[F.relu,None])


    def forward(self,XS,XT):
        ZS = self.XS_mapping(XS)
        ZT = self.XT_mapping(XT)

        z = torch.sigmoid(ZS+ZT).permute(0, 2, 3, 1)
        H = torch.mul(z,XS) + torch.mul(1-z,XT)

        H = self.out_mapping(H) #输出是(Batch,in_channel,in_h,in_w)

        return H


class transformAttention(nn.Module):
    '''
    transform attention mechanism
    X:      [batch_size, P, N, D]
    STE_P:  [batch_size, P, N, D]
    STE_Q:  [batch_size, Q, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, Q, N, D]
    '''
    def __init__(self,STE_Q_dim,STE_P_dim,X_dim,k,d,batch_size):
        super(transformAttention, self).__init__()

        self.STE_Q_dim = STE_Q_dim
        self.STE_P_dim = STE_P_dim
        self.X_dim = X_dim
        self.k = k
        self.d = d
        self.D = k*d
        self.batch_size = batch_size

        self.query_mapping = FC(self.STE_Q_dim,self.D,F.relu)
        self.key_mapping = FC(self.STE_P_dim,self.D,F.relu)
        self.value_mapping = FC(self.X_dim,self.D,F.relu)
        self.out_mapping = FC(self.D,[self.D,self.D],[F.relu,None])



    def forward(self,X,STE_P,STE_Q):

        # query: [batch_size, Q, N, K * d]
        # key:   [batch_size, P, N, K * d]
        # value: [batch_size, P, N, K * d]
        query = self.query_mapping(STE_Q).permute(0, 2, 3, 1)
        key = self.key_mapping(STE_P).permute(0, 2, 3, 1)
        value = self.value_mapping(X).permute(0, 2, 3, 1)

        # query: [K * batch_size, Q, N, d]
        # key:   [K * batch_size, P, N, d]
        # value: [K * batch_size, P, N, d]
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)  # 切分之后 [K*batch_size,num_step,N,d]
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)

        # query: [K * batch_size, N, Q, d]
        # key:   [K * batch_size, N, d, P]
        # value: [K * batch_size, N, P, d]
        query = query.permute(0,2,1,3)
        key = key.permute(0,2,3,1)
        value = value.permute(0,2,1,3)

        attention = torch.matmul(query,key)
        attention /= (self.d**0.5)
        attention = torch.softmax(attention,dim=-1)

        X = torch.matmul(attention,value).permute(0,2,1,3)
        X = torch.cat(torch.split(X,self.batch_size,dim=0),dim=-1)

        X = self.out_mapping(X)

        return X

class STAttBlock(nn.Module):
    def __init__(self,K,d,x_dim,batch_size):
        super(STAttBlock, self).__init__()
        self.K = K
        self.d = d
        self.x_dim = x_dim
        self.batch_size = batch_size
        self.D = K*d

        self.SA = spatialAttention(self.K,self.d,self.x_dim,self.batch_size)
        self.TA = temporalAttention(self.K,self.d,self.x_dim,self.batch_size)
        self.GF = gateFusion(self.D,self.D,self.D)


    def forward(self, X,STE):
        HS = self.SA(X,STE)
        HT = self.TA(X,STE)
        H = self.GF(HS,HT)

        return (X+H)





if __name__ == '__main__':
    pass

    ###########对 STEmbedding的测试 ########## start
    # SE = torch.randn((1,1,325,64))
    # TE = torch.randn((10,24,1,295)) #(Batch,in_h,in_w,in_channel)
    #
    # eb = STEmbedding(288,64,64)
    #
    # x = eb(SE,TE)  #输出是(Batch,in_channel,in_h,in_w)
    # print(x.shape)


    ########## 测试 spatialAttention #########
    # sta = spatialAttention(8,8,128,10)
    #
    # x = torch.randn((10,12,325,64))
    # STE = torch.randn((10,12,325,64))
    #
    # result = sta(x,STE) #输出是(Batch,in_channel,in_h,in_w)
    # print(result.shape)


    ########## 测试 temporalAttention ########
    # tra = temporalAttention(8,8,128,10)
    # x = torch.randn((10,12,325,64))
    # STE = torch.randn((10,12,325,64))
    # result = tra(x,STE)
    # print(result.shape)

    ####### 测试  gateFusion ##########
    # gf = gateFusion(64,64,64)
    # XS = torch.randn((10,12,325,64))
    # XT = torch.randn((10, 12, 325, 64))
    # result = gf(XS,XT)
    # print(result.shape) #输出是(Batch,in_channel,in_h,in_w)

    ######### 测试 transformAttention #####

    STE_P = torch.randn((10,12,325,64))
    STE_Q = torch.randn((10,12,325,64))
    X = torch.randn((10,12,325,64))

    tfa = transformAttention(64,64,64,8,8,10)
    result = tfa(X,STE_P,STE_Q) #(self,X,STE_P,STE_Q)









