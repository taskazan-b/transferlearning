import torch
import torch.nn as nn
import scipy.special as sp
import numpy as np

class SoS_loss(nn.Module):
    def __init__(self, d, mord=1, gpu_id=0):
        DEVICE = torch.device('cuda:%d'%gpu_id if torch.cuda.is_available() else 'cpu')
        self.gpu = DEVICE
        super(SoS_loss, self).__init__()
        self.mord = mord
        s = int(sp.binom(mord+d,mord))
        self.s = s
        Ad = torch.zeros((s,s), device = DEVICE, requires_grad=False)
        for i in range(s):
            Ad[i][s-i-1] = 1
        self.Ad = Ad
        powers = []
        for i in range(2,mord+1):
            powers.append(self.exponent(i,d))
        self.powers = powers
        self.Mj = None

    def exponent(self, n, K):
        # Credit: python implementation from the original MATLAb code of Rene Vidal, 2013
        idd = np.eye(K)
        exp = idd
        for i in range(2,n+1):
            rene = []
            for j in range(K):
                for k in range(exp.shape[0]-int(sp.binom(i+K-j-2,i-1)), exp.shape[0]):
                    rene.append(idd[j]+exp[k])
            exp = np.array(rene)
        return exp   

    def calcQ(self, V, x, rho):
        #  x: row vectors in veronese space
        # V is s-by-n where n is number of samples s is dimension
        Q = torch.diag(x @ torch.t(x) - x @ V @ \
        torch.inverse(rho*torch.eye(V.shape[1], device = self.gpu, requires_grad=False) + torch.t(V) @ V) \
        @ torch.t(V) @ torch.t(x))
        return Q

    def veronese(self, X, n, powers=None):
        if n==0:
            y = torch.ones((1,X.shape[1]), device = self.gpu, requires_grad=False)
        elif n==1:
            y = X
        else:
            if powers.any()==None:
                raise ValueError("powers cannot be None for mord>=2")
            X[torch.abs(X)<1e-10] = 1e-10
            y = torch.exp(torch.from_numpy(powers).to(self.gpu).type(torch.cuda.FloatTensor) @ torch.log(X))

        return y     

    def vmap(self, X):
        
        vx = torch.cat((self.veronese(torch.t(X), 0), self.veronese(torch.t(X),1)),0)
        # dtype = torch.cuda.FloatTensor
        # vx = torch.cat((self.veronese(torch.t(X), 0).type(dtype), self.veronese(torch.t(X),1).type(dtype)),0)

        p = 0
        for i in range(2,self.mord+1):
            vx = torch.cat((vx, self.veronese(torch.t(X), i, self.powers[p])),0)
            p+=1
        return vx    

    def rho_val(self, V):
        import math
        return torch.norm( torch.t(V) @ V )/(500*math.sqrt(V.shape[1]))

    def mom(self, X):
        Vx = self.vmap(X)
        rho = self.rho_val(Vx)
        Mx = rho * torch.eye(self.s, device = self.gpu, requires_grad=False) + (Vx @ torch.t(Vx)) / Vx.shape[1]
        return Mx

    def loss_squeeze(self, source, source_train, label_source):
        pass

    def forward(self, Ms, Mt, source_tr, target_tr):
        

        XX = torch.mean(kernels[:batch_size, :batch_size])
        YY = torch.mean(kernels[batch_size:, batch_size:])
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])
        loss = torch.mean(XX + YY - XY - YX)
        return loss
