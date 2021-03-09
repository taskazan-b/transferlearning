import torch
import torch.nn as nn


class SoS_loss(nn.Module):
    def __init__(self, d, mord=1):
        super(SoS_loss, self).__init__()
        self.mord = mord
        import scipy.special as sp
        s = int(sp.binom(mord+d,mord))
        self.s = s
        Ad = torch.zeros((s,s), requires_grad=False)
        for i in range(s):
            Ad[i][s-i-1] = 1
        self.Ad = Ad
        powers = []
        for i in range(0,mord+1):
            powers.append(self.exponent(i,d))
        self.powers = powers

    def exponent(self, i, d):

    def calcQ(self):
        pass

    def vmap(self):
        pass

    def mom(self, X):
        pass

  
    def forward(self, Ms, Mt, source_tr, target_tr):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
