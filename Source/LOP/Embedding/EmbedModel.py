import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from LOP.Embedding.DenseCNNFunc import singleLayer, addTransition, addTransitionBack


# Attention layer
class ChordLevelAttention(nn.Module):
    # this follows the word-level attention from Yang et al. 2016
    # https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
    def __init__(self, n_hidden, batch_first=True):
        super(ChordLevelAttention, self).__init__()
        self.mlp = nn.Linear(n_hidden, n_hidden)
        self.u_w = nn.Parameter(torch.rand(n_hidden))
        self.batch_first = batch_first

    def forward(self, X):
        if not self.batch_first:
            # make the input (batch_size, timesteps, features)
            X = X.transpose(1, 0)
        # get the hidden representation of the sequence
        u_it = F.tanh(self.mlp(X))
        # get attention weights for each timestep
        alpha = F.softmax(torch.matmul(u_it, self.u_w))
        # get the weighted sum of the sequence
        out = torch.sum(torch.matmul(alpha, X), dim=1)
        return out, alpha



# Network Classic CNN
class embedClassicNet(torch.nn.Module):
    def __init__(self, nFilters, KerSize, fullyhiddensize, Dim_embed, lstmhiddensize, nreclayer, attention, gpu):
        super(embedClassicNet, self).__init__()
        # encoder
        self.gpu = gpu
        self.attention = attention
        self.conv1 = nn.Conv1d(1, nFilters, KerSize, stride=1, padding=6, bias=True)
        self.batchnorm = nn.BatchNorm1d(nFilters, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(128 * nFilters, fullyhiddensize[0])
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(fullyhiddensize[0], fullyhiddensize[1])
        self.relu3 = nn.ReLU()
        self.linear3 = nn.Linear(fullyhiddensize[1], Dim_embed)
        # decoder
        if attention:
            self.attn = ChordLevelAttention(Dim_embed)
            if gpu:
                self.attn = self.attn.cuda()
        self.LSTM1 = nn.LSTM(Dim_embed, lstmhiddensize, nreclayer, batch_first=True)
        self.linear4 = nn.Linear(lstmhiddensize, Dim_embed)
        self.deconv1 = nn.ConvTranspose1d(nFilters, 1, KerSize, padding=5, stride=1, bias=True)

    def forward(self, x, rho):
        """
        Forward pass of the network.
        Batch size N must be greater than rho (seq length)
        Input : batch of 128-dimensionnal piano-roll vectors (N, 1, 128)
        Output : batch of prediction (N, 1, 128)
        """
        if rho > 0:
            # encoder
            outconv = self.conv1(x)                                                         # In : (N,1,128), Out : (N, nFilters, 129)
            outnorm = self.batchnorm(outconv[:,:,:-1].contiguous())                         # Out : (N, nFilters, 128)
            outrelu = self.relu1(outnorm)                                                   # Out : (N, nFilters, 128)
            outlin1 = self.linear1(outrelu.view(x.size(0), self.linear1.in_features))       # In : (N, 128 * nFilters), Out : (N, fullyhiddensize(0))
            outrelu2 = self.relu2(outlin1)
            outlin2 = self.linear2(outrelu2)                                                # Out : (N, fullyhiddensize(1))
            outrelu3 = self.relu3(outlin2)
            outfin = self.linear3(outrelu3)                                                 # Out : (N, Dim_embed)
            # decoder
            decod_in = Variable(torch.Tensor(x.size(0)-rho+1, rho, outfin.size(1)))
            if self.gpu:
                decod_in = decod_in.cuda()
            for i in range(x.size(0)-rho+1):                                                  # take sequence of size rho. From this point N = N - rho
                decod_in[i, :, :] = outfin[i:i+rho, :]
            if self.attention:
                outputAttn, attn_weights = self.attn(decod_in)
                outlstm = self.LSTM1(outputAttn.view(outputAttn.size(0), 1, outputAttn.size(1)))
            else :
                outlstm = self.LSTM1(decod_in)                                              # In : (N, rho, Dim_embed), Out : (N, rho, lstmhiddensize)
            outlin4 = self.linear4(outlstm[0])                                              # Out : (N, rho, Dim_embed)
            outinv1 = torch.matmul(outlin4[:,-1,:], self.linear3.weight)                    # invert linear 3 - In : (N, Dim_embed), Out : (N, fullyhiddensize[1])
            outinv2 = torch.matmul(outinv1, self.linear2.weight)                            # Out : (N, 1, fullyhiddensize[0])
            outinv3 = torch.matmul(outinv2, self.linear1.weight)                            # Out : (N, 1, 128 * nFilters)
            out = self.deconv1(outinv3.view(x.size(0)-rho, self.deconv1.in_channels, 128))  # In : (N, nFilters, 128), Out : (N, 1, 129)
            out = F.sigmoid(out[:,:,:-1])
            return out, outfin.data, attn_weights if self.attention else None               # Out : batch of prediction (N, 1, 128)
        else :
            # encoder
            outconv = self.conv1(x)                                                         # In : (N,1,128), Out : (N, nFilters, 129)
            outnorm = self.batchnorm(outconv[:,:,:-1].contiguous())                         # Out : (N, nFilters, 128)
            outrelu = self.relu1(outnorm)                                                   # Out : (N, nFilters, 128)
            outlin1 = self.linear1(outrelu.view(x.size(0), self.linear1.in_features))       # In : (N, 128 * nFilters), Out : (N, fullyhiddensize(0))
            outrelu2 = self.relu2(outlin1)
            outlin2 = self.linear2(outrelu2)                                                # Out : (N, fullyhiddensize(1))
            outrelu3 = self.relu3(outlin2)
            outfin = self.linear3(outrelu3)                                                 # Out : (N, Dim_embed)
            return outfin.data



# Network Dense CNN
class embedDenseNet(torch.nn.Module):
    def __init__(self, nFilters, KerSize, fullyhiddensize, Dim_embed, lstmhiddensize, nreclayer, nblock,
                 growthRate, reduction, dropout, attention, gpu):
        super(embedDenseNet, self).__init__()
        # encoder
        self.gpu = gpu
        self.nblock = nblock
        self.attention = attention
        self.initialConv = nn.Conv1d(1, nFilters, KerSize, stride=1, padding=6, bias=True)
        for i in range(nblock):
            singleLayer(self, nFilters, nFilters + growthRate, dropout, i)
            nFilters = nFilters + growthRate
        addTransition(self, nFilters, int(math.floor(nFilters * reduction)), dropout)
        nFilters = int(math.floor(nFilters * reduction))
        self.finalbatchnorm = nn.BatchNorm1d(nFilters)
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(64 * nFilters, fullyhiddensize[0])
        self.reluL1 = nn.ReLU()
        self.linear2 = nn.Linear(fullyhiddensize[0], fullyhiddensize[1])
        self.reluL2 = nn.ReLU()
        self.linear3 = nn.Linear(fullyhiddensize[1], Dim_embed)
        # decoder with attention mechanism or not
        if attention:
            self.attn = ChordLevelAttention(Dim_embed)
            if gpu:
                self.attn = self.attn.cuda()
        self.LSTM1 = nn.LSTM(Dim_embed, lstmhiddensize, nreclayer, batch_first=True)
        self.linear4 = nn.Linear(lstmhiddensize, Dim_embed)
        addTransitionBack(self, nFilters, int(math.floor(nFilters / reduction)), dropout)
        nFilters = int(math.floor(nFilters / reduction))
        for i in range(nblock):
            singleLayer(self, nFilters, nFilters + growthRate, dropout, i + nblock)
            nFilters = nFilters + growthRate
        self.deconv1 = nn.ConvTranspose1d(nFilters, 1, KerSize, padding=5, stride=1, bias=True)

    def forward(self, x, rho):
        """
        Forward pass of the network.
        Batch size N must be greater than rho (seq length)
        Input : batch of 128-dimensionnal piano-roll vectors (N, 1, 128)
        Output : batch of prediction (N, 1, 128)
        """
        if rho > 0:
            # encoder
            outconv = self.initialConv(x)
            for i in range(self.nblock):
                b = getattr(self, 'batchnorm' + str(i))
                r = getattr(self, 'relu' + str(i))
                c = getattr(self, 'conv' + str(i))
                outconv = b(outconv[:,:,:-1].contiguous())
                outconv = r(outconv)
                outconv = c(outconv)
                if hasattr(self, 'dropoutT'):
                    d = getattr(self, 'dropout' + str(i))
                    outconv = d(outconv)
            outtrans = self.batchnormT(outconv[:,:,0:127].contiguous())
            outtrans = self.reluT(outtrans)
            outtrans = self.convT(outtrans)
            if hasattr(self, 'dropoutT'):
                outtrans = self.dropoutT(outtrans)
            outtrans, pool_ind = self.poolingT(outtrans[:,:,:-1])
            outfin = self.finalbatchnorm(outtrans.contiguous())
            outfin = self.tanh(outfin)
            outfin = self.linear1(outfin.view(x.size(0), self.linear1.in_features))
            outfin = self.reluL1(outfin)
            outfin = self.linear2(outfin)
            outfin = self.reluL2(outfin)
            outfin = self.linear3(outfin)
            # decoder
            decod_in = Variable(torch.Tensor(x.size(0)-rho+1, rho, outfin.size(1)))
            if self.gpu:
                decod_in = decod_in.cuda()
            for i in range(x.size(0)-rho+1):                                          # take sequence of size rho. From this point N = N - rho
                decod_in[i, :, :] = outfin[i:i+rho, :]
            if self.attention:
                outputAttn, attn_weights = self.attn(decod_in)
                outlstm = self.LSTM1(outputAttn.view(outputAttn.size(0), 1, outputAttn.size(1)))
            else :
                outlstm = self.LSTM1(decod_in)
            outlin4 = self.linear4(outlstm[0])
            outinv1 = torch.matmul(outlin4[:,-1,:], self.linear3.weight)
            outinv2 = torch.matmul(outinv1, self.linear2.weight)
            outinv3 = torch.matmul(outinv2, self.linear1.weight)
            if sys.version_info[0] < 3:
                outtransB = self.unpoolTB(outinv3.view(x.size(0)-rho+1, (self.linear1.in_features / 64), 64), pool_ind[rho-1:,:,:])
            else :
                outtransB = self.unpoolTB(outinv3.view(x.size(0)-rho+1, (self.linear1.in_features // 64), 64), pool_ind[rho-1:,:,:])
            outtransB = self.batchnormTB(outtransB.contiguous())
            outtransB = self.reluTB(outtransB)
            outtransB = self.convTB(outtransB)
            if hasattr(self, 'dropoutTB'):
                outtransB = self.dropoutTB(outtransB)
            for i in range(self.nblock):
                b = getattr(self, 'batchnorm' + str(i + self.nblock))
                r = getattr(self, 'relu' + str(i + self.nblock))
                c = getattr(self, 'conv' + str(i + self.nblock))
                outtransB = b(outtransB[:,:,:-1].contiguous())
                outtransB = r(outtransB)
                outtransB = c(outtransB)
                if hasattr(self, 'dropoutT'):
                    d = getattr(self, 'dropout' + str(i))
                    outtransB = d(outtransB)
            out = self.deconv1(outtransB[:,:,0:127])
            out = F.sigmoid(out.view(out.size(0), out.size(2)))
            return out.view(out.size(0), 1, out.size(1)), outfin.data, attn_weights if self.attention else None
        else :
            # encoder
            outconv = self.initialConv(x)
            for i in range(self.nblock):
                b = getattr(self, 'batchnorm' + str(i))
                r = getattr(self, 'relu' + str(i))
                c = getattr(self, 'conv' + str(i))
                outconv = b(outconv[:,:,:-1].contiguous())
                outconv = r(outconv)
                outconv = c(outconv)
                if hasattr(self, 'dropoutT'):
                    d = getattr(self, 'dropout' + str(i))
                    outconv = d(outconv)
            outtrans = self.batchnormT(outconv[:,:,0:127].contiguous())
            outtrans = self.reluT(outtrans)
            outtrans = self.convT(outtrans)
            if hasattr(self, 'dropoutT'):
                outtrans = self.dropoutT(outtrans)
            outtrans, pool_ind = self.poolingT(outtrans[:,:,:-1])
            outfin = self.finalbatchnorm(outtrans.contiguous())
            outfin = self.tanh(outfin)
            outfin = self.linear1(outfin.view(x.size(0), self.linear1.in_features))
            outfin = self.reluL1(outfin)
            outfin = self.linear2(outfin)
            outfin = self.reluL2(outfin)
            outfin = self.linear3(outfin)
            return outfin.data