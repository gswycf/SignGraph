from torch_geometric.nn import GCNConv, SAGEConv, EdgeConv
import torch.nn as nn
import torch
from timm.models.layers import DropPath
from einops import rearrange
import numpy as np
import random
import torch.nn.functional as F


def EucDis(x, y):
    # x =[b, c, t_1, n], y=[b, c,t_1,n]
    xy = torch.einsum('bctn,bctq->btnq', x, y)
    xx = torch.einsum('bctn,bctq->btnq', x, x)
    yy = torch.einsum('bctn,bctq->btnq', y, y)
    return (xx+yy-2*xy) #-dis=sim


def ForEucDis(x, y):
    with torch.no_grad():
        b, c, t, n = x.shape
        x = x.permute(0, 2,3, 1) #b t n c
        y = y.permute(0, 2,3, 1)
        x = x.reshape(b, t, n, c)
        y = y.reshape(b, t, n, c)
        return torch.cdist(x, y)




def CosDis(x, y):
    # x =[b, c, t_1, n], y=[b, c,t_1,n]
    xy = torch.einsum('bctn,bctq->btnq', x, y)
    xx = torch.einsum('bctn,bctq->btnq', x, x)
    yy = torch.einsum('bctn,bctq->btnq', y, y)
    return -xy/(torch.sqrt(xx)*torch.sqrt(yy))

def Chebyshev(x, y):
    x = torch.einsum('bctn,bctq->btnq', x, x)
    y = torch.einsum('bctn,bctq->btnq', y, y)
    return -torch.abs(x-y)
    # x = torch.einsum('bctn,bctq->bctnq', x, x)
    # y = torch.einsum('bctn,bctq->bctnq', y, y)
    # return torch.max(torch.abs(x-y), dim=1)


class TemporalGraph(nn.Module):
    def __init__(self, in_channels, k=4, drop_path=0.0):
        super(TemporalGraph, self).__init__()
        self.k = k
        self.reduction_channel = in_channels
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, self.reduction_channel, kernel_size=(3,1,1), bias=False,padding=(1,0,0)),
            nn.BatchNorm3d(self.reduction_channel)
        )
        self.up_conv = nn.Sequential(
            nn.Conv3d(in_channels, self.reduction_channel, kernel_size=(3,1,1), bias=False,padding=(1,0,0)),
            nn.BatchNorm3d(self.reduction_channel)
        )
        self.gconv = GCNConv(self.reduction_channel, self.reduction_channel)
        # self.gconv = SAGEConv(self.reduction_channel, self.reduction_channel)

    def forward(self, x, batch):
        tlen, c, h, w = x.shape
        x = rearrange(x.view(batch, tlen // batch, c, h, w), "b v c h w-> b c v h w")
        x = self.down_conv(x)
        x = rearrange(x, "b c v h w-> b c v (h w)")
        x1, x2 = x[:, :, :-1, :], x[:,:,1:,:] # b c t-1 hw
        sim = -ForEucDis(x1, x2)
        b, t_1, hw, hw = sim.shape
        sim = F.normalize(sim.view(b,t_1, -1), dim=-1)
        sim = torch.where(sim < 0.05, 100, sim)
        _, topk_indices = torch.topk(sim, k=self.k)
        row_indices,col_indices = topk_indices // hw, topk_indices % hw
        finaledge = torch.zeros((b, t_1, self.k, 2), dtype=torch.int)
        for i in range(t_1):
            finaledge[:,i, :, 0] = row_indices[:, i,:]+ i* hw
            finaledge[:,  i, :, 1] = col_indices[:, i, :] + (i+1) * hw
        finaledge = finaledge.view(b, t_1*self.k, 2)
        finaledge_re = torch.stack((finaledge[:,:,1], finaledge[:,:,0]), dim=-1)
        # torch.save(finaledge, "./work_dir/test/temproal.pt")

        finaledge = torch.cat((finaledge, finaledge_re), dim=1).permute(0,2,1).detach()
        x = rearrange(x, "b c v n-> b (v n) c")
        out = torch.zeros_like(x).to(x.device)
        for i in range(batch):
            out[i] = self.gconv(x[i], finaledge[i].to(x.device))
        x = out.permute(0,2,1).view(b, self.reduction_channel, tlen//b, h, w)
        x = self.up_conv(x).permute(0, 2, 1, 3, 4).contiguous().view(tlen, c, h, w)
        return x


    # class TemporalGraph(nn.Module):
#     def __init__(self, in_channels, k=4, drop_path=0.0):
#         super(TemporalGraph, self).__init__()
#         self.graph_conv = GrapherT(in_channels=in_channels, kernel_size=k, dilation=1, conv='edge', act='relu', norm=None,
#                  bias=True,  stochastic=False, epsilon=0.0, r=1, drop_path=0.0, relative_pos=False)
#
#     def forward(self, x, batch):
#         tlen, c, h, w =x.shape
#         x = x.view(batch, -1, c, h, w)
#         b, v, c, h, w, =x.shape
#         x = rearrange(x, "b v c h w-> b c v (h w)")
#         x = self.graph_conv(x)
#         x = x.view(b, c, v, h,w)
#         x = rearrange(x, "b c v h w -> (b v) c h w")
#         return x

def test_new():
    # x = torch.rand([78, 256, 14, 14]).to("cuda:1")
    # model = TemporalGraph(k=4, in_channels=256, drop_path=0).to("cuda:1")
    # y = model(x, 2)
    # print(y.shape)
    x= torch.ones([1,2,3, 4])+1
    y =  torch.ones([1,2,3, 4]) +2
    print(x, y)
    print(F.softmax(sim(x,y)))

if __name__ == '__main__':
    test_new()
    # model = TemporalGraph(k=4, in_channels=32, drop_path=0)
    # torch.manual_seed(1)
    # seed=2
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # x = torch.rand([6,32,3,3])
    # # print(x)
    # x_len = [3,2]
    #
    # y = model(x, 2)
    #
    # print(y.shape)
