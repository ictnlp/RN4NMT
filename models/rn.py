import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import wargs

class RelationLayer(nn.Module):

    def __init__(self, d_in, d_out, fltr_windows, d_fltr_feats, d_mlp=128):

        super(RelationLayer, self).__init__()

        self.n_windows = len(fltr_windows)
        assert len(d_fltr_feats) == self.n_windows, 'Require same number of windows and features'
        self.cnnlayer = nn.ModuleList(
            [
                nn.Conv1d(in_channels=1,
                          out_channels=d_fltr_feats[k],
                          kernel_size=d_in * fltr_windows[k],
                          stride=d_in,
                          padding=( (fltr_windows[k] - 1) / 2 ) * d_in
                         ) for k in range(self.n_windows)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(d_fltr_feats[k]) for k in range(self.n_windows)])
        self.leakyRelu = nn.LeakyReLU(0.1)

        self.d_sum_fltr_feats = sum(d_fltr_feats)

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.d_sum_fltr_feats, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1)
        )

        self.mlp_layer = nn.Sequential(
            nn.Linear(d_mlp, d_mlp),
            nn.LeakyReLU(0.1),
            nn.Linear(d_mlp, d_out),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x, xs_mask=None):

        L, B, E = x.size()
        if xs_mask is not None: x = x * xs_mask[:, :, None]
        x = x.permute(1, 0, 2)    # (B, L, d_in)

        ''' CNN Layer '''
        # (B, L, d_in) -> (B, d_in * L) -> (B, 1, d_in * L)
        x = x.contiguous().view(B, -1)[:, None, :]
        # (B, 1, d_in * L) -> [ (B, d_fltr_feats[k], L) ]
        x = [self.leakyRelu(self.bns[k](self.cnnlayer[k](x))) for k in range(self.n_windows)]
        #x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # [ (B, d_fltr_feats[k], L) ] -> (B, d_sum_fltr_feats, L)
        x = tc.cat(x, dim=1)
        # (B, d_sum_fltr_feats, L) -> (L, B, d_sum_fltr_feats)
        x = x.permute(2, 0, 1)

        ''''' Graph Propagation Layer '''''
        # (in: (L, B, d_sum_fltr_feats))
        if xs_mask is not None: x = x * xs_mask[:, :, None]
        x1 = x[:, None, :, :].repeat(1, L, 1, 1)
        x2 = x[None, :, :, :].repeat(L, 1, 1, 1)
        x = tc.cat([x1, x2], dim=-1)    # (L, _L, B, 2 * d_sum_fltr_feats)
        x = self.mlp(x)
        if xs_mask is not None: x = x * xs_mask[:, None, :, None]
        # (L, _L, B, 2 * d_sum_fltr_feats) -> (L, B, 2 * d_sum_fltr_feats)
        x = x.mean(dim=1)

        ''' MLP Layer '''
        x = self.mlp_layer(x)
        if xs_mask is not None: x = x * xs_mask[:, :, None]

        return x



