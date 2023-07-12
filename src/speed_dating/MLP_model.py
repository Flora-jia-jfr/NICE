import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd

# Define and instantiate the model
# revised to be able to replace the two branches with only one branch for estimating treatment effect
class MLP(nn.Module):
    def __init__(self, dim, flags):
        super(MLP, self).__init__()
        hidden_dim = flags.hidden_dim # 250
        hypo_dim = 100
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.lin1_1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin1_2 = nn.Linear(hidden_dim, hypo_dim)
        self.flags = flags
        # tarnet
        if self.flags.net == 'tarnet' or self.flags.net == 'tarnet_single':
            self.lin1_3 = nn.Linear(dim, 1)
        elif self.flags.net == 'dragon':
            self.lin1_3 = nn.Linear(hypo_dim, 1)
        else:
            print(self.flags.net)
            raise NotImplementedError("No implementation for this model")

        self.lin2_0 = nn.Linear(hypo_dim, hypo_dim)
        self.lin2_1 = nn.Linear(hypo_dim, hypo_dim)

        self.lin3_0 = nn.Linear(hypo_dim, hypo_dim)
        self.lin3_1 = nn.Linear(hypo_dim, hypo_dim)

        self.lin4_0 = nn.Linear(hypo_dim, 1)
        self.lin4_1 = nn.Linear(hypo_dim, 1)

        all_lin_layers = [self.lin1, self.lin1_1, self.lin1_2, self.lin2_0, self.lin2_1, self.lin1_3, self.lin3_0,
                    self.lin3_1, self.lin4_0, self.lin4_1]

        if self.flags.net =='tarnet_single':
            # add treatment as a node into the NN -- like S-learner
            self.lins_1 = nn.Linear(hypo_dim + 1, 2 * hypo_dim)
            self.lins_2 = nn.Linear(2 * hypo_dim, 2 * hypo_dim)
            self.lins_h = nn.Linear(2 * hypo_dim, 1)
            all_lin_layers = [self.lin1, self.lin1_1, self.lin1_2, self.lin2_0, self.lin2_1, self.lin1_3, self.lin3_0,
                    self.lin3_1, self.lin4_0, self.lin4_1, self.lins_1, self.lins_2, self.lins_h]

        for lin in all_lin_layers:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, input):
        # print("input: ", input.shape) # (1500, 185) # (4500, 185)
        initial = input.view(input.shape)
        # print("initial: ", initial.shape) # (1500, 185) # (4500, 185)
        x = F.relu(self.lin1(initial))
        x = F.relu(self.lin1_1(x))
        x = F.relu(self.lin1_2(x))
        x = F.relu(x)
        if self.flags.net == 'tarnet' or self.flags.net == 'tarnet_single':
            t = self.lin1_3(initial)
        else:
            t = self.lin1_3(x)
        if self.flags.net == 'tarnet_single':
            # print(x.shape) # (1500, 100)
            if self.flags.gpu == 1:
                x_t1 = torch.cat((x, torch.ones(x.shape[0], 1).cuda()), dim=1) # (1500, 101)
                x_t0 = torch.cat((x, torch.zeros(x.shape[0], 1).cuda()), dim=1) # (1500, 101)
            else:
                x_t1 = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1) # (1500, 101)
                x_t0 = torch.cat((x, torch.zeros(x.shape[0], 1)), dim=1) # (1500, 101)
            # print("x_t1: ", x_t1.shape, x_t1[:, -1])
            # print("x_t0: ", x_t0.shape, x_t0[:, -1])
            x_t1 = F.relu(self.lins_1(x_t1))
            x_t0 = F.relu(self.lins_1(x_t0))

            x_t1 = F.relu(self.lins_2(x_t1))
            x_t0 = F.relu(self.lins_2(x_t0))

            h1 = self.lins_h(x_t1)
            h0 = self.lins_h(x_t0)
        else:
            h1 = F.relu(self.lin2_1(x))
            h0 = F.relu(self.lin2_0(x))

            h1 = F.relu(h1)
            h0 = F.relu(h0)

            h0 = F.relu(self.lin3_0(h0))
            h1 = F.relu(self.lin3_1(h1))

            h0 = self.lin4_0(h0)
            h1 = self.lin4_1(h1)

        return torch.cat((h0, h1, t), 1) # (data_num, 3)