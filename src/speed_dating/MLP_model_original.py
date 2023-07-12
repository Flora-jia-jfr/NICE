import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd

# Define and instantiate the model
class MLP(nn.Module):
    def __init__(self, dim, flags):
        super(MLP, self).__init__()
        hidden_dim = flags.hidden_dim
        hypo_dim = 100
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.lin1_1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin1_2 = nn.Linear(hidden_dim, hypo_dim)
        # tarnet
        if flags.net =='tarnet':
            self.lin1_3 = nn.Linear(dim, 1)
        else:
            self.lin1_3 = nn.Linear(hypo_dim, 1)

        self.lin2_0 = nn.Linear(hypo_dim, hypo_dim)
        self.lin2_1 = nn.Linear(hypo_dim, hypo_dim)

        self.lin3_0 = nn.Linear(hypo_dim, hypo_dim)
        self.lin3_1 = nn.Linear(hypo_dim, hypo_dim)

        self.lin4_0 = nn.Linear(hypo_dim, 1)
        self.lin4_1 = nn.Linear(hypo_dim, 1)

        for lin in [self.lin1, self.lin1_1, self.lin1_2, self.lin2_0, self.lin2_1, self.lin1_3, self.lin3_0,
                    self.lin3_1, self.lin4_0, self.lin4_1]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, input):
        initial = input.view(input.shape)

        x = F.relu(self.lin1(initial))
        x = F.relu(self.lin1_1(x))
        x = F.relu(self.lin1_2(x))
        x = F.relu(x)
        if flags.net == 'tarnet':
            t = self.lin1_3(initial)
        else:
            t = self.lin1_3(x)

        h1 = F.relu(self.lin2_1(x))
        h0 = F.relu(self.lin2_0(x))

        h1 = F.relu(h1)
        h0 = F.relu(h0)

        h0 = F.relu(self.lin3_0(h0))
        h1 = F.relu(self.lin3_1(h1))

        h0 = self.lin4_0(h0)
        h1 = self.lin4_1(h1)
        return torch.cat((h0, h1, t), 1)