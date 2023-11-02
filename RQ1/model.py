import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class MLP1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

class TwoTower(nn.Module):
    def __init__(self, input_article_dim, input_customer_dim, output_dim=3):
        super(TwoTower,self).__init__()
        # Article tower
        self.afc1 = nn.Linear(input_article_dim, 5)
        self.afc2 = nn.Linear(5, output_dim)
        # Customer tower
        self.cfc1 = nn.Linear(input_customer_dim, 5)
        self.cfc2 = nn.Linear(5, output_dim)

    def forward(self, x, y):
        #customers
        x = F.relu(self.cfc1(x))
        x = self.cfc2(x)
        # articles
        y = F.relu(self.afc1(y))
        y = self.afc2(y)
        # return product 
        return torch.matmul(x,y.T).diag()
    
    
