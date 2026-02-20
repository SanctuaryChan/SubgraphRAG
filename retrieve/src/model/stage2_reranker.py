import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


class Stage2NodeReranker(nn.Module):
    def __init__(self, in_dim, hid_dim=256, num_layers=2, dropout=0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError('num_layers must be >= 1')

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hid_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hid_dim, hid_dim))
        self.out = nn.Linear(hid_dim, 1)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out(h).squeeze(-1)
