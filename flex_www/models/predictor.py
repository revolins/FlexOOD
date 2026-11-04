"""Hadamard MLP link predictor (the GCN rows' scorer): sigmoid(MLP(h_u * h_v))."""
import torch
import torch.nn.functional as F


class mlp_score(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i=None, x_j=None, logits=False):
        """Probability by default; logits=True returns the pre-sigmoid value (IRM scales a logit)."""
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x if logits else torch.sigmoid(x)
