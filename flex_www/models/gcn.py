"""GCN encoder (unGCN) with DropAdj edge dropout.

`ood_method` attaches the extra modules some OOD baselines need (DANN discriminator, IRM / GI-Graph dummy classifier,
CIGA / GALA / GI-Graph edge-attention MLP, GI-Graph prototypes). They are created BEFORE the convolutions, so they change
the initialisation stream and the state_dict keys of those baselines exactly as in the paper's runs; plain ERM attaches
nothing.
"""
from typing import Any, Final, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor, masked_select_nnz


class DropAdj(torch.nn.Module):
    """Drop each stored adjacency entry with probability dp during training and rescale the rest by 1/(1-dp)."""
    doscale: Final[bool]

    def __init__(self, dp: float = 0.0, doscale=True) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1 / (1 - dp)))
        self.doscale = doscale

    def forward(self, adj: SparseTensor) -> SparseTensor:
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = masked_select_nnz(adj, mask, layout="coo")
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value() * self.ratio, layout="coo")
            else:
                adj.fill_value_(1 / (1 - self.dp), dtype=torch.float)
        return adj


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        return input * 1.0

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(torch.nn.Module):
    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class mlp_adv(torch.nn.Module):
    """DANN domain discriminator."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.lins = torch.nn.ModuleList([torch.nn.Linear(in_channels, hidden_channels)])
        if num_layers > 1:
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t=None):
        for lin in self.lins[:-1]:
            x = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
        return self.lins[-1](x)


class unGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, edge_drop, ood_method=None):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.adjdrop = DropAdj(edge_drop)
        self.ood_method = ood_method
        if ood_method == 'DANN':
            self.grl = GradientReverseLayer()
            self.d_c = mlp_adv(hidden_channels, hidden_channels, 1, 3, dropout)
        if ood_method in ('IRM', 'GIGraph'):
            self.dummy_w = torch.nn.Parameter(torch.Tensor([1.0]))
        if ood_method in ('CIGA', 'GALA', 'GIGraph'):
            self.edge_att = torch.nn.Sequential(torch.nn.Linear(hidden_channels * 2, hidden_channels * 4), torch.nn.ReLU(),
                                                torch.nn.Linear(hidden_channels * 4, 1))
        if ood_method == 'GIGraph':
            self.register_buffer('gigraph_proto', torch.zeros(2, hidden_channels))
            self.gigraph_proto_init = [False, False]
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        elif num_layers > 1:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        adj_t_drop = self.adjdrop(adj_t)
        for conv in self.convs[:-1]:
            x = F.dropout(F.relu(conv(x, adj_t_drop)), p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t_drop)
        if self.ood_method == 'DANN':
            return x, self.d_c(self.grl(x))
        return x
