"""Neural Common Neighbor (Wang et al., ICLR 2024): NCN_GCN encoder + CNLinkPredictor, from the authors' code (only the
pieces the paper's NCN rows construct: conv 'gcn' / 'puregcn' family, CN predictor without degree sampling).

The predictor needs the adjacency and the target edges (`needs_adj = True`): its common-neighbour features are computed
at prediction time, so it is called as predictor(h, adj, tar_ei) and returns logits.
"""
from typing import Final, Iterable, List

import torch
import torch.nn as nn
import torch_sparse
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
from torch_sparse.matmul import spmm_add, spmm_max, spmm_mean


# ---------------------------------------------------------------------------------------------- sparse overlap helpers
def elem2spm(element: Tensor, sizes: List[int]) -> SparseTensor:
    col = torch.bitwise_and(element, 0xffffffff)
    row = torch.bitwise_right_shift(element, 32)
    return SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(element.device).fill_value_(1.0)


def spm2elem(spm: SparseTensor) -> Tensor:
    return torch.bitwise_left_shift(spm.storage.row(), 32).add_(spm.storage.col())


def spmoverlap_(adj1: SparseTensor, adj2: SparseTensor) -> SparseTensor:
    assert adj1.sizes() == adj2.sizes()
    element1, element2 = spm2elem(adj1), spm2elem(adj2)
    if element2.shape[0] > element1.shape[0]:
        element1, element2 = element2, element1
    idx = torch.searchsorted(element1[:-1], element2)
    mask = (element1[idx] == element2)
    return elem2spm(element2[mask], adj1.sizes())


def adjoverlap(adj1: SparseTensor, adj2: SparseTensor, tarei: Tensor) -> SparseTensor:
    """Common-neighbour indicator rows for target edges tarei: row k marks the common neighbours of tarei[:, k]."""
    return spmoverlap_(adj1[tarei[0]], adj2[tarei[1]])


# ---------------------------------------------------------------------------------------------- encoder
class PureConv(nn.Module):
    aggr: Final[str]

    def __init__(self, indim, outdim, aggr="gcn") -> None:
        super().__init__()
        self.aggr = aggr
        if indim != outdim:
            raise NotImplementedError
        self.lin = nn.Identity()

    def forward(self, x, adj_t):
        x = self.lin(x)
        if self.aggr == "mean":
            return spmm_mean(adj_t, x)
        if self.aggr == "max":
            return spmm_max(adj_t, x)[0]
        if self.aggr == "sum":
            return spmm_add(adj_t, x)
        norm = torch.rsqrt_((1 + adj_t.sum(dim=-1))).reshape(-1, 1)
        x = norm * x
        x = spmm_add(adj_t, x) + x
        return norm * x


convdict = {
    "gcn": GCNConv,
    "gcn_cached": lambda indim, outdim: GCNConv(indim, outdim, cached=True),
    "sage": lambda indim, outdim: GCNConv(indim, outdim, aggr="mean", normalize=False, add_self_loops=False),
    "gin": lambda indim, outdim: GCNConv(indim, outdim, aggr="sum", normalize=False, add_self_loops=False),
    "max": lambda indim, outdim: GCNConv(indim, outdim, aggr="max", normalize=False, add_self_loops=False),
    "puremax": lambda indim, outdim: PureConv(indim, outdim, aggr="max"),
    "puresum": lambda indim, outdim: PureConv(indim, outdim, aggr="sum"),
    "puremean": lambda indim, outdim: PureConv(indim, outdim, aggr="mean"),
    "puregcn": lambda indim, outdim: PureConv(indim, outdim, aggr="gcn"),
    "none": None,
}


class DropAdj(nn.Module):
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
        adj = torch_sparse.masked_select_nnz(adj, mask, layout="coo")
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value() * self.ratio, layout="coo")
            else:
                adj.fill_value_(1 / (1 - self.dp), dtype=torch.float)
        return adj


class NCN_GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, ln=False, res=False, max_x=-1,
                 conv_fn="gcn", jk=False, edrop=0.0, xdropout=0.0, taildropout=0.0, noinputlin=False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.adjdrop = DropAdj(edrop)
        if max_x >= 0:
            tmp = nn.Embedding(max_x + 1, hidden_channels)
            nn.init.orthogonal_(tmp.weight)
            self.xemb = nn.Sequential(tmp, nn.Dropout(dropout))
            in_channels = hidden_channels
        else:
            self.xemb = nn.Sequential(nn.Dropout(xdropout))
            if not noinputlin and ("pure" in conv_fn or num_layers == 0):
                self.xemb.append(nn.Linear(in_channels, hidden_channels))
                self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())
        self.res = res
        self.jk = jk
        if jk:
            self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers,))))
        if num_layers == 0 or conv_fn == "none":
            self.jk = False
            return
        convfn = convdict[conv_fn]
        if num_layers == 1:
            hidden_channels = out_channels
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()  # noqa: E731
        if "pure" in conv_fn:
            self.convs.append(convfn(hidden_channels, hidden_channels))
            for i in range(num_layers - 1):
                self.lins.append(nn.Identity())
                self.convs.append(convfn(hidden_channels, hidden_channels))
            self.lins.append(nn.Dropout(taildropout, True))
        else:
            self.convs.append(convfn(in_channels, hidden_channels))
            self.lins.append(nn.Sequential(lnfn(hidden_channels, ln), nn.Dropout(dropout, True), nn.ReLU(True)))
            for i in range(num_layers - 1):
                self.convs.append(convfn(hidden_channels, hidden_channels if i == num_layers - 2 else out_channels))
                if i < num_layers - 2:
                    self.lins.append(nn.Sequential(lnfn(hidden_channels if i == num_layers - 2 else out_channels, ln),
                                                   nn.Dropout(dropout, True), nn.ReLU(True)))
                else:
                    self.lins.append(nn.Identity())     # no non-linearity before the Hadamard product

    def reset_parameters(self):
        for m in self.modules():
            if m is not self and hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, x, adj_t):
        x = self.xemb(x)
        jkx = []
        for i, conv in enumerate(self.convs):
            x1 = self.lins[i](conv(x, self.adjdrop(adj_t)))
            x = x1 + x if (self.res and x1.shape[-1] == x.shape[-1]) else x1
            if self.jk:
                jkx.append(x)
        if self.jk:
            x = torch.sum(torch.stack(jkx, dim=0) * self.jkparams.reshape(-1, 1, 1), dim=0)
        return x


# ---------------------------------------------------------------------------------------------- predictor
class CNLinkPredictor(nn.Module):
    needs_adj = True

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, edrop=0.0, ln=False, cndeg=-1,
                 use_xlin=False, tailact=False, twolayerlin=False, beta=1.0, no_cn=False):
        super().__init__()
        assert cndeg <= 0, 'common-neighbour degree sampling is not used by the paper rows'
        self.register_parameter("beta", nn.Parameter(beta * torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()  # noqa: E731
        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.Dropout(dropout, inplace=True),
                                  nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels),
                                  lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
                                  nn.ReLU(inplace=True)) if use_xlin else lambda x: 0
        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels), lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg
        self.no_cn = no_cn

    def reset_parameters(self):
        for m in self.modules():
            if m is not self and hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def multidomainforward(self, x, adj, tar_ei, filled1: bool = False, cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi, xj = x[tar_ei[0]], x[tar_ei[1]]
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei)
        xcns = [spmm_add(cn, x)]
        xij = self.xijlin(xi * xj)
        if self.no_cn:
            return torch.cat([self.lin(xij) for xcn in xcns], dim=-1)
        return torch.cat([self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns], dim=-1)

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])
