"""Refined-GAE (Ma et al., "Reconsidering the Performance of GAE in Link Prediction", CIKM 2025) in PyG, following the
authors' DGL classes (GCN_with_feature + Hadamard_MLPPredictor) used by their published collab command.

Faithfulness notes
  * DGL GraphConv(norm='both') on a graph with self-loops and unit edge weights == PyG GCNConv(normalize=True,
    add_self_loops=True) on the same graph without self-loops: D^-1/2 (A + I) D^-1/2 X W + b. The encoder therefore
    ignores edge values (the authors' collab graph is binary: weight 1 on every edge, train and test).
  * conv2 is SHARED across propagation steps 2..prop_step (GCN_with_feature reuses self.conv2).
  * LayerNorm + dropout precede every propagation step after the first (no ReLU in the published collab command).
  * collab.py passes args.dropout into the dp4norm slot, which GCN_with_feature never reads; --dp4norm has no effect.
  * The predictor returns LOGITS; training uses -logsigmoid(pos) - logsigmoid(-neg) (flex_www.train.train_rgae).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def _binary(adj_t):
    """Structure only: drop edge values (the authors' graphs carry weight 1 everywhere) and existing self-loops
    (GCNConv adds exactly one per node, as dgl.add_self_loop does)."""
    return adj_t.set_value(None, layout='coo').remove_diag()


class RGAEEncoder(nn.Module):
    def __init__(self, in_feats, h_feats, norm=True, prop_step=4, dropout=0.2, residual=0.0, relu=False):
        super().__init__()
        self.conv1 = GCNConv(in_feats, h_feats, normalize=True, add_self_loops=True)
        self.conv2 = GCNConv(h_feats, h_feats, normalize=True, add_self_loops=True)
        self.prop_step, self.residual, self.relu, self.norm = prop_step, residual, relu, norm
        if norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(prop_step)])
            self.dp = nn.Dropout(dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters(); self.conv2.reset_parameters()
        if self.norm:
            for n in self.norms:
                n.reset_parameters()

    def forward(self, x, adj_t):
        A = _binary(adj_t)
        h = self.conv1(x, A)
        ori = h
        for i in range(1, self.prop_step):
            if self.norm:
                h = self.norms[i](h)
            if self.relu:
                h = F.relu(h)
            if self.norm:
                h = self.dp(h)
            h = self.conv2(h, A) + self.residual * ori
        return h


class RGAEEmb(nn.Module):
    """Optional trainable node embedding concatenated in front of the features (collab.py --emb_hidden > 0)."""
    def __init__(self, encoder, num_nodes, emb_dim):
        super().__init__()
        self.encoder, self.emb = encoder, nn.Embedding(num_nodes, emb_dim)

    def reset_parameters(self):
        self.encoder.reset_parameters(); nn.init.orthogonal_(self.emb.weight)

    def forward(self, x, adj_t):
        return self.encoder(torch.cat([self.emb.weight, x], dim=-1), adj_t)


class RGAEScore(nn.Module):
    """Hadamard_MLPPredictor (model.py), verbatim semantics; returns logits of shape [N, 1]."""
    def __init__(self, h_feats, dropout, layer=2, res=False, norm=False, scale=False):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(h_feats, h_feats)])
        for _ in range(layer - 2):
            self.lins.append(nn.Linear(h_feats, h_feats))
        self.lins.append(nn.Linear(h_feats, 1))
        self.dropout, self.res, self.scale, self.norm = dropout, res, scale, norm
        if scale:
            self.scale_norm = nn.LayerNorm(h_feats)
        if norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(layer - 1)])

    def reset_parameters(self):
        for m in self.modules():
            if m is not self and hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        if self.scale:
            x = self.scale_norm(x)
        ori = x
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            if self.res:
                x = x + ori
            if self.norm:
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[-1](x)
