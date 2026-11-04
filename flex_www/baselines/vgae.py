"""VGAE (Kipf & Welling, 2016, "Variational Graph Auto-Encoders") on the shared GCN backbone.

The encoder is unGCN plus a log-std GCN head on the last hidden layer; training uses a reparameterised sample
mu + eps * std of the node embeddings and adds `loss_coeff` (beta) * KL / N to the ERM link loss (PyG's VGAE scaling:
KL averaged over nodes, divided by N). Evaluation uses mu.

RNG: the extra head is initialised inside a forked RNG and the reparameterisation noise comes from a dedicated
generator seeded with torch.initial_seed() + 7919, so the global RNG stream (shuffle, edge dropout, dropout, negatives)
is ERM's.
"""
import contextlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv

from ..evaluate import predict
from ..models.gcn import unGCN
from ..models.predictor import mlp_score
from ..train import Hooks, typed_negatives

MAX_LOGSTD = 10          # PyG VGAE's clamp


@contextlib.contextmanager
def preserve_rng(module=None):
    """Fork the CPU RNG and the CUDA RNG of every device `module` lives on."""
    devs = []
    if module is not None:
        devs = sorted({p.device.index for p in module.parameters() if p.device.type == 'cuda'})
    with torch.random.fork_rng(devices=devs):
        yield


def noise_generator(device, offset):
    g = torch.Generator(device=device)
    g.manual_seed(int(torch.initial_seed()) + offset)
    return g


def hidden_stack(model, x, adj):
    """unGCN's forward up to (not including) the last convolution."""
    for conv in model.convs[:-1]:
        x = F.dropout(F.relu(conv(x, adj)), p=model.dropout, training=model.training)
    return x


class VGAEGCN(unGCN):
    """unGCN + a log-std GCNConv on the last hidden layer (subclass, so the `convs.*` keys stay unGCN's)."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, edge_drop, ood_method='VGAE'):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, edge_drop, ood_method)
        with preserve_rng():
            self.logstd_conv = GCNConv(in_channels if num_layers == 1 else hidden_channels, out_channels)
        self._gen = None
        self._mu = self._logstd = None

    def reset_parameters(self):
        super().reset_parameters()
        with preserve_rng(self):
            self.logstd_conv.reset_parameters()
        self._gen = None

    def forward(self, x, adj_t):
        adj = self.adjdrop(adj_t)
        hid = hidden_stack(self, x, adj)
        mu = self.convs[-1](hid, adj)
        if not self.training:
            return mu
        logstd = self.logstd_conv(hid, adj).clamp(max=MAX_LOGSTD)
        self._mu, self._logstd = mu, logstd
        if self._gen is None:
            self._gen = noise_generator(mu.device, 7919)
        eps = torch.randn(mu.shape, generator=self._gen, device=mu.device, dtype=mu.dtype)
        return mu + eps * torch.exp(logstd)

    def kl_loss(self):
        """Mean over nodes of the Gaussian KL to N(0, I) (PyG VGAE.kl_loss)."""
        mu, logstd = self._mu, self._logstd
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))


def build(ctx, input_channel):
    hp = ctx.spec.hp
    model = VGAEGCN(input_channel, hp['hidden_channels'], hp['hidden_channels'], hp['num_layers'], hp['dropout'],
                    hp.get('edge_drop', 0.0)).to(ctx.device)
    predictor = mlp_score(hp['hidden_channels'], hp['hidden_channels'], 1, hp['num_layers_predictor'], hp['dropout']).to(ctx.device)
    model.reset_parameters()
    predictor.reset_parameters()
    return model, predictor


def train_epoch(model, predictor, train_pos, data, optimizer, batch_size, beta, neg_dst=None):
    model.train()
    predictor.train()
    x = data.x
    total_loss = total_examples = 0
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        num_nodes = x.size(0)
        adj = data.adj_t
        h = model(x, adj)
        pos_edge = train_pos[perm].t()
        pos_out = predict(predictor, h, pos_edge, adj)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_out = predict(predictor, h, typed_negatives(pos_edge, num_nodes, neg_dst, device=h.device), adj)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        if beta != 0.0:
            loss = loss + beta * (1.0 / num_nodes) * model.kl_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * pos_out.size(0)
        total_examples += pos_out.size(0)
    return total_loss / total_examples


def hooks(opts):
    def step(ctx, epoch, extra_loss_fn):
        return train_epoch(ctx.model, ctx.predictor, ctx.edges[0], ctx.graph.data, ctx.optimizer, ctx.batch_sizes[0],
                           opts['loss_coeff'], ctx.split.typed_neg_dst)
    return Hooks(build_models=build, train_step=step)
