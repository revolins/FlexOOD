"""SIG-VAE (Hasanzadeh et al., NeurIPS 2019, "Semi-Implicit Graph Variational Auto-Encoders") on the shared GCN backbone.

The first GCN layer also propagates K + J copies of Bernoulli(0.5) noise (`gce`, rescaled as sigvae-torch does), so the
posterior is a mixture; one log-variance head on the noise-free hidden layer is shared by all copies. Training scores
J reparameterised samples against one negative draw per batch (mean ERM link loss over the samples) and adds
`loss_coeff` (beta) * (log_posterior_ker - log_prior_ker) / N^2, the sigvae-torch ELBO term. Evaluation replaces the
noise by its mean (deterministic embedding). Noise and samples come from dedicated generators (seed offset 104729), so
the global RNG stream is ERM's; the extra heads are initialised inside a forked RNG.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from ..models.gcn import unGCN
from ..models.predictor import mlp_score
from ..train import Hooks, typed_negatives
from .vgae import noise_generator, preserve_rng

SIGVAE_SMALL = 1e-6      # sigvae-torch/optimizer.py:SMALL


def gcn_multi(conv, h, adj):
    """GCNConv `conv` on a stack of feature copies h [C, N, F] in one propagation (copy axis folded into features)."""
    C, N, _ = h.shape
    hw = conv.lin(h)
    Fo = hw.size(-1)
    flat = hw.permute(1, 0, 2).reshape(N, C * Fo)
    adj_n = gcn_norm(adj, None, N, conv.improved, conv.add_self_loops, conv.flow, dtype=flat.dtype)
    out = conv.propagate(adj_n, x=flat, edge_weight=None).reshape(N, C, Fo).permute(1, 0, 2)
    if conv.bias is not None:
        out = out + conv.bias
    return out


class SIGVAEGCN(unGCN):
    """unGCN made semi-implicit (sigvae-torch GCNModelSIGVAE, encsto='semi') at the backbone's depth L:
    L >= 2: h1_c = relu(conv1 x) + relu(gce e_c) per noise copy, layers 2..L on every copy, mu_c = last conv;
    L == 1: mu_c = conv1 x + gce e_c; logvar = logvar_conv(noise-free last hidden layer)."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, edge_drop, ood_method='SIGVAE',
                 K=15, J=20, noise_dim=32):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, edge_drop, ood_method)
        first_out = out_channels if num_layers == 1 else hidden_channels
        last_in = in_channels if num_layers == 1 else hidden_channels
        with preserve_rng():
            self.gce = GCNConv(noise_dim, first_out)
            self.logvar_conv = GCNConv(last_in, out_channels)
        self.K, self.J, self.noise_dim = int(K), int(J), int(noise_dim)
        self.reweight = ((noise_dim + first_out) / (in_channels + first_out)) ** 0.5
        self._gen = None
        self._cache = None

    def reset_parameters(self):
        super().reset_parameters()
        with preserve_rng(self):
            self.gce.reset_parameters()
            self.logvar_conv.reset_parameters()
        self._gen = None

    def _noise(self, copies, n, device):
        if not self.training:
            return torch.full((copies, n, self.noise_dim), 0.5 * self.reweight, device=device)
        if self._gen is None:
            self._gen = noise_generator(device, 104729)
        e = torch.bernoulli(torch.full((copies, n, self.noise_dim), 0.5, device=device), generator=self._gen)
        return e * self.reweight

    def forward(self, x, adj_t):
        adj = self.adjdrop(adj_t)
        n = x.size(0)
        copies = (self.K + self.J) if self.training else 1
        e = self._noise(copies, n, x.device)
        if len(self.convs) == 1:
            mu = self.convs[0](x, adj).unsqueeze(0) + gcn_multi(self.gce, e, adj)
            hid_free = x
        else:
            hx = F.relu(self.convs[0](x, adj))
            he = F.relu(gcn_multi(self.gce, e, adj))
            h = F.dropout(hx.unsqueeze(0) + he, p=self.dropout, training=self.training)
            hid_free = F.dropout(hx, p=self.dropout, training=self.training)
            for conv in self.convs[1:-1]:
                h = F.dropout(F.relu(gcn_multi(conv, h, adj)), p=self.dropout, training=self.training)
                hid_free = F.dropout(F.relu(conv(hid_free, adj)), p=self.dropout, training=self.training)
            mu = gcn_multi(self.convs[-1], h, adj)
        if not self.training:
            return mu[0]
        logvar = self.logvar_conv(hid_free, adj)
        mu_emb = mu[self.K:]
        std = torch.exp(0.5 * logvar)
        if self._gen is None:
            self._gen = noise_generator(x.device, 104729)
        eps = torch.randn(mu_emb.shape, generator=self._gen, device=x.device, dtype=mu_emb.dtype)
        Z = mu_emb + eps * std.unsqueeze(0)
        self._cache = (mu, logvar, Z, eps)
        return Z

    def kl_term(self):
        """(log_posterior_ker - log_prior_ker) of sigvae-torch's loss, via a [J, K] Gram matrix in float64."""
        mu, logvar, Z, eps = self._cache
        K, J = self.K, Z.size(0)
        s = (torch.exp(0.5 * logvar) + SIGVAE_SMALL).double()
        Zs = (Z.double() / s).reshape(J, -1)
        Ms = (mu[:K].double() / s).reshape(K, -1)
        sq = Zs.pow(2).sum(1, keepdim=True) - 2.0 * Zs @ Ms.t() + Ms.pow(2).sum(1).unsqueeze(0)
        log_s = torch.log(s).sum()
        log_post_JK = -0.5 * sq - log_s
        log_post_J = (-0.5 * eps.double().pow(2).sum(dim=[1, 2]) - log_s).view(-1, 1)
        log_post = torch.cat([log_post_JK, log_post_J], dim=-1)
        log_post = log_post - np.log(K + 1.0) / J        # sic, sigvae-torch
        log_posterior_ker = torch.logsumexp(log_post, dim=-1).mean()
        log_prior_ker = torch.sum(-0.5 * Z.double().pow(2), dim=[1, 2]).mean()
        return (log_posterior_ker - log_prior_ker).float()


def builder(opts):
    def build(ctx, input_channel):
        hp = ctx.spec.hp
        model = SIGVAEGCN(input_channel, hp['hidden_channels'], hp['hidden_channels'], hp['num_layers'], hp['dropout'],
                          hp.get('edge_drop', 0.0), K=opts.get('K', 2), J=opts.get('J', 3),
                          noise_dim=opts.get('noise_dim', 32)).to(ctx.device)
        predictor = mlp_score(hp['hidden_channels'], hp['hidden_channels'], 1, hp['num_layers_predictor'],
                              hp['dropout']).to(ctx.device)
        model.reset_parameters()
        predictor.reset_parameters()
        return model, predictor
    return build


def train_epoch(model, predictor, train_pos, data, optimizer, batch_size, beta, neg_dst=None):
    model.train()
    predictor.train()
    x = data.x
    total_loss = total_examples = 0
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        num_nodes = x.size(0)
        Z = model(x, data.adj_t)                                   # [J, N, H]
        pos_edge = train_pos[perm].t()
        neg_edge = typed_negatives(pos_edge, num_nodes, neg_dst, device=Z.device)
        pos_out = predictor(x_i=Z[:, pos_edge[0]], x_j=Z[:, pos_edge[1]])
        neg_out = predictor(x_i=Z[:, neg_edge[0]], x_j=Z[:, neg_edge[1]])
        loss = -torch.log(pos_out + 1e-15).mean() - torch.log(1 - neg_out + 1e-15).mean()
        if beta != 0.0:
            loss = loss + beta * model.kl_term() / float(num_nodes) ** 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * pos_edge.size(1)
        total_examples += pos_edge.size(1)
    return total_loss / total_examples


def hooks(opts):
    def step(ctx, epoch, extra_loss_fn):
        return train_epoch(ctx.model, ctx.predictor, ctx.edges[0], ctx.graph.data, ctx.optimizer, ctx.batch_sizes[0],
                           opts['loss_coeff'], ctx.split.typed_neg_dst)
    return Hooks(build_models=builder(opts), train_step=step)
