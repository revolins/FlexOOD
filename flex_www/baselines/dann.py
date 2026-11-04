"""DANN (Ganin et al., JMLR 2016, "Domain-Adversarial Training of Neural Networks").

A domain discriminator (unGCN's `d_c`, behind a gradient-reversal layer) classifies each training edge's
representation h_u * h_v as reference environment (env 0) vs shifted (env != 0); its BCE, weighted by `loss_coeff`, is
added to the ERM link loss, so the encoder learns environment-invariant edge representations. The encoder's own
per-node discriminator output (computed in unGCN.forward, consuming dropout randomness) is unused. Environments: envs.py.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..train import Hooks, typed_negatives
from .envs import EnvLabels


def train_epoch(model, predictor, train_pos, data, optimizer, batch_size, env_idx, loss_coeff, neg_dst=None):
    model.train()
    predictor.train()
    x = data.x
    total_loss = total_examples = 0
    has_envs = env_idx.numel() == train_pos.size(0) and torch.unique(env_idx).numel() > 1
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        h, _ = model(x, data.adj_t)
        edge = train_pos[perm].t()
        pos_out = predictor(x_i=h[edge[0]], x_j=h[edge[1]])
        pos_loss = torch.nan_to_num(-torch.log(pos_out + 1e-15), nan=0.0)
        neg_edge = typed_negatives(edge, x.size(0), neg_dst, device=h.device)
        neg_out = predictor(x_i=h[neg_edge[0]], x_j=h[neg_edge[1]])
        neg_loss = torch.nan_to_num(-torch.log(1 - neg_out + 1e-15), nan=0.0)
        loss = pos_loss + neg_loss
        dc_pred = model.d_c(model.grl(h[edge[0]] * h[edge[1]])).view(-1)
        if has_envs:
            domain_label = (env_idx[perm] != 0).float().to(h.device)
        else:
            domain_label = torch.randint(0, 2, (edge.size(1),), device=h.device).float()
        dc_loss = F.binary_cross_entropy_with_logits(dc_pred, domain_label, reduction='none')
        loss = loss.mean() + loss_coeff * dc_loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * pos_out.size(0)
        total_examples += pos_out.size(0)
    return total_loss / total_examples


def hooks(opts):
    envs = EnvLabels(opts.get('num_envs', 3))

    def step(ctx, epoch, extra_loss_fn):
        return train_epoch(ctx.model, ctx.predictor, ctx.edges[0], ctx.graph.data, ctx.optimizer, ctx.batch_sizes[0],
                           ctx.extra['env_idx'], opts['loss_coeff'], ctx.split.typed_neg_dst)
    return Hooks(setup=envs, train_step=step)
