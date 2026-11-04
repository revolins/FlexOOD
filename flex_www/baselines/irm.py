"""IRM (Arjovsky et al., 2019, "Invariant Risk Minimization"), IRMv1 penalty.

Adds to the ERM link loss the per-environment IRMv1 penalty: the product of the gradients, w.r.t. a dummy scale on the
predictor's logit, of the loss on the two halves (even / odd positions) of each environment's edges (the unbiased
squared-gradient estimate). The penalty weight is 1 until epoch `anneal_iters` and `loss_coeff` afterwards; above 1 the
total is divided by the weight. Environments: envs.py.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..train import Hooks, typed_negatives
from .envs import EnvLabels


def train_epoch(model, predictor, train_pos, data, optimizer, batch_size, num_envs, env_idx, mask, loss_coeff, epoch,
                anneal_iters, neg_dst=None):
    model.train()
    predictor.train()
    x = data.x
    total_loss = total_examples = 0
    has_envs = env_idx.numel() == train_pos.size(0) and torch.unique(env_idx).numel() > 1
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        h = model(x, data.adj_t)
        edge = train_pos[perm].t()
        n_batch = edge.size(1)
        if has_envs:
            batch_env_idx, batch_mask = env_idx[perm], mask[perm].to(h.device)
        else:       # no environment signal: a random 0/1 split
            batch_env_idx = torch.randint(0, 2, (n_batch,), device=h.device)
            batch_mask = torch.ones(n_batch, device=h.device)
        pos_logit = predictor(x_i=h[edge[0]], x_j=h[edge[1]], logits=True)
        neg_edge = typed_negatives(edge, x.size(0), neg_dst, device=h.device)
        neg_logit = predictor(x_i=h[neg_edge[0]], x_j=h[neg_edge[1]], logits=True)
        loss = (F.binary_cross_entropy_with_logits(pos_logit, torch.ones_like(pos_logit), reduction='none')
                + F.binary_cross_entropy_with_logits(neg_logit, torch.zeros_like(neg_logit), reduction='none'))
        scale = torch.ones(1, device=h.device, requires_grad=True)
        scaled = (F.binary_cross_entropy_with_logits(pos_logit * scale, torch.ones_like(pos_logit), reduction='none')
                  + F.binary_cross_entropy_with_logits(neg_logit * scale, torch.zeros_like(neg_logit), reduction='none'))
        penalties = []
        for i in range(num_envs + 1):
            idx = (batch_env_idx == i).nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            g1 = torch.autograd.grad(scaled[idx[0::2]].mean(), scale, create_graph=True)[0]
            g2 = torch.autograd.grad(scaled[idx[1::2]].mean(), scale, create_graph=True)[0]
            penalties.append(torch.sum(g1 * g2))
        penalty = sum(penalties) / len(penalties) if penalties else torch.zeros((), device=h.device)
        mean_loss = loss.sum() / batch_mask.sum()
        weight = loss_coeff if epoch >= anneal_iters else 1.0
        loss = mean_loss + weight * penalty
        if weight > 1.0:
            loss = loss / weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * pos_logit.size(0)
        total_examples += pos_logit.size(0)
    return total_loss / total_examples


def hooks(opts):
    envs = EnvLabels(opts.get('num_envs', 3))

    def step(ctx, epoch, extra_loss_fn):
        e = ctx.extra
        return train_epoch(ctx.model, ctx.predictor, ctx.edges[0], ctx.graph.data, ctx.optimizer, ctx.batch_sizes[0],
                           e['num_envs'], e['env_idx'], e['mask'], opts['loss_coeff'], epoch,
                           opts.get('anneal_iters', 100), ctx.split.typed_neg_dst)
    return Hooks(setup=envs, train_step=step)
