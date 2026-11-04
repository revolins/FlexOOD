"""V-REx (Krueger et al., ICML 2021, "Out-of-Distribution Generalization via Risk Extrapolation").

Adds to the ERM link loss the variance of the per-environment mean losses. The penalty weight is 1 until epoch
`anneal_iters` and `loss_coeff` afterwards (IRM's schedule); above 1 the total is divided by the weight.
Environments: envs.py.
"""
import torch
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
        else:
            batch_env_idx = torch.randint(0, 2, (n_batch,), device=h.device)
            batch_mask = torch.ones(n_batch, device=h.device)
        pos_out = predictor(x_i=h[edge[0]], x_j=h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15)
        neg_edge = typed_negatives(edge, x.size(0), neg_dst, device=h.device)
        neg_out = predictor(x_i=h[neg_edge[0]], x_j=h[neg_edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15)
        loss = pos_loss + neg_loss
        env_losses = []
        for i in range(num_envs + 1):
            env_mask = batch_env_idx == i
            if loss[env_mask].shape[0] > 0:
                env_losses.append(loss[env_mask].sum() / batch_mask[env_mask].sum())
        var_penalty = torch.var(torch.stack(env_losses))
        if torch.isnan(var_penalty):          # a single environment in the batch
            var_penalty = torch.zeros((), device=h.device)
        mean_loss = loss.sum() / batch_mask.sum()
        weight = loss_coeff if epoch >= anneal_iters else 1.0
        loss = mean_loss + weight * var_penalty
        if weight > 1.0:
            loss = loss / weight
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
        e = ctx.extra
        return train_epoch(ctx.model, ctx.predictor, ctx.edges[0], ctx.graph.data, ctx.optimizer, ctx.batch_sizes[0],
                           e['num_envs'], e['env_idx'], e['mask'], opts['loss_coeff'], epoch,
                           opts.get('anneal_iters', 100), ctx.split.typed_neg_dst)
    return Hooks(setup=envs, train_step=step)
