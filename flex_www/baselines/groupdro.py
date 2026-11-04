"""GroupDRO (Sagawa et al., ICLR 2020, "Distributionally Robust Neural Networks for Group Shifts").

Replaces the mean ERM link loss by a weighted sum of the per-environment mean losses; the group weights q are updated
multiplicatively, q_g <- q_g * exp(loss_coeff * loss_g), renormalised, and carried across batches and epochs (one q per
seed). Environments: envs.py.
"""
import torch
from torch.utils.data import DataLoader

from ..train import Hooks, typed_negatives
from .envs import EnvLabels


def train_epoch(model, predictor, train_pos, data, optimizer, batch_size, num_envs, env_idx, mask, loss_coeff, state,
                neg_dst=None):
    """state: dict holding the group weights 'q' across calls."""
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
        pos_loss = torch.nan_to_num(-torch.log(pos_out + 1e-15), nan=0.0)
        neg_edge = typed_negatives(edge, x.size(0), neg_dst, device=h.device)
        neg_out = predictor(x_i=h[neg_edge[0]], x_j=h[neg_edge[1]])
        neg_loss = torch.nan_to_num(-torch.log(1 - neg_out + 1e-15), nan=0.0)
        loss = pos_loss + neg_loss
        env_losses, present_envs = [], []
        for i in range(num_envs + 1):
            env_mask = batch_env_idx == i
            if loss[env_mask].shape[0] > 0 and batch_mask[env_mask].sum() > 0:
                env_losses.append(loss[env_mask].sum() / batch_mask[env_mask].sum())
                present_envs.append(i)
        losses = torch.stack(env_losses)
        if state.get('q') is None or state['q'].numel() != num_envs + 1:
            state['q'] = torch.ones(num_envs + 1, device=h.device)
        q = state['q'].to(h.device)
        present = torch.tensor(present_envs, device=h.device, dtype=torch.long)
        q[present] = q[present] * torch.exp(loss_coeff * losses.detach())
        q = q / q.sum()
        state['q'] = q.detach()
        w = q[present] / q[present].sum()
        loss = losses @ w
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
                           e['num_envs'], e['env_idx'], e['mask'], opts['loss_coeff'], e.setdefault('groupdro', {}),
                           ctx.split.typed_neg_dst)
    return Hooks(setup=envs, train_step=step)
