"""Deep CORAL (Sun & Saenko, ECCV-W 2016, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation").

Adds `loss_coeff` times the mean squared difference (/4) between the covariance matrices of the training-edge
representations h_u * h_v of every ordered pair of environments present in the batch (>= 2 edges each) to the ERM link
loss (summed over edges, divided by the mask count). Environments: envs.py.
"""
import torch
from torch.utils.data import DataLoader

from ..train import Hooks, typed_negatives
from .envs import EnvLabels


def covariance(x, device):
    n = x.shape[0]
    mean = torch.div(torch.mm(torch.ones((1, n), device=device), x), n)
    return torch.add(torch.mm(x.t(), x), (-1 * torch.mm(mean.t(), mean))) * 1 / (n - 1)


def train_epoch(model, predictor, train_pos, data, optimizer, batch_size, num_envs, env_idx, mask, loss_coeff, neg_dst=None):
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
            batch_env_idx, batch_mask = env_idx[perm].to(h.device), mask[perm].to(h.device)
        else:
            batch_env_idx = torch.randint(0, 2, (n_batch,), device=h.device)
            batch_mask = torch.ones(n_batch, device=h.device)
        pos_out = predictor(x_i=h[edge[0]], x_j=h[edge[1]])
        pos_loss = torch.nan_to_num(-torch.log(pos_out + 1e-15), nan=0.0)
        neg_edge = typed_negatives(edge, x.size(0), neg_dst, device=h.device)
        neg_out = predictor(x_i=h[neg_edge[0]], x_j=h[neg_edge[1]])
        neg_loss = torch.nan_to_num(-torch.log(1 - neg_out + 1e-15), nan=0.0)
        loss = pos_loss + neg_loss
        pair_rep = h[edge[0]] * h[edge[1]]
        covs = []
        for i in range(num_envs):
            feat = pair_rep[batch_env_idx == i]
            covs.append(covariance(feat, h.device) if feat.shape[0] > 1 else None)
        terms = []
        for i in range(num_envs):
            for j in range(num_envs):
                if i != j and covs[i] is not None and covs[j] is not None:
                    dis = covs[i] - covs[j]
                    terms.append(torch.mean(torch.mul(dis, dis)) / 4)
        coral_loss = torch.tensor(0) if not terms else sum(terms) / len(terms)
        spec_loss = loss_coeff * coral_loss
        if torch.isnan(spec_loss):
            spec_loss = 0
        loss = loss.sum() / batch_mask.sum().clamp(min=1) + spec_loss
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
                           e['num_envs'], e['env_idx'], e['mask'], opts['loss_coeff'], ctx.split.typed_neg_dst)
    return Hooks(setup=envs, train_step=step)
