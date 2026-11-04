"""GALA (Chen et al., NeurIPS 2023, "Does Invariant Graph Learning via Environment Augmentation Learn Invariance?").

CIGA's causal / spurious split and losses (ciga.py) with GALA's contrastive sampling: only HARD pairs are contrasted,
as judged by an assistant prediction per edge -- same label but different assistant prediction are pulled together,
different label but same assistant prediction pushed apart. The assistant is the environment label (envs.py) when the
batch has more than one environment, else the spurious-subgraph predictor's correctness (computed every batch).
"""
import torch
from torch.utils.data import DataLoader

from ..train import Hooks
from .ciga import causal_pass, spurious_hinge
from .envs import EnvLabels


def contrast_loss_gala(rep, labels, y_pred, contrast_t=1.0, norm=True):
    """Supervised contrastive loss restricted to hard pairs w.r.t. the assistant prediction `y_pred`."""
    if norm:
        rep = torch.nn.functional.normalize(rep)
    device = rep.device
    labels, y_pred = labels.to(device), y_pred.to(device)
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
    anchor_dot_contrast = torch.div(torch.matmul(rep, rep.T), contrast_t)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    n = labels.size(0)
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(n, device=device).view(-1, 1), 0)
    mask = mask * logits_mask
    pos_mask = torch.not_equal(y_pred.unsqueeze(1), y_pred.unsqueeze(1).T).float().to(device)
    neg_mask = torch.eq(y_pred.unsqueeze(1), y_pred.unsqueeze(1).T).float().to(device)
    logits_mask = logits_mask * torch.logical_not(mask * neg_mask)                      # drop easy positives
    logits_mask = logits_mask * torch.logical_not(pos_mask * torch.logical_not(mask))   # drop easy negatives
    pos_mask = mask * pos_mask
    if pos_mask.sum() == 0:
        pos_mask = mask
    if logits_mask.sum() == 0:
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(n, device=device).view(-1, 1), 0)
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    is_valid = pos_mask.sum(1) != 0
    if is_valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)[is_valid] / pos_mask.sum(1)[is_valid]
    return -mean_log_prob_pos.mean()


def train_epoch(model, predictor, train_pos, data, optimizer, batch_size, env_idx, ratio=0.25, contrast_coe=4.0,
                contrast_t=1.0, spu_coe=0.0, contrast_max_pairs=512, neg_dst=None):
    model.train()
    predictor.train()
    x = data.x
    total_loss = total_examples = 0
    has_envs = env_idx.numel() == train_pos.size(0) and torch.unique(env_idx).numel() > 1
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        edge = train_pos[perm].t()
        c = causal_pass(model, predictor, x, data, edge, ratio, neg_dst)
        h_c, neg_edge, dev = c['h_c'], c['neg_edge'], c['h'].device
        pred_loss = c['pos_loss_elem'].mean() + c['neg_loss_elem'].mean()
        h_s = model(x, c['spu_adj_t']).detach()
        pos_out_s = predictor(x_i=h_s[edge[0]], x_j=h_s[edge[1]])
        neg_out_s = predictor(x_i=h_s[neg_edge[0]], x_j=h_s[neg_edge[1]])
        n_sample = min(edge.size(1), contrast_max_pairs)
        sub = torch.randperm(edge.size(1), device=dev)[:n_sample]
        pair_rep = torch.cat([h_c[edge[0][sub]] * h_c[edge[1][sub]], h_c[neg_edge[0][sub]] * h_c[neg_edge[1][sub]]], dim=0)
        pair_label = torch.cat([torch.ones(n_sample, device=dev), torch.zeros(n_sample, device=dev)])
        if has_envs:
            batch_env = env_idx[perm].float().to(dev)[sub]
            pair_y_pred = torch.cat([batch_env, batch_env])
        else:
            correct_pos = (pos_out_s.detach() > 0.5).float().view(-1)
            correct_neg = (neg_out_s.detach() <= 0.5).float().view(-1)
            pair_y_pred = torch.cat([correct_pos[sub], correct_neg[sub]])
        batch_loss = pred_loss + contrast_coe * contrast_loss_gala(pair_rep, pair_label, pair_y_pred, contrast_t=contrast_t)
        if spu_coe > 0:
            batch_loss = batch_loss + spu_coe * spurious_hinge(pos_out_s, neg_out_s, c['pos_loss_elem'], c['neg_loss_elem'])
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        total_loss += batch_loss.item() * c['pos_out_c'].size(0)
        total_examples += c['pos_out_c'].size(0)
    return total_loss / total_examples


def hooks(opts):
    envs = EnvLabels(opts.get('num_envs', 3))

    def step(ctx, epoch, extra_loss_fn):
        return train_epoch(ctx.model, ctx.predictor, ctx.edges[0], ctx.graph.data, ctx.optimizer, ctx.batch_sizes[0],
                           ctx.extra['env_idx'], ratio=opts.get('ratio', 0.25), contrast_coe=opts.get('contrast_coe', 4.0),
                           contrast_t=opts.get('contrast_t', 1.0), spu_coe=opts.get('spu_coe', 0.0),
                           contrast_max_pairs=opts.get('contrast_max_pairs', 512), neg_dst=ctx.split.typed_neg_dst)
    return Hooks(setup=envs, train_step=step)
