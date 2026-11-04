"""SR-GNN (Zhu et al., NeurIPS 2021, "Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data"),
the CMD regulariser (the KMM instance-reweighting variant is not used).

Adds `loss_coeff` times the central moment discrepancy (order `cmd_k`) between the representations h_u * h_v of the
reference-environment training edges (env 0) and of the shifted ones (env != 0), size-matched by random subsampling,
to the ERM link loss. Skipped for batches with fewer than 2 edges on either side. Environments: envs.py.
"""
import torch
from torch.utils.data import DataLoader

from ..train import Hooks, typed_negatives
from .envs import EnvLabels


def l2diff(x1, x2):
    return (x1 - x2).norm(p=2)


def moment_diff(sx1, sx2, k):
    return l2diff(sx1.pow(k).mean(0), sx2.pow(k).mean(0))


def cmd(x1, x2, K=5):
    """Central moment discrepancy (Zellinger et al., ICLR 2017): mean difference + centred moments 2..K."""
    mx1, mx2 = x1.mean(0), x2.mean(0)
    sx1, sx2 = x1 - mx1, x2 - mx2
    scms = [l2diff(mx1, mx2)]
    for i in range(K - 1):
        scms.append(moment_diff(sx1, sx2, i + 2))
    return sum(scms)


def train_epoch(model, predictor, train_pos, data, optimizer, batch_size, env_idx, loss_coeff, cmd_k=5, neg_dst=None):
    model.train()
    predictor.train()
    x = data.x
    total_loss = total_examples = 0
    has_envs = env_idx.numel() == train_pos.size(0) and torch.unique(env_idx).numel() > 1
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        h = model(x, data.adj_t)
        pos_edge = train_pos[perm].t()
        pos_out = predictor(x_i=h[pos_edge[0]], x_j=h[pos_edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_edge = typed_negatives(pos_edge, x.size(0), neg_dst, device=h.device)
        neg_out = predictor(x_i=h[neg_edge[0]], x_j=h[neg_edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        if has_envs:
            src_mask = env_idx[perm] == 0
            tgt_mask = ~src_mask
            if src_mask.sum() > 1 and tgt_mask.sum() > 1:
                rep = h[pos_edge[0]] * h[pos_edge[1]]
                rep_src, rep_tgt = rep[src_mask], rep[tgt_mask]
                n = min(rep_src.size(0), rep_tgt.size(0))
                src_perm = torch.randperm(rep_src.size(0), device=h.device)[:n]
                tgt_perm = torch.randperm(rep_tgt.size(0), device=h.device)[:n]
                loss = loss + loss_coeff * cmd(rep_src[src_perm], rep_tgt[tgt_perm], K=cmd_k)
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
                           ctx.extra['env_idx'], opts['loss_coeff'], opts.get('cmd_k', 5), ctx.split.typed_neg_dst)
    return Hooks(setup=envs, train_step=step)
