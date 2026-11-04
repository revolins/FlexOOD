"""CIGA (Chen et al., NeurIPS 2022, "Learning Causally Invariant Representations for Out-of-Distribution Generalization
on Graphs"), adapted to transductive link prediction.

An edge-attention MLP (unGCN's `edge_att`) scores every train-graph edge from its endpoint embeddings; the top `ratio`
fraction forms the causal subgraph (sigmoid-weighted), the rest the spurious one. The link loss is computed on the
causal subgraph's embeddings, plus `contrast_coe` times a supervised contrastive loss over (positive, negative) edge
representations of the batch (subsampled to `contrast_max_pairs`), plus (CIGAv2, `spu_coe` > 0) a hinge loss on the
spurious subgraph. CIGA uses no environment labels. These helpers are shared with GALA and GI-Graph.
"""
import torch
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

from ..train import Hooks, typed_negatives


def split_graph(edge_index, edge_score, ratio, num_nodes):
    """(causal adj_t, spurious adj_t): the top-ceil(ratio * E) edges by score (weights sigmoid(score)) and the rest
    (weights sigmoid(-score)); ratio < 0 keeps every edge in both with soft weights. Directed, as the scores are."""
    device = edge_score.device
    num_edges = edge_score.size(0)
    if ratio is None or ratio < 0:
        causal_ei = spu_ei = edge_index
        causal_w, spu_w = torch.sigmoid(edge_score), torch.sigmoid(-edge_score)
    else:
        k = min(max(1, int(torch.tensor(ratio * num_edges).ceil().item())), num_edges)
        _, topk_idx = torch.topk(edge_score, k, largest=True, sorted=False)
        is_causal = torch.zeros(num_edges, dtype=torch.bool, device=device)
        is_causal[topk_idx] = True
        causal_ei, spu_ei = edge_index[:, is_causal], edge_index[:, ~is_causal]
        causal_w, spu_w = torch.sigmoid(edge_score[is_causal]), torch.sigmoid(-edge_score[~is_causal])
    return (SparseTensor.from_edge_index(causal_ei, causal_w, sparse_sizes=(num_nodes, num_nodes)),
            SparseTensor.from_edge_index(spu_ei, spu_w, sparse_sizes=(num_nodes, num_nodes)))


def contrast_loss(rep, labels, contrast_t=1.0, sampling='mul', norm=True):
    """Supervised contrastive loss (CIGA's 'mul' / 'var' sampling) over representations with binary labels."""
    sampling = sampling.lower()
    if sampling not in ('mul', 'var'):
        raise ValueError(f'unsupported sampling {sampling!r}')
    if norm:
        rep = torch.nn.functional.normalize(rep)
    device = rep.device
    labels = labels.to(device)
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
    anchor_dot_contrast = torch.div(torch.matmul(rep, rep.T), contrast_t)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    n = labels.size(0)
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(n, device=device).view(-1, 1), 0)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-15)
    is_valid = mask.sum(1) != 0
    mean_log_prob_pos = (mask * log_prob).sum(1)[is_valid] / mask.sum(1)[is_valid]
    loss = -mean_log_prob_pos.mean()
    if sampling == 'var':
        loss = loss + mean_log_prob_pos.var()
    return loss


def causal_pass(model, predictor, x, data, edge, ratio, neg_dst):
    """Shared first half of CIGA / GALA / GI-Graph: full-graph embeddings -> edge attention -> causal / spurious split ->
    causal embeddings; per-element link losses of the positives and one typed negative each."""
    num_nodes = x.size(0)
    h = model(x, data.adj_t)
    row, col = data.edge_index
    pred_edge_weight = model.edge_att(torch.cat([h[row], h[col]], dim=-1)).view(-1)
    causal_adj_t, spu_adj_t = split_graph(data.edge_index, pred_edge_weight, ratio, num_nodes)
    h_c = model(x, causal_adj_t)
    pos_out_c = predictor(x_i=h_c[edge[0]], x_j=h_c[edge[1]])
    pos_loss_elem = -torch.log(pos_out_c + 1e-15)
    neg_edge = typed_negatives(edge, num_nodes, neg_dst, device=h.device)
    neg_out_c = predictor(x_i=h_c[neg_edge[0]], x_j=h_c[neg_edge[1]])
    neg_loss_elem = -torch.log(1 - neg_out_c + 1e-15)
    return dict(h=h, h_c=h_c, spu_adj_t=spu_adj_t, neg_edge=neg_edge, pos_out_c=pos_out_c, neg_out_c=neg_out_c,
                pos_loss_elem=pos_loss_elem, neg_loss_elem=neg_loss_elem)


def spurious_hinge(pos_out_s, neg_out_s, pos_loss_elem, neg_loss_elem):
    """CIGAv2 hinge: the spurious-subgraph loss on the edges where it exceeds the causal loss."""
    spu = torch.cat([-torch.log(pos_out_s + 1e-15), -torch.log(1 - neg_out_s + 1e-15)]).view(-1)
    causal = torch.cat([pos_loss_elem, neg_loss_elem]).view(-1).detach()
    hinge = (spu > causal).float()
    return (spu * hinge).sum() / (hinge.sum() + 1e-15)


def train_epoch(model, predictor, train_pos, data, optimizer, batch_size, ratio=0.25, contrast_coe=4.0, contrast_t=1.0,
                spu_coe=0.0, sampling='mul', contrast_max_pairs=512, neg_dst=None):
    model.train()
    predictor.train()
    x = data.x
    total_loss = total_examples = 0
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        edge = train_pos[perm].t()
        c = causal_pass(model, predictor, x, data, edge, ratio, neg_dst)
        h_c, neg_edge, dev = c['h_c'], c['neg_edge'], c['h'].device
        pred_loss = c['pos_loss_elem'].mean() + c['neg_loss_elem'].mean()
        n_sample = min(edge.size(1), contrast_max_pairs)
        sub = torch.randperm(edge.size(1), device=dev)[:n_sample]
        pair_rep = torch.cat([h_c[edge[0][sub]] * h_c[edge[1][sub]], h_c[neg_edge[0][sub]] * h_c[neg_edge[1][sub]]], dim=0)
        pair_label = torch.cat([torch.ones(n_sample, device=dev), torch.zeros(n_sample, device=dev)])
        batch_loss = pred_loss + contrast_coe * contrast_loss(pair_rep, pair_label, contrast_t=contrast_t, sampling=sampling)
        if spu_coe > 0:
            h_s = model(x, c['spu_adj_t']).detach()
            pos_out_s = predictor(x_i=h_s[edge[0]], x_j=h_s[edge[1]])
            neg_out_s = predictor(x_i=h_s[neg_edge[0]], x_j=h_s[neg_edge[1]])
            batch_loss = batch_loss + spu_coe * spurious_hinge(pos_out_s, neg_out_s, c['pos_loss_elem'], c['neg_loss_elem'])
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        total_loss += batch_loss.item() * c['pos_out_c'].size(0)
        total_examples += c['pos_out_c'].size(0)
    return total_loss / total_examples


def hooks(opts):
    def step(ctx, epoch, extra_loss_fn):
        return train_epoch(ctx.model, ctx.predictor, ctx.edges[0], ctx.graph.data, ctx.optimizer, ctx.batch_sizes[0],
                           ratio=opts.get('ratio', 0.25), contrast_coe=opts.get('contrast_coe', 4.0),
                           contrast_t=opts.get('contrast_t', 1.0), spu_coe=opts.get('spu_coe', 0.0),
                           sampling=opts.get('sampling', 'mul'), contrast_max_pairs=opts.get('contrast_max_pairs', 512),
                           neg_dst=ctx.split.typed_neg_dst)
    return Hooks(train_step=step)
