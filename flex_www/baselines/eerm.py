"""EERM (Wu et al., ICLR 2022, "Handling Distribution Shifts on Graphs: An Invariance Perspective").

A learned graph editor generates K environments per batch by adding `num_sample` edges to the train graph, each drawn
from its own softmax distribution over all node pairs. The objective is the mean ERM link loss over the K environments
plus `loss_coeff` times its variance across them; the editor is trained by REINFORCE (-log p(sampled edges) times the
detached objective). EERM ignores the heuristic environment labels: its K edited graphs are its environments.

The editor (`model.gl`, [K, N, N] logits) is created at the first training epoch, as in the paper's runs, so it is part
of the encoder's state_dict (`gl.B`) and of its gradient clipping, and its uniform initialisation draws from the CPU
generator at that point.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_sparse import SparseTensor

from ..train import Hooks, typed_negatives


class GraphEditor(nn.Module):
    """K learned edge distributions over the N x N node pairs (softmax over B[k])."""

    def __init__(self, K, n, device=None):
        super().__init__()
        self.B = nn.Parameter(torch.empty(K, n, n))
        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, edge_index, n, num_sample, k):
        """(edge_index of the train graph plus `num_sample` sampled symmetric edges, log-probability of the sample)."""
        device = self.B.device
        A = to_dense_adj(edge_index, max_num_nodes=n)[0].to(torch.float).to(device)
        P = torch.softmax(self.B[k].view(-1), dim=0)
        S = torch.multinomial(P, num_samples=num_sample, replacement=True)
        rows, cols = torch.clamp(S // n, 0, n - 1), torch.clamp(S % n, 0, n - 1)
        M = torch.zeros((n, n), dtype=torch.float, device=device)
        M[rows, cols] = 1.0
        M[cols, rows] = 1.0
        new_edge_index = dense_to_sparse(torch.clamp(A + M, max=1.0))[0]
        return new_edge_index, torch.sum(torch.log(P[S] + 1e-15))


def attach_editor(model, optimizer, K, num_nodes, device):
    """Create the editor on the model (and register its parameters with the optimiser) unless it already exists."""
    if not hasattr(model, 'gl') or model.gl.B.shape[0] != K or model.gl.B.shape[1] != num_nodes:
        model.gl = GraphEditor(K=K, n=num_nodes, device=device)
        if optimizer is not None:
            optimizer.add_param_group({'params': model.gl.parameters()})


def train_epoch(model, predictor, train_pos, data, optimizer, batch_size, loss_coeff, K=5, neg_dst=None):
    model.train()
    predictor.train()
    x = data.x
    num_nodes = x.size(0)
    num_sample = min(10, train_pos.size(0) // 10)
    attach_editor(model, optimizer, K, num_nodes, x.device)
    total_loss = total_examples = 0
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        all_losses, log_ps = [], []
        for k in range(K):
            aug_edge_index, log_p = model.gl(data.edge_index, num_nodes, num_sample, k)
            h = model(x, SparseTensor.from_edge_index(aug_edge_index, sparse_sizes=(num_nodes, num_nodes)))
            edge = train_pos[perm].t()
            pos_out = predictor(x_i=h[edge[0]], x_j=h[edge[1]])
            pos_loss = torch.nan_to_num(-torch.log(pos_out + 1e-15), nan=0.0)
            neg_edge = typed_negatives(edge, num_nodes, neg_dst, device=h.device)
            neg_out = predictor(x_i=h[neg_edge[0]], x_j=h[neg_edge[1]])
            neg_loss = torch.nan_to_num(-torch.log(1 - neg_out + 1e-15), nan=0.0)
            all_losses.append((pos_loss + neg_loss).view(-1))
            log_ps.append(log_p)
        losses = torch.cat(all_losses, dim=0)
        batch_loss = torch.mean(losses) + loss_coeff * torch.var(losses)
        editor_loss = -torch.stack(log_ps).sum() * batch_loss.detach()
        (batch_loss + editor_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        total_loss += batch_loss.item() * train_pos[perm].size(0)
        total_examples += train_pos[perm].size(0)
    return total_loss / total_examples


def hooks(opts):
    def step(ctx, epoch, extra_loss_fn):
        return train_epoch(ctx.model, ctx.predictor, ctx.edges[0], ctx.graph.data, ctx.optimizer, ctx.batch_sizes[0],
                           opts['loss_coeff'], opts.get('K', 5), ctx.split.typed_neg_dst)
    return Hooks(train_step=step)


def before_load(ctx, opts):
    """Weights mode: a trained checkpoint carries the editor's parameters (gl.B), so create it before a strict load."""
    attach_editor(ctx.model, None, opts.get('K', 5), ctx.split.num_nodes, ctx.device)
