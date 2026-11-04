"""GI-Graph's augmentation pool on a single graph: a graph instance is the k-hop ego-subgraph of a training pair (capped
at `max_nodes`); it is split into invariant / environment parts by the encoder's edge attention, Env-DM and Inv-DM
sample new parts, and the two are spliced with random virtual edges (eq. 17). The pool is regenerated every
`diffuse_every` epochs; the diffusion models train every batch on a few extracted subgraphs (own optimiser)."""
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, to_dense_adj

from .diffusion import AttributePredictionNetwork, DiffusionModel, forward_noise
from .fgw import fgw_finetune_loss, fgw_guidance_grad


def extract_ego_subgraph(edge_index, u, v, x, k_hops=2, max_nodes=50, num_nodes=None):
    """k-hop subgraph around (u, v), keeping the query nodes plus the nearest (earliest discovered) others up to
    max_nodes -> {'A' dense, 'X', 'global_ids', 'center_local'} or None."""
    if num_nodes is None:
        num_nodes = x.size(0)
    device = edge_index.device
    node_idx = torch.as_tensor([int(u), int(v)], device=device)
    subset, local_ei, mapping, _ = k_hop_subgraph(node_idx, k_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes)
    if subset.numel() > max_nodes:
        keep = torch.zeros(subset.size(0), dtype=torch.bool, device=device)
        keep[mapping] = True
        budget = max_nodes - int(keep.sum().item())
        other_idx = (~keep).nonzero(as_tuple=True)[0]
        if budget > 0:
            keep[other_idx[:budget]] = True
        keep_ids = keep.nonzero(as_tuple=True)[0]
        old_to_new = -torch.ones(subset.size(0), dtype=torch.long, device=device)
        old_to_new[keep_ids] = torch.arange(keep_ids.size(0), device=device)
        edge_keep = keep[local_ei[0]] & keep[local_ei[1]]
        local_ei = old_to_new[local_ei[:, edge_keep]]
        mapping = old_to_new[mapping]
        subset = subset[keep_ids]
    n = subset.size(0)
    if n == 0:
        return None
    return {'A': to_dense_adj(local_ei, max_num_nodes=n)[0], 'X': x[subset], 'global_ids': subset, 'center_local': mapping}


def splice_subgraphs(inv_sub, env_sub_aug, r_add=0.5):
    """eq. 17: invariant nodes first, then the environment nodes not shared with it (shared nodes identified by global
    id), plus max(1, r_add * |E|) random undirected virtual edges."""
    inv_ids, env_ids = inv_sub['global_ids'], env_sub_aug['global_ids']
    shared_mask_env = torch.isin(env_ids, inv_ids)
    env_unique_ids = env_ids[~shared_mask_env]
    merged_ids = torch.cat([inv_ids, env_unique_ids])
    n_inv = inv_ids.size(0)
    n_total = n_inv + env_unique_ids.size(0)
    X = torch.cat([inv_sub['X'], env_sub_aug['X'][~shared_mask_env]], dim=0)
    A = torch.zeros(n_total, n_total, device=X.device)
    A[:n_inv, :n_inv] = inv_sub['A']
    inv_pos = {gid: i for i, gid in enumerate(inv_ids.tolist())}
    env_unique_pos = {gid: n_inv + i for i, gid in enumerate(env_unique_ids.tolist())}
    env_to_merged = torch.tensor([inv_pos.get(gid, env_unique_pos.get(gid)) for gid in env_ids.tolist()],
                                 dtype=torch.long, device=X.device)
    env_row, env_col = env_sub_aug['A'].nonzero(as_tuple=True)
    if env_row.numel() > 0:
        A[env_to_merged[env_row], env_to_merged[env_col]] = 1.0
    n_virtual = max(1, int(r_add * (int(inv_sub['A'].sum().item()) + int(env_sub_aug['A'].sum().item()))))
    if n_virtual > 0 and n_total > 1:
        src = torch.randint(0, n_total, (n_virtual,), device=X.device)
        dst = torch.randint(0, n_total, (n_virtual,), device=X.device)
        valid = src != dst
        A[src[valid], dst[valid]] = 1.0
        A[dst[valid], src[valid]] = 1.0
    return {'A': A, 'X': X, 'global_ids': merged_ids}


def split_local_subgraph(sub, h, edge_att, ratio):
    """Local version of the causal / spurious split: the top int(ratio * E) subgraph edges by the encoder's edge
    attention (on the full-graph embeddings h) are invariant, the rest environment."""
    A, global_ids = sub['A'], sub['global_ids']
    n = A.size(0)
    local_row, local_col = A.nonzero(as_tuple=True)
    if local_row.numel() == 0:
        return ({'A': A.clone(), 'X': sub['X'], 'global_ids': global_ids},
                {'A': A.clone(), 'X': sub['X'], 'global_ids': global_ids})
    scores = edge_att(torch.cat([h[global_ids[local_row]], h[global_ids[local_col]]], dim=-1)).view(-1)
    k = min(max(1, int(ratio * local_row.numel())), local_row.numel())
    _, topk_idx = torch.topk(scores, k, largest=True, sorted=False)
    is_causal = torch.zeros(local_row.numel(), dtype=torch.bool, device=A.device)
    is_causal[topk_idx] = True
    A_inv = torch.zeros(n, n, device=A.device)
    A_env = torch.zeros(n, n, device=A.device)
    A_inv[local_row[is_causal], local_col[is_causal]] = 1.0
    A_env[local_row[~is_causal], local_col[~is_causal]] = 1.0
    return {'A': A_inv, 'X': sub['X'], 'global_ids': global_ids}, {'A': A_env, 'X': sub['X'], 'global_ids': global_ids}


class GIGraphPool:
    """Env-DM / Inv-DM / attribute network, the current augmentation pool and the FGW reference caches of one seed."""

    def __init__(self, feat_dim, hidden_dim=32, num_layers=2, time_dim=16, ood_alpha=0.1, k_hops=2, max_nodes=30,
                 diffuse_frac=0.1, max_pool_size=20, r_add=0.5, sample_steps=50, score_train_samples=4,
                 use_attribute_net=True, guidance_strength=1.0, fgw_cache_size=8, fgw_max_samples=4, fgw_alpha=0.5,
                 fgw_epsilon=1.0, fgw_guidance_every=5, fgw_finetune_every=5, device='cpu'):
        self.env_dm = DiffusionModel(feat_dim, hidden_dim, num_layers, time_dim, ood_alpha=ood_alpha).to(device)
        self.inv_dm = DiffusionModel(feat_dim, hidden_dim, num_layers, time_dim, ood_alpha=None).to(device)
        self.attribute_net = (AttributePredictionNetwork(feat_dim, hidden_dim, num_layers).to(device)
                              if use_attribute_net else None)
        self.k_hops, self.max_nodes, self.diffuse_frac, self.max_pool_size = k_hops, max_nodes, diffuse_frac, max_pool_size
        self.r_add, self.sample_steps, self.score_train_samples = r_add, sample_steps, score_train_samples
        self.guidance_strength = guidance_strength
        self.fgw_cache_size, self.fgw_max_samples = fgw_cache_size, fgw_max_samples
        self.fgw_alpha, self.fgw_epsilon = fgw_alpha, fgw_epsilon
        self.fgw_guidance_every, self.fgw_finetune_every = fgw_guidance_every, fgw_finetune_every
        self.batch_counter = 0
        self.device = device
        self.pool = []         # (spliced subgraph, true label)
        self.id_cache = []     # invariant parts of correctly classified pairs
        self.ood_cache = []    # invariant parts of misclassified pairs

    def diffusion_model_params(self):
        params = list(self.env_dm.parameters()) + list(self.inv_dm.parameters())
        if self.attribute_net is not None:
            params += list(self.attribute_net.parameters())
        return params

    def train_step(self, data, x, h, edge_att, train_pos, train_labels, ratio):
        """Score-matching loss of Env-DM / Inv-DM plus the attribute network's BCE on noised invariant parts, over
        `score_train_samples` random training pairs."""
        num_nodes = x.size(0)
        idx = torch.randperm(train_pos.size(0))[:self.score_train_samples]
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0
        for i in idx.tolist():
            u, v = int(train_pos[i, 0]), int(train_pos[i, 1])
            label = float(train_labels[i]) if train_labels is not None else 1.0
            sub = extract_ego_subgraph(data.edge_index, u, v, x.detach(), self.k_hops, self.max_nodes, num_nodes)
            if sub is None or sub['A'].sum() == 0:
                continue
            inv_sub, env_sub = split_local_subgraph(sub, h.detach(), edge_att, ratio)
            if inv_sub['A'].sum() == 0 or env_sub['A'].sum() == 0:
                continue
            loss = (self.inv_dm.score_matching_loss(inv_sub['X'], inv_sub['A'])
                    + self.env_dm.score_matching_loss(env_sub['X'], env_sub['A']))
            if self.attribute_net is not None:
                t = torch.rand(1, device=self.device).clamp(min=1e-3)
                Xt, _, _ = forward_noise(inv_sub['X'], t, self.inv_dm.sde)
                At, _, _ = forward_noise(inv_sub['A'], t, self.inv_dm.sde)
                At = (At + At.T) / 2
                p = torch.nan_to_num(self.attribute_net(Xt, At), nan=0.5).clamp(1e-6, 1 - 1e-6)
                loss = loss + F.binary_cross_entropy(p, torch.tensor(label, device=self.device).expand_as(p))
            total_loss = total_loss + loss
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return total_loss / count

    @torch.no_grad()
    def update_fgw_caches(self, data, x, h, edge_att, ratio, all_edges, correct_mask, pair_label):
        """Push the invariant parts of up to fgw_max_samples correctly classified / misclassified batch pairs into the
        FIFO caches (capped at fgw_cache_size)."""
        num_nodes = x.size(0)

        def extract_and_cache(idx_pool, cache):
            if idx_pool.numel() == 0:
                return
            sel = idx_pool[torch.randperm(idx_pool.numel())[:self.fgw_max_samples]]
            for i in sel.tolist():
                u, v = int(all_edges[0, i]), int(all_edges[1, i])
                sub = extract_ego_subgraph(data.edge_index, u, v, x, self.k_hops, self.max_nodes, num_nodes)
                if sub is None or sub['A'].sum() == 0:
                    continue
                inv_sub, _ = split_local_subgraph(sub, h, edge_att, ratio)
                if inv_sub['A'].sum() == 0:
                    continue
                cache.append({'A': inv_sub['A'], 'X': inv_sub['X']})
                if len(cache) > self.fgw_cache_size:
                    cache.pop(0)

        extract_and_cache(correct_mask.nonzero(as_tuple=True)[0], self.id_cache)
        extract_and_cache((~correct_mask).nonzero(as_tuple=True)[0], self.ood_cache)

    def fgw_finetune_loss(self):
        if len(self.id_cache) == 0 or len(self.ood_cache) == 0:
            return torch.tensor(0.0, device=self.device)
        return fgw_finetune_loss(self.inv_dm, self.id_cache, self.ood_cache, alpha=self.fgw_alpha, epsilon=self.fgw_epsilon)

    @torch.no_grad()
    def regenerate_pool(self, data, x, h, edge_att, train_pos, train_labels, ratio):
        """New pool from min(max(1, diffuse_frac * P), max_pool_size) random training pairs."""
        num_nodes = x.size(0)
        n_sample = min(max(1, int(self.diffuse_frac * train_pos.size(0))), self.max_pool_size)
        idx = torch.randperm(train_pos.size(0))[:n_sample]
        new_pool = []
        for i in idx.tolist():
            u, v = int(train_pos[i, 0]), int(train_pos[i, 1])
            label = float(train_labels[i]) if train_labels is not None else 1.0
            sub = extract_ego_subgraph(data.edge_index, u, v, x, self.k_hops, self.max_nodes, num_nodes)
            if sub is None or sub['A'].sum() == 0:
                continue
            inv_sub, env_sub = split_local_subgraph(sub, h, edge_att, ratio)
            if inv_sub['A'].sum() == 0 or env_sub['A'].sum() == 0:
                continue
            n_env = env_sub['X'].size(0)
            X_env_gen, A_env_gen = self.env_dm.sample((n_env, x.size(1)), (n_env, n_env), num_steps=self.sample_steps,
                                                      device=self.device)
            env_sub_aug = {'A': A_env_gen, 'X': X_env_gen, 'global_ids': env_sub['global_ids']}
            n_inv = inv_sub['X'].size(0)
            has_fgw_refs = len(self.id_cache) > 0 or len(self.ood_cache) > 0
            guidance_fn = None
            if self.attribute_net is not None or has_fgw_refs:
                guidance_fn = self._guidance(bool(label), has_fgw_refs)
            X_inv_gen, A_inv_gen = self.inv_dm.sample((n_inv, x.size(1)), (n_inv, n_inv), num_steps=self.sample_steps,
                                                      guidance_fn=guidance_fn, device=self.device)
            spliced = splice_subgraphs({'A': A_inv_gen, 'X': X_inv_gen, 'global_ids': inv_sub['global_ids']}, env_sub_aug,
                                       r_add=self.r_add)
            spliced['center_local'] = sub['center_local']      # the query pair: the first nodes of the invariant part
            new_pool.append((spliced, label))
        self.pool = new_pool
        return len(new_pool)

    def _guidance(self, target_positive, has_fgw_refs):
        """Inv-DM guidance: attribute-network gradient every step, FGW feature gradient every fgw_guidance_every steps,
        both scaled by guidance_strength * (1 - t)."""
        step = {'i': 0}

        def guidance_fn(X_t, A_t, t):
            strength = self.guidance_strength * (1.0 - float(t))
            grad_x, grad_a = torch.zeros_like(X_t), torch.zeros_like(A_t)
            if self.attribute_net is not None:
                with torch.enable_grad():
                    gx, ga = self.attribute_net.guidance_grad(X_t, A_t, target_is_positive=target_positive, strength=strength)
                grad_x, grad_a = grad_x + gx, grad_a + ga
            if has_fgw_refs and step['i'] % self.fgw_guidance_every == 0:
                with torch.enable_grad():
                    grad_x = grad_x + fgw_guidance_grad({'A': A_t, 'X': X_t}, self.id_cache, self.ood_cache,
                                                        alpha=self.fgw_alpha, epsilon=self.fgw_epsilon, strength=strength)
            step['i'] += 1
            return grad_x, grad_a
        return guidance_fn
