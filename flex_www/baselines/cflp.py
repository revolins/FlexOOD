"""CFLP (Zhao et al., ICML 2022, "Learning from Counterfactual Links for Link Prediction") on the shared GCN backbone.

Treatment T(i, j) = the cell's shift heuristic (CN / PA / SP by dataset tag, else CN) on the train graph, thresholded at
the median over the training positives (positives scored with their own edge removed). For every node pair the
counterfactual link is found by walking the two nodes' nearest-neighbour lists (Euclidean on DGI embeddings) for the
closest pair with the opposite treatment within the gamma-th distance percentile (cf_utils.get_CF). The decoder is
mlp_score plus a zero-initialised projection of T on its first layer (= the reference cat(z_i * z_j, T) decoder).

Loss = ERM link loss + `loss_coeff` (alpha) * weighted BCE on the counterfactual links (T_cf) + `beta` * the 'lin'
discrepancy between random factual and counterfactual pair representations. The last `ft_frac` of the epoch budget
(or from the epoch joint training would early-stop) fine-tunes the decoder alone from the best joint checkpoint with
the encoder frozen.

The treatment, DGI and walk are computed under a forked RNG (DGI seeded 0, the walk's subsample generator seeded 0),
so the global RNG stream is ERM's; the discrepancy pairs come from a generator seeded torch.initial_seed() + 31337.
Memory: the walk keeps N x N float64 / int32 / uint8 buffers on the GPU (16 N^2 bytes).
"""
import hashlib
import time

import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.gcn import unGCN
from ..models.predictor import mlp_score
from ..train import Hooks, typed_negatives
from .envs import cn_scores, sp_scores
from .vgae import preserve_rng

_PREP_CACHE = {}                  # per process: the precompute is identical for every seed of a cell


class CFLPScore(mlp_score):
    """mlp_score whose first layer also sees T: lins[0](z_i * z_j) + t_proj(T). `needs_adj` routes evaluation through
    forward(h, adj, edge), which looks T up in the dense train-graph treatment matrix `self.T`."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
        with preserve_rng():
            self.t_proj = torch.nn.Linear(1, self.lins[0].out_features, bias=False)
        torch.nn.init.zeros_(self.t_proj.weight)
        self.needs_adj = True
        self.T = None

    def reset_parameters(self):
        super().reset_parameters()
        with torch.no_grad():
            self.t_proj.weight.zero_()

    def forward(self, *args, x_i=None, x_j=None, t=None, logits=False):
        if args:                                   # evaluate.predict(): (h, adj, edge [2, M])
            h, _adj, edge = args
            x_i, x_j = h[edge[0]], h[edge[1]]
            t = self.T[edge[0], edge[1]]
        x = self.lins[0](x_i * x_j)
        if t is not None:
            x = x + self.t_proj(t.to(x.dtype).unsqueeze(-1))
        if len(self.lins) > 1:
            x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
            for lin in self.lins[1:-1]:
                x = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
            x = self.lins[-1](x)
        return x if logits else torch.sigmoid(x)


def build(ctx, input_channel):
    hp = ctx.spec.hp
    model = unGCN(input_channel, hp['hidden_channels'], hp['hidden_channels'], hp['num_layers'], hp['dropout'],
                  hp.get('edge_drop', 0.0), 'CFLP').to(ctx.device)
    predictor = CFLPScore(hp['hidden_channels'], hp['hidden_channels'], 1, hp['num_layers_predictor'], hp['dropout']).to(ctx.device)
    model.reset_parameters()
    predictor.reset_parameters()
    return model, predictor


# ------------------------------------------------------------------------------------------------ treatment
def heuristic_tag(dataset):
    """SP, then CN, then PA (last match wins); CN without a tag (the environment heuristics' precedence)."""
    tag = 'CN'
    for t in ('SP', 'CN', 'PA'):
        if t in dataset:
            tag = t
    return tag


def train_graph(edge_index, n):
    """Symmetric 0/1 CSR of the message-passing graph without self-loops."""
    ei = edge_index.cpu().numpy()
    A = ssp.csr_matrix((np.ones(ei.shape[1], dtype=np.float32), (ei[0], ei[1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def positive_scores(tag, A, pos_train_edge):
    """Heuristic score of every training positive with its own edge removed (CN needs no removal without self-loops;
    PA uses (d_u - a_uv)(d_v - a_uv); SP removes the edge)."""
    e = pos_train_edge.t().cpu()
    if tag == 'CN':
        return cn_scores(A, e).numpy().astype(np.float64)
    if tag == 'PA':
        deg = np.asarray(A.sum(1)).ravel()
        u, v = e[0].numpy().copy(), e[1].numpy().copy()
        a_uv = np.asarray(A[u, v]).ravel()
        return ((deg[u] - a_uv) * (deg[v] - a_uv)).astype(np.float32).astype(np.float64)
    return sp_scores(A, e, True).numpy().astype(np.float64)


def fill_treatment(T_f, tag, A, cmp, device, rows=1024):
    """Dense T_f[i, j] = cmp(score(i, j)) on the train graph, in row blocks (SP: 1/hops as float32, 1/999 unreachable)."""
    n = A.shape[0]
    if tag == 'PA':
        deg = torch.as_tensor(np.asarray(A.sum(1)).ravel(), dtype=torch.float64, device=device)
        for r0 in range(0, n, rows):
            r1 = min(n, r0 + rows)
            T_f[r0:r1] = cmp(deg[r0:r1, None] * deg[None, :]).to(torch.uint8)
        return
    Ad = torch.as_tensor(A.toarray(), dtype=torch.float32, device=device)
    if tag == 'CN':
        for r0 in range(0, n, rows):
            r1 = min(n, r0 + rows)
            T_f[r0:r1] = cmp((Ad[r0:r1] @ Ad).double()).to(torch.uint8)
        return
    sc = lambda s: torch.tensor(np.float32(1.0 / s), dtype=torch.float64)  # noqa: E731
    if bool(cmp(sc(999))):
        T_f.fill_(1)
        return
    kmax = 0
    while kmax + 1 < 999 and bool(cmp(sc(kmax + 1))):
        kmax += 1
    T_f.zero_()
    if kmax == 0:
        return
    Ah = Ad.half()
    del Ad
    for r0 in range(0, n, rows):
        r1 = min(n, r0 + rows)
        reach = torch.zeros(r1 - r0, n, dtype=torch.bool, device=device)
        reach[torch.arange(r1 - r0, device=device), torch.arange(r0, r1, device=device)] = True
        frontier = reach.clone()
        for _ in range(kmax):
            frontier = ((frontier.half() @ Ah) > 0) & ~reach
            if not bool(frontier.any()):
                break
            reach |= frontier
            T_f[r0:r1][frontier] = 1


# ------------------------------------------------------------------------------------------------ DGI embeddings
class _DGIEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.prelu = torch.nn.PReLU()

    def forward(self, x, edge_index):
        return self.prelu(self.conv2(self.conv(x, edge_index), edge_index))


def dgi_embed(x, adj_t, hidden, epochs, device):
    """Deep Graph Infomax node embeddings (2-layer GCN + PReLU, Adam 1e-3), seeded 0 inside a forked RNG."""
    from torch_geometric.nn.models import DeepGraphInfomax
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        torch.manual_seed(0)
        model = DeepGraphInfomax(hidden_channels=hidden, encoder=_DGIEncoder(x.size(1), hidden),
                                 summary=lambda z, *a, **k: torch.sigmoid(z.mean(dim=0)),
                                 corruption=lambda x, ei: (x[torch.randperm(x.size(0))], ei)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            pos_z, neg_z, summary = model(x, adj_t)
            model.loss(pos_z, neg_z, summary).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            z = model.encoder(x, adj_t)
    return z.detach()


# ------------------------------------------------------------------------------------------------ counterfactual walk
def sorted_distances(z, sd, nns, gamma, rows=2048):
    """Fill sd / nns with each row's ascending Euclidean distances / node ids and return the gamma-percentile threshold,
    with get_CF's own host arithmetic (scipy cdist, percentile before the diagonal fill, np.argsort), so ties and the
    last ulp match the reference."""
    from scipy.spatial.distance import cdist
    zn = z.double().cpu().numpy()
    simi = cdist(zn, zn, 'euclidean')
    thresh = float(np.percentile(simi, gamma))
    np.fill_diagonal(simi, np.max(simi) + 1)
    n = simi.shape[0]
    for r0 in range(0, n, rows):
        r1 = min(n, r0 + rows)
        idx = np.argsort(simi[r0:r1], axis=1)
        nns[r0:r1] = torch.from_numpy(idx.astype(np.int32)).to(nns.device)
        sd[r0:r1] = torch.from_numpy(np.take_along_axis(simi[r0:r1], idx, 1)).to(sd.device)
    del simi
    return thresh


def cf_walk(sd, nns, T_f, adj, thresh, gen, chunk_pairs=1 << 24, keep_cap=4_000_000):
    """get_CF_single for every pair a < b, in lockstep on the GPU: stop with the factual values once
    d(a, nn_a[i]) + d(b, nn_b[j]) > 2 * thresh, with a counterfactual once T_f[nn_a[i], nn_b[j]] != T_f[a, b], else
    advance the list whose next distance is smaller (j on ties); exhausted pairs keep zeros. Mirrored to the lower
    triangle. The counterfactual pairs for the discrepancy loss are a Bernoulli subsample of at most ~keep_cap."""
    n = sd.size(0)
    sdf, nnf, tff, adf = sd.view(-1), nns.view(-1), T_f.view(-1), adj.view(-1)
    T_cf, adj_cf = torch.zeros_like(T_f), torch.zeros_like(T_f)
    tcf, acf = T_cf.view(-1), adj_cf.view(-1)
    dev = sd.device
    total = n * (n - 1) // 2
    q = min(1.0, keep_cap / max(total, 1))
    kept = []
    two_t = 2.0 * thresh
    rows_left = torch.arange(n - 1, 0, -1, device='cpu')
    a0 = 0
    t_start, done_pairs = time.time(), 0
    while a0 < n - 1:
        cum = torch.cumsum(rows_left[a0:], 0)
        a1 = min(n - 1, a0 + max(1, int((cum <= chunk_pairs).sum())))
        a_rows = torch.arange(a0, a1, device=dev)
        counts = torch.arange(n - 1 - a0, n - 1 - a1, -1, device=dev)
        a = torch.repeat_interleave(a_rows, counts)
        start = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
        b = a + 1 + (torch.arange(a.numel(), device=dev) - start)
        del start
        chunk = a.numel()
        ia, jb = a * n, b * n
        ab = ia + b
        while a.numel():
            far = (sdf[ia] + sdf[jb]) > two_t
            na, nb = nnf[ia].long(), nnf[jb].long()
            tab = tff[ab]
            found = (~far) & (tff[na * n + nb] != tab)
            f_idx = ab[far]
            tcf[f_idx] = tff[f_idx]
            acf[f_idx] = adf[f_idx]
            g_idx = ab[found]
            fna, fnb = na[found], nb[found]
            tcf[g_idx] = 1 - tab[found]
            acf[g_idx] = adf[fna * n + fnb]
            if q >= 1.0:
                kept.append(torch.stack([fna, fnb], 1))
            else:
                keep = torch.rand(fna.numel(), generator=gen, device=dev) < q
                kept.append(torch.stack([fna[keep], fnb[keep]], 1))
            cont = ~(far | found)
            a, b, ia, jb, ab = a[cont], b[cont], ia[cont], jb[cont], ab[cont]
            adv_i = sdf[ia + 1] < sdf[jb + 1]
            ia = ia + adv_i
            jb = jb + (~adv_i)
            inloop = ((ia - a * n) < n - 1) & ((jb - b * n) < n - 1)
            a, b, ia, jb, ab = a[inloop], b[inloop], ia[inloop], jb[inloop], ab[inloop]
        done_pairs += chunk
        print(f'CFLP walk: {done_pairs}/{total} pairs, {time.time() - t_start:.0f}s', flush=True)
        a0 = a1
    T_cf |= T_cf.t().clone()
    adj_cf |= adj_cf.t().clone()
    cf_pairs = torch.cat(kept, 0) if kept else torch.empty(0, 2, dtype=torch.long, device=dev)
    return T_cf, adj_cf, cf_pairs


def _digest(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def prepare(data, edge_index, pos_train_edge, dataset, device, gamma=30.0, dgi_epochs=2000, dgi_hidden=256, x_d=False):
    """{'T_f', 'T_cf', 'adj_cf', 'cf_pairs', 'n'} for a cell, under a forked CPU + CUDA RNG (cached per process)."""
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        n = int(data.num_nodes)
        key = (dataset, bool(x_d), n, gamma, dgi_epochs, dgi_hidden, _digest(edge_index, pos_train_edge, data.x))
        if key in _PREP_CACHE:
            return _PREP_CACHE[key]
        # dense buffers first, so a graph that cannot hold them fails before DGI training
        sd = torch.empty(n, n, dtype=torch.float64, device=device)
        nns = torch.empty(n, n, dtype=torch.int32, device=device)
        T_f = torch.zeros(n, n, dtype=torch.uint8, device=device)
        adj = torch.zeros(n, n, dtype=torch.uint8, device=device)
        tag = heuristic_tag(dataset)
        A = train_graph(edge_index, n)
        pos = positive_scores(tag, A, pos_train_edge)
        m = float(np.median(pos))
        t_pos = pos > m
        cmp = lambda s: s > m  # noqa: E731
        if t_pos.all() or not t_pos.any():          # degenerate median split: >= instead of >
            t_pos = pos >= m
            cmp = lambda s: s >= m  # noqa: E731
        fill_treatment(T_f, tag, A, cmp, device)
        pe = pos_train_edge.to(device)
        tp = torch.as_tensor(t_pos.astype(np.uint8), device=device)
        T_f[pe[:, 0], pe[:, 1]] = tp
        T_f[pe[:, 1], pe[:, 0]] = tp
        coo = A.tocoo()
        adj[torch.as_tensor(coo.row, device=device).long(), torch.as_tensor(coo.col, device=device).long()] = 1
        z = dgi_embed(data.x.float(), data.adj_t, dgi_hidden, dgi_epochs, device)
        thresh = sorted_distances(z, sd, nns, gamma)
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        T_cf, adj_cf, cf_pairs = cf_walk(sd, nns, T_f, adj, thresh, gen)
        del sd, nns, adj
        torch.cuda.empty_cache()
        prep = dict(T_f=T_f, T_cf=T_cf, adj_cf=adj_cf, cf_pairs=cf_pairs, n=n)
        _PREP_CACHE[key] = prep
        return prep


# ------------------------------------------------------------------------------------------------ training
class CFLPState:
    """Per seed: the precompute, the joint -> decoder fine-tune switch and the discrepancy-pair generator."""

    def __init__(self, prep, alpha, beta, num_np, ft_frac, epochs, device):
        self.prep = prep
        self.alpha, self.beta, self.num_np = float(alpha), float(beta), int(num_np)
        self.R = max(1, int(epochs * ft_frac)) if ft_frac > 0 else 0
        self.phase = 'joint'
        self.snap = self.h = self.ft_opt = None
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(torch.initial_seed()) + 31337)

    def on_improve(self, model, predictor):
        if self.phase == 'joint' and self.R > 0:
            self.snap = ({k: v.detach().clone() for k, v in model.state_dict().items()},
                         {k: v.detach().clone() for k, v in predictor.state_dict().items()})

    def should_switch(self, epoch, best_epoch, kill_cnt, epochs):
        return self.phase == 'joint' and self.R > 0 and (epoch - best_epoch > kill_cnt or epoch >= epochs - self.R)

    def switch(self, model, predictor, data, lr, l2):
        """Restore the best joint checkpoint, freeze the encoder, cache its embeddings, new Adam for the decoder."""
        if self.snap is not None:
            model.load_state_dict(self.snap[0])
            predictor.load_state_dict(self.snap[1])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            self.h = model(data.x, data.adj_t).detach()
        self.ft_opt = torch.optim.Adam(predictor.parameters(), lr=lr, weight_decay=l2)
        self.phase = 'ft'


def disc_lin(h, np_f, np_cf):
    """cf_utils.calc_disc('lin'): distance between the mean factual and counterfactual pair representations."""
    X_f = torch.cat((h[np_f[:, 0]], h[np_f[:, 1]]), 1)
    X_cf = torch.cat((h[np_cf[:, 0]], h[np_cf[:, 1]]), 1)
    return torch.sqrt(F.mse_loss(X_f.mean(0), X_cf.mean(0)) + 1e-6)


def train_epoch(model, predictor, train_pos, data, optimizer, batch_size, state, neg_dst=None):
    p = state.prep
    T_f, T_cf, adj_cf, n = p['T_f'], p['T_cf'], p['adj_cf'], p['n']
    ft = state.phase == 'ft'
    model.eval() if ft else model.train()
    predictor.train()
    x = data.x
    opt = state.ft_opt if ft else optimizer
    total_loss = total_examples = 0
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        opt.zero_grad()
        h = state.h if ft else model(x, data.adj_t)
        pos_edge = train_pos[perm].t()
        pos_out = predictor(x_i=h[pos_edge[0]], x_j=h[pos_edge[1]], t=T_f[pos_edge[0], pos_edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_edge = typed_negatives(pos_edge, x.size(0), neg_dst, device=h.device)
        neg_out = predictor(x_i=h[neg_edge[0]], x_j=h[neg_edge[1]], t=T_f[neg_edge[0], neg_edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        if not ft and state.alpha != 0.0:
            pairs = torch.cat([pos_edge, neg_edge], 1)
            y_cf = adj_cf[pairs[0], pairs[1]].float()
            logit_cf = predictor(x_i=h[pairs[0]], x_j=h[pairs[1]], t=T_cf[pairs[0], pairs[1]], logits=True).view(-1)
            npos = float(y_cf.sum())
            pw = (y_cf.numel() - npos) / npos if 0 < npos < y_cf.numel() else 1.0
            loss = loss + state.alpha * F.binary_cross_entropy_with_logits(logit_cf, y_cf, pos_weight=torch.tensor(pw, device=h.device))
        if not ft and state.beta != 0.0 and p['cf_pairs'].size(0) > 0:
            k = min(state.num_np, p['cf_pairs'].size(0))
            a = torch.randint(0, n, (state.num_np,), generator=state.gen, device=h.device)
            b = (a + torch.randint(1, n, (state.num_np,), generator=state.gen, device=h.device)) % n
            sel = torch.randperm(p['cf_pairs'].size(0), generator=state.gen, device=h.device)[:k]
            loss = loss + state.beta * disc_lin(h, torch.stack([a, b], 1), p['cf_pairs'][sel])
        loss.backward()
        if not ft:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * pos_out.size(0)
        total_examples += pos_out.size(0)
    return total_loss / total_examples


def precompute(ctx, opts):
    s = ctx.split
    return prepare(ctx.graph.data, ctx.graph.edge_index, ctx.edges[0], s.dataset, ctx.device, gamma=opts.get('gamma', 30.0),
                   dgi_epochs=opts.get('dgi_epochs', 2000), dgi_hidden=opts.get('dgi_hidden', 256), x_d=ctx.spec.x_d)


def hooks(opts):
    def setup(ctx):
        prep = precompute(ctx, opts)
        ctx.predictor.T = prep['T_f']
        ctx.extra['cflp'] = CFLPState(prep, opts['loss_coeff'], opts.get('beta', 1.0), opts.get('num_np', 1000),
                                      opts.get('ft_frac', 0.2), ctx.spec.epochs, ctx.device)

    def step(ctx, epoch, extra_loss_fn):
        return train_epoch(ctx.model, ctx.predictor, ctx.edges[0], ctx.graph.data, ctx.optimizer, ctx.batch_sizes[0],
                           ctx.extra['cflp'], ctx.split.typed_neg_dst)

    def on_improve(ctx, epoch):
        ctx.extra['cflp'].on_improve(ctx.model, ctx.predictor)

    def stop(ctx, epoch, best_epoch):
        # joint training does not early-stop at the switch point: the decoder fine-tune starts there instead
        st, s = ctx.extra['cflp'], ctx.spec
        return epoch - best_epoch > s.kill_cnt and not st.should_switch(epoch, best_epoch, s.kill_cnt, s.epochs)

    def after_eval(ctx, epoch, best_epoch):
        st, s = ctx.extra['cflp'], ctx.spec
        if st.should_switch(epoch, best_epoch, s.kill_cnt, s.epochs):
            st.switch(ctx.model, ctx.predictor, ctx.graph.data, s.hp['lr'], s.hp.get('l2', 0.0))
            return epoch
        return None
    return Hooks(build_models=build, setup=setup, train_step=step, on_improve=on_improve, stop=stop, after_eval=after_eval)


def before_load(ctx, opts):
    """Weights mode: the decoder reads T at evaluation."""
    ctx.predictor.T = precompute(ctx, opts)['T_f']
