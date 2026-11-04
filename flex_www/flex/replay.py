"""The AUG encoder: replay of train-graph patches as an auxiliary loss, and the combined epoch selection.

Replay. Every unique train positive (u, v) is replayed once per epoch (one reshuffled pass, AUX_B anchors per training
batch) as a small graph: the subgraph of the train graph induced on u, v and their neighbours, with the anchor edge
present. The encoder embeds the batch of patches, the predictor scores the anchor pairs, and AUX_W * BCE(anchor = 1) is
added to the training loss. On LPShift cells whose test lies where a 1-hop patch can be edited toward it, patches are
edited along the cell's shift (validation fold = direction.threshold on validation positives):
  PA down  degree-capped: each patch draws an endpoint-degree pair from the fold; each anchor keeps min(own, drawn)
           random neighbours
  SP down  path-cut (AUG rows only): with probability = the fold's share at distance >= 5, neighbours are split between
           the anchors' sides and every cross-side edge except the anchor edge is deleted
All other cells replay unedited patches. The patch cache and the NumPy streams persist across seeds and are rebuilt only
when the train graph changes (Amazon: one split per seed).

Combined selection. Each evaluation is scored as validation metric + the same metric on the validation fold along the
shift (direction.threshold; CN = 0 on non-LPShift cells); the encoder at the first maximum is the one FLEX fuses.
"""
import numpy as np
import scipy.sparse as ss
import torch
from torch_sparse import SparseTensor

from . import direction as DIR
from . import features as F
from .channels import PSEUDO_SEED, graph_of
from ..evaluate import predict

AUX_W, AUX_B = 1.0, 256


class Replay:
    """Per-row replay state. cell: cell_info dict; pa_only: AG rows edit PA-down cells only (no SP path-cut);
    sp: SP columns in the direction features (gated rows)."""

    def __init__(self, cell, pa_only=False, sp=False, aux_w=AUX_W, aux_b=AUX_B, logits=False):
        self.cell, self.pa_only, self.sp, self.aux_w, self.aux_b, self.logits = cell, pa_only, sp, aux_w, aux_b, logits
        self.st = {}
        self.rebuilds = 0
        self.info = None

    def _build(self, data, valid_pos):
        st = self.st
        st['A'] = graph_of(data.adj_t)
        coo = st['A'].tocoo()
        st['P'] = F.canon(np.c_[coo.row, coo.col])
        st['rng'] = np.random.default_rng(PSEUDO_SEED)
        st['edit'] = None
        st['erng'] = np.random.default_rng(PSEUDO_SEED + 5)
        sp = DIR.spec(self.cell)
        if sp is not None and sp[1] < 0 and sp[0] in (('PA',) if self.pa_only else ('PA', 'SP')):
            vp = valid_pos.detach().cpu().numpy().reshape(-1, 2)
            m, _, info = DIR.threshold(sp, st['A'], vp, self.sp)
            st['edit'] = ('PA', DIR.fold_degree_pairs(st['A'], vp, m)) if sp[0] == 'PA' else ('SP', info['fold_far_share'])
            self.info = {'edit': 'degree-capped' if sp[0] == 'PA' else 'path-cut', **info}

    def loss_factory(self, model, predictor, data, valid_pos):
        """Called once per epoch (before the training pass): returns the epoch's aux loss fn(h, adj, ...)."""
        st = self.st
        if 'A' in st:
            A_new = graph_of(data.adj_t)
            if A_new.shape != st['A'].shape or A_new.nnz != st['A'].nnz or (A_new != st['A']).nnz:
                st.clear()
                self.rebuilds += 1
        if 'A' not in st:
            self._build(data, valid_pos)
        perm = st['rng'].permutation(len(st['P']))
        pos = [0]
        xsrc = data.x

        def aux_loss(h, adj, pos_edge, pos_edge_t, pos_out):
            if pos[0] >= len(perm):
                return None
            sel = st['P'][perm[pos[0]:pos[0] + self.aux_b]]
            pos[0] += self.aux_b
            A = st['A']
            xs, rows, cols, anc, off = [], [], [], [], 0
            for u, v in sel:
                nu, nv = A.indices[A.indptr[u]:A.indptr[u + 1]], A.indices[A.indptr[v]:A.indptr[v + 1]]
                ed = st['edit']
                cut = False
                if ed is not None and ed[0] == 'PA':
                    du, dv = ed[1][st['erng'].integers(len(ed[1]))]
                    er = st['erng']
                    nu, nv = nu[(nu != u) & (nu != v)], nv[(nv != u) & (nv != v)]
                    nu = er.choice(nu, min(len(nu), int(du)), replace=False) if len(nu) else nu
                    nv = er.choice(nv, min(len(nv), int(dv)), replace=False) if len(nv) else nv
                elif ed is not None and ed[0] == 'SP':
                    cut = st['erng'].random() < ed[1]
                nodes = np.unique(np.r_[u, v, nu, nv])
                S = A[nodes][:, nodes].tocoo()
                if cut:          # neighbours to one anchor's side (shared ones at random); drop cross-side edges
                    iu, iv = np.searchsorted(nodes, u), np.searchsorted(nodes, v)
                    inu, inv_ = np.isin(nodes, nu), np.isin(nodes, nv)
                    side = np.where(inu & inv_, st['erng'].random(len(nodes)) < .5, inv_ & ~inu).astype(int)
                    side[iu], side[iv] = 0, 1
                    anchor = ((S.row == iu) & (S.col == iv)) | ((S.row == iv) & (S.col == iu))
                    keep = (side[S.row] == side[S.col]) | anchor
                    S = ss.coo_matrix((S.data[keep], (S.row[keep], S.col[keep])), shape=S.shape)
                rows.append(S.row + off)
                cols.append(S.col + off)
                anc.append([off + np.searchsorted(nodes, u), off + np.searchsorted(nodes, v)])
                xs.append(nodes)
                off += len(nodes)
            ids = torch.as_tensor(np.concatenate(xs).astype(np.int64), device=h.device)
            r = torch.as_tensor(np.concatenate(rows).astype(np.int64), device=h.device)
            c = torch.as_tensor(np.concatenate(cols).astype(np.int64), device=h.device)
            adj_b = SparseTensor(row=r, col=c, sparse_sizes=(off, off)).coalesce()
            hp = model(xsrc[ids], adj_b)
            hp = hp[0] if isinstance(hp, tuple) else hp
            an = torch.as_tensor(np.array(anc), device=h.device)
            if getattr(predictor, 'needs_adj', False):          # NCN: reads the patch adjacency, emits logits
                po = predict(predictor, hp, an.t(), adj_b).view(-1)
                return -torch.nn.functional.logsigmoid(po).mean() * self.aux_w
            po = predictor(x_i=hp[an[:, 0]], x_j=hp[an[:, 1]]).view(-1)
            if self.logits:                                      # Refined-GAE: logits
                return -torch.nn.functional.logsigmoid(po).mean() * self.aux_w
            return -torch.log(po + 1e-15).mean() * self.aux_w
        return aux_loss


def selection_fold(cell, A, valid_pos, sp=False):
    """Validation positives of the combined selection: the direction fold on LPShift cells, else CN = 0 on the train graph."""
    pv = np.asarray(valid_pos, np.int64).reshape(-1, 2)
    axis = DIR.spec(cell)
    if axis is None:
        return np.asarray(A[pv[:, 0]].multiply(A[pv[:, 1]]).sum(1)).ravel() == 0
    return DIR.threshold(axis, A, pv, sp)[0]


def fold_metric(mf, P, mask, metric):
    """Validation metric on the fold positives and their own negatives (0.0 on an empty fold or any failure)."""
    idx = torch.as_tensor(np.flatnonzero(mask))
    pv, nv = P[1], P[2]
    if len(idx) == 0:
        return 0.0
    Q = list(P)
    Q[1], Q[2] = pv[idx], mf.sub_neg(nv, pv.numel(), idx)
    try:
        return mf.value(Q, metric, 1)
    except Exception:
        return 0.0
