"""GI-Graph (Zhang et al., IEEE TKDE 2025, "GI-Graph: A Generative Invariant Graph Learning Scheme Towards
Out-of-Distribution Generalization"), adapted to transductive link prediction (a graph instance = the ego-subgraph of
a training pair).

On top of CIGA's causal-subgraph link loss (ciga.py) GI-Graph adds
  * a pool loss (`pool_coe`): BCE of the encoder + predictor on diffusion-augmented ego-subgraphs (pool.py), whose
    Env-DM / Inv-DM / attribute networks train every batch with their own Adam (`diffusion_lr`);
  * a prototype loss (`proto_coe`, eq. 15) pulling correctly classified edge representations towards their class
    prototype (attention-weighted running prototypes, eq. 14);
  * query feedback: FGW fine-tuning of Inv-DM on misclassified pairs once both FGW caches are filled; before that a
    hinge repulsion of misclassified representations from the wrong prototype (`mis_coe`);
  * optional CIGAv2 hinge (`spu_coe`) and IRM / V-REx / GroupDRO fusion terms (`irm_coe`, `var_coe`, `groupdro_coe`,
    eq. 24) on the environment labels (envs.py).
`env_coe` multiplies the loss of the diffusion-free Gaussian-augmentation fallback, which is 0 with the diffusion
pipeline on; every paper run had it on and the fallback is not ported.

Per seed the pool, its optimiser and the prototype-initialisation flags are created at the first epoch (CPU RNG for
the diffusion networks' initialisation), as in the paper's runs; the prototypes live in unGCN's `gigraph_proto` buffer.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

from ...train import Hooks
from ..ciga import causal_pass, spurious_hinge
from ..envs import EnvLabels
from .pool import GIGraphPool

DEFAULTS = dict(ratio=0.25, proto_coe=1.0, proto_temp=0.1, mis_coe=1.0, margin=0.2, env_coe=1.0, env_alpha=0.1,
                spu_coe=0.0, irm_coe=0.0, var_coe=0.0, groupdro_coe=0.0, use_diffusion=True, diffuse_every=10,
                diffuse_frac=0.1, max_pool_size=20, pool_batch_size=16, pool_coe=1.0, k_hops=2, max_nodes=30,
                sample_steps=20, diffusion_lr=1e-3, ood_alpha=0.1, guidance_strength=1.0, use_attribute_net=True)


# ------------------------------------------------------------------------------------------------ prototypes
def prototype_loss(rep, labels, prototypes):
    """eq. 15: cross-entropy of the dot-product similarity to the two class prototypes."""
    labels = labels.long().to(rep.device)
    logits = rep @ prototypes.T
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -log_prob.gather(1, labels.view(-1, 1)).mean()


def update_prototypes(prototypes, proto_init, rep, labels, temperature=0.1):
    """eq. 14 (in place): mean of the class's representations at first, then the softmax(cos / tau)-weighted average
    of the batch's class representations w.r.t. the previous prototype."""
    labels = labels.long().to(rep.device)
    for y in (0, 1):
        y_mask = labels == y
        if y_mask.sum() == 0:
            continue
        rep_y = rep[y_mask]
        if not proto_init[y]:
            prototypes[y] = rep_y.mean(dim=0)
            proto_init[y] = True
        else:
            sim = F.cosine_similarity(rep_y, prototypes[y].unsqueeze(0).expand_as(rep_y), dim=1)
            weights = torch.softmax(sim / temperature, dim=0)
            prototypes[y] = (weights.unsqueeze(1) * rep_y).sum(dim=0)


def repulsion_loss(rep, labels, prototypes, margin=0.2):
    """relu(margin - cos(rep, own prototype) + cos(rep, other prototype)), mean."""
    labels = labels.long().to(rep.device)
    rep_n = F.normalize(rep)
    proto_n = F.normalize(prototypes)
    cos_true = (rep_n * proto_n[labels]).sum(dim=1)
    cos_wrong = (rep_n * proto_n[1 - labels]).sum(dim=1)
    return torch.relu(margin - cos_true + cos_wrong).mean()


# ------------------------------------------------------------------------------------------------ one epoch
def pool_loss(model, predictor, pool, pool_batch_size, device):
    """Mean BCE of the predictor at the query pair of up to pool_batch_size random pool instances (own forward each)."""
    out = torch.tensor(0.0, device=device)
    if len(pool) == 0:
        return out
    losses = []
    for i in torch.randperm(len(pool))[:pool_batch_size].tolist():
        spliced, label = pool[i]
        A_local, X_local = spliced['A'], spliced['X']
        row_l, col_l = A_local.nonzero(as_tuple=True)
        if row_l.numel() == 0:
            continue
        adj_local = SparseTensor(row=row_l, col=col_l, value=torch.ones(row_l.numel(), device=X_local.device),
                                 sparse_sizes=(X_local.size(0), X_local.size(0)))
        h_local = model(X_local, adj_local)
        c0, c1 = spliced['center_local'][0], spliced['center_local'][1]
        pred = predictor(x_i=h_local[c0].unsqueeze(0), x_j=h_local[c1].unsqueeze(0))
        pred = torch.nan_to_num(pred, nan=0.5).clamp(1e-6, 1 - 1e-6)
        losses.append(F.binary_cross_entropy(pred, torch.tensor([[label]], device=X_local.device)))
    return torch.stack(losses).mean() if losses else out


def env_fusion(model, batch_loss, c, perm, env_idx, mask, num_envs, loss_coeff, irm_coe, var_coe, groupdro_coe, device):
    """eq. 24: optional V-REx variance, GroupDRO-weighted and IRM gradient-penalty terms over the environments."""
    env_full = torch.cat([env_idx[perm].to(device)] * 2)
    mask_full = torch.cat([mask[perm].to(device)] * 2)
    loss_elem = torch.cat([c['pos_loss_elem'], c['neg_loss_elem']]).view(-1)
    env_losses = []
    for i in range(num_envs + 1):
        m = env_full == i
        if loss_elem[m].shape[0] > 0 and mask_full[m].sum() > 0:
            env_losses.append(loss_elem[m].sum() / mask_full[m].sum())
    if not env_losses:
        return batch_loss
    losses = torch.stack(env_losses)
    if var_coe > 0:
        var_pen = torch.var(losses)
        if not torch.isnan(var_pen):
            batch_loss = batch_loss + var_coe * var_pen
    if groupdro_coe > 0:
        w = torch.ones(losses.shape[0], device=device) * torch.exp(loss_coeff * losses.data)
        batch_loss = batch_loss + groupdro_coe * (losses @ (w / w.sum()))
    if irm_coe > 0:
        dummy = torch.cat([-torch.log(model.dummy_w * c['pos_out_c'] + 1e-15),
                           -torch.log(1 - model.dummy_w * c['neg_out_c'] + 1e-15)]).view(-1)
        pens = []
        for i in range(num_envs + 1):
            m = env_full == i
            if dummy[m].shape[0] > 0 and mask_full[m].sum() > 0:
                g = torch.autograd.grad(dummy[m].sum() / mask_full[m].sum(), model.dummy_w, create_graph=True)[0]
                pens.append(torch.sum(g.pow(2)))
        if pens:
            irm_pen = sum(pens) / len(pens)
            if not torch.isnan(irm_pen):
                batch_loss = batch_loss + irm_coe * irm_pen
    return batch_loss


def train_epoch(model, predictor, train_pos, data, optimizer, batch_size, num_envs, env_idx, mask, loss_coeff, state,
                epoch, o, neg_dst=None):
    """state: per-seed dict ('pool', 'opt', 'proto_init'); o: the method options (DEFAULTS overridden by the config)."""
    model.train()
    predictor.train()
    x = data.x
    dev = x.device
    total_loss = total_examples = 0
    has_envs = env_idx.numel() == train_pos.size(0) and torch.unique(env_idx).numel() > 1
    if o['use_diffusion']:
        if 'pool' not in state:
            state['pool'] = GIGraphPool(feat_dim=x.size(1), ood_alpha=o['ood_alpha'], k_hops=o['k_hops'],
                                        max_nodes=o['max_nodes'], diffuse_frac=o['diffuse_frac'],
                                        max_pool_size=o['max_pool_size'], sample_steps=o['sample_steps'],
                                        use_attribute_net=o['use_attribute_net'], guidance_strength=o['guidance_strength'],
                                        device=dev)
            state['opt'] = torch.optim.Adam(state['pool'].diffusion_model_params(), lr=o['diffusion_lr'])
        gp, dopt = state['pool'], state['opt']
        # pool candidates: every training positive (label 1) and one uniform pair per positive (label 0)
        pool_neg = torch.randint(0, x.size(0), train_pos.t().size(), dtype=torch.long, device=dev).t()
        pool_edges = torch.cat([train_pos, pool_neg], dim=0)
        pool_labels = torch.cat([torch.ones(train_pos.size(0), device=dev), torch.zeros(pool_neg.size(0), device=dev)])
        if epoch is not None and o['diffuse_every'] > 0 and epoch % o['diffuse_every'] == 0:
            with torch.no_grad():
                h_for_pool = model(x, data.adj_t)
            gp.regenerate_pool(data, x, h_for_pool, model.edge_att, pool_edges, pool_labels, o['ratio'])
    proto_init = model.gigraph_proto_init
    for perm in DataLoader(range(train_pos.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        edge = train_pos[perm].t()
        c = causal_pass(model, predictor, x, data, edge, o['ratio'], neg_dst)
        h, h_c, neg_edge = c['h'], c['h_c'], c['neg_edge']
        pos_out_c, neg_out_c = c['pos_out_c'], c['neg_out_c']
        pred_loss = c['pos_loss_elem'].mean() + c['neg_loss_elem'].mean()
        env_pred_loss = torch.tensor(0.0, device=h.device)     # diffusion-free fallback term: 0 with diffusion on
        p_loss = torch.tensor(0.0, device=h.device)
        if o['use_diffusion']:
            score_loss = gp.train_step(data, x, h, model.edge_att, pool_edges, pool_labels, o['ratio'])
            if score_loss.requires_grad:
                dopt.zero_grad()
                score_loss.backward()
                dopt.step()
            p_loss = pool_loss(model, predictor, gp.pool, o['pool_batch_size'], h.device)
        pair_rep = torch.cat([h_c[edge[0]] * h_c[edge[1]], h_c[neg_edge[0]] * h_c[neg_edge[1]]], dim=0)
        pair_label = torch.cat([torch.ones(pos_out_c.size(0), device=h.device), torch.zeros(neg_out_c.size(0), device=h.device)])
        correct_mask = torch.cat([(pos_out_c.detach() > 0.5).view(-1), (neg_out_c.detach() <= 0.5).view(-1)])
        correct_reps, correct_labels = pair_rep[correct_mask], pair_label[correct_mask]
        mis_reps, mis_labels = pair_rep[~correct_mask], pair_label[~correct_mask]
        has_both_protos = bool(proto_init[0]) and bool(proto_init[1])
        proto_l = torch.tensor(0.0, device=h.device)
        if has_both_protos and correct_reps.size(0) > 0:
            proto_l = prototype_loss(correct_reps, correct_labels, model.gigraph_proto.detach())
        mis_l = torch.tensor(0.0, device=h.device)
        fgw_active = False
        if o['use_diffusion']:
            gp.update_fgw_caches(data, x, h.detach(), model.edge_att, o['ratio'], torch.cat([edge, neg_edge], dim=1),
                                 correct_mask, pair_label)
            if len(gp.id_cache) > 0 and len(gp.ood_cache) > 0:
                fgw_active = True
                if gp.batch_counter % gp.fgw_finetune_every == 0:
                    fgw_l = gp.fgw_finetune_loss()
                    if fgw_l.requires_grad:
                        dopt.zero_grad()
                        fgw_l.backward()
                        dopt.step()
                gp.batch_counter += 1
        if not fgw_active and has_both_protos and mis_reps.size(0) > 0:
            mis_l = repulsion_loss(mis_reps, mis_labels, model.gigraph_proto.detach(), o['margin'])
        batch_loss = (pred_loss + o['env_coe'] * env_pred_loss + o['proto_coe'] * proto_l + o['mis_coe'] * mis_l
                      + o['pool_coe'] * p_loss)
        if o['spu_coe'] > 0:
            h_s = model(x, c['spu_adj_t']).detach()
            pos_out_s = predictor(x_i=h_s[edge[0]], x_j=h_s[edge[1]])
            neg_out_s = predictor(x_i=h_s[neg_edge[0]], x_j=h_s[neg_edge[1]])
            batch_loss = batch_loss + o['spu_coe'] * spurious_hinge(pos_out_s, neg_out_s, c['pos_loss_elem'], c['neg_loss_elem'])
        if has_envs and (o['irm_coe'] > 0 or o['var_coe'] > 0 or o['groupdro_coe'] > 0):
            batch_loss = env_fusion(model, batch_loss, c, perm, env_idx, mask, num_envs, loss_coeff, o['irm_coe'],
                                    o['var_coe'], o['groupdro_coe'], h.device)
        batch_loss.backward()
        with torch.no_grad():
            update_prototypes(model.gigraph_proto, proto_init, correct_reps.detach(), correct_labels, temperature=o['proto_temp'])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        total_loss += batch_loss.item() * pos_out_c.size(0)
        total_examples += pos_out_c.size(0)
    return total_loss / total_examples


def hooks(opts):
    o = {**DEFAULTS, **{k: v for k, v in opts.items() if k in DEFAULTS}}
    assert o['use_diffusion'], 'only the diffusion pipeline is ported'
    envs = EnvLabels(opts.get('num_envs', 3))

    def step(ctx, epoch, extra_loss_fn):
        e = ctx.extra
        return train_epoch(ctx.model, ctx.predictor, ctx.edges[0], ctx.graph.data, ctx.optimizer, ctx.batch_sizes[0],
                           e['num_envs'], e['env_idx'], e['mask'], opts.get('loss_coeff', 1.0), e.setdefault('gigraph', {}),
                           epoch, o, ctx.split.typed_neg_dst)
    return Hooks(setup=envs, train_step=step)
