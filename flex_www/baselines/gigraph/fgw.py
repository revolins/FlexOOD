"""Fused Gromov-Wasserstein query feedback for GI-Graph (paper eq. 21-22): partially denoise a misclassified invariant
subgraph with Inv-DM and move it towards the correctly classified ones and away from the misclassified ones (entropic
FGW from POT, hop-distance structure matrices)."""
import ot.gromov as ot_gromov
import torch

from .diffusion import forward_noise, reverse_step


def structure_matrix(A, X):
    """All-pairs hop distances of a small dense graph (unreachable = n); feature distances when it has no edges."""
    n = A.size(0)
    if A.sum() == 0:
        return torch.cdist(X, X)
    INF = float(n + 1)
    dist = torch.full((n, n), INF, device=A.device)
    dist.fill_diagonal_(0.0)
    adj_list = [torch.nonzero(A[i] > 0.5).view(-1).tolist() for i in range(n)]
    for src in range(n):
        visited, frontier, d = {src}, [src], 0
        while frontier:
            d += 1
            nxt = []
            for node in frontier:
                for nbr in adj_list[node]:
                    if nbr not in visited:
                        visited.add(nbr)
                        dist[src, nbr] = float(d)
                        nxt.append(nbr)
            frontier = nxt
    return torch.where(dist == INF, torch.full_like(dist, float(n)), dist)


def fgw_distance(sub_a, sub_b, alpha=0.5, epsilon=1.0, max_iter=20):
    orig_device = sub_a['X'].device
    X_a, X_b = sub_a['X'].cpu(), sub_b['X'].cpu()
    C_a, C_b = structure_matrix(sub_a['A'].cpu(), X_a), structure_matrix(sub_b['A'].cpu(), X_b)
    p_a = torch.ones(X_a.size(0)) / X_a.size(0)
    p_b = torch.ones(X_b.size(0)) / X_b.size(0)
    M = torch.cdist(X_a, X_b)
    C_a = C_a / C_a.max().clamp(min=1e-06)
    C_b = C_b / C_b.max().clamp(min=1e-06)
    M = M / M.max().clamp(min=1e-06)
    dist = ot_gromov.entropic_fused_gromov_wasserstein2(M, C_a, C_b, p_a, p_b, alpha=alpha, epsilon=epsilon, max_iter=max_iter)
    return torch.nan_to_num(dist, nan=0.0, posinf=1.0, neginf=0.0).to(orig_device)


def fgw_loss(g_bar, id_reference, ood_reference, alpha=0.5, epsilon=1.0, max_iter=20):
    """mean FGW(g_bar, correctly classified) - mean FGW(g_bar, misclassified)."""
    device = g_bar['X'].device

    def mean_fgw(refs):
        if len(refs) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack([fgw_distance(ref, g_bar, alpha=alpha, epsilon=epsilon, max_iter=max_iter) for ref in refs]).mean()
    return mean_fgw(id_reference) - mean_fgw(ood_reference)


def partial_denoise(diffusion_model, g_mis, noise_t=0.3, num_steps=5):
    X0, A0 = g_mis['X'], g_mis['A']
    t0 = torch.tensor(noise_t, device=X0.device)
    X_t, _, _ = forward_noise(X0, t0, diffusion_model.sde)
    A_t, _, _ = forward_noise(A0, t0, diffusion_model.sde)
    A_t = (A_t + A_t.T) / 2
    ts = torch.linspace(noise_t, 0.001, num_steps, device=X0.device)
    dt = -(noise_t - 0.001) / num_steps
    for t in ts:
        score_x = diffusion_model.score_x(X_t, A_t, t)
        score_a = diffusion_model.score_a(X_t, A_t, t)
        X_t = reverse_step(X_t, t, dt, score_x, diffusion_model.sde, corrector_steps=0)
        A_t = reverse_step(A_t, t, dt, score_a, diffusion_model.sde, corrector_steps=0)
        A_t = (A_t + A_t.T) / 2
    return {'A': (torch.sigmoid(A_t) > 0.5).float(), 'X': X_t}


def fgw_finetune_loss(inv_dm, id_reference, ood_reference, alpha=0.5, epsilon=1.0, noise_t=0.3, denoise_steps=5):
    if len(ood_reference) == 0:
        return torch.tensor(0.0, device=inv_dm.score_x.in_proj.weight.device)
    g_bar = partial_denoise(inv_dm, ood_reference[0], noise_t=noise_t, num_steps=denoise_steps)
    return fgw_loss(g_bar, id_reference, ood_reference, alpha=alpha, epsilon=epsilon)


def fgw_guidance_grad(g_bar, id_reference, ood_reference, alpha=0.5, epsilon=1.0, strength=1.0):
    """Feature-space guidance: -strength * d fgw_loss / dX."""
    if len(id_reference) == 0 and len(ood_reference) == 0:
        return torch.zeros_like(g_bar['X'])
    X = g_bar['X'].detach().requires_grad_(True)
    loss = fgw_loss({'A': g_bar['A'], 'X': X}, id_reference, ood_reference, alpha=alpha, epsilon=epsilon)
    (grad_x,) = torch.autograd.grad(loss, [X])
    return -strength * grad_x.detach()
