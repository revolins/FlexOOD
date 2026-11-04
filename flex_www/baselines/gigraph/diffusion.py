"""Score-based graph diffusion for GI-Graph's Env-DM / Inv-DM (VP-SDE, GDSS-style dense GCN score networks) and the
attribute-prediction network that guides Inv-DM sampling (paper eq. 16-20)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------------------------ VP-SDE
class VPSDE:
    """Variance-preserving SDE, beta(t) linear in [beta_min, beta_max]."""

    def __init__(self, beta_min=0.1, beta_max=20.0, num_steps=1000):
        self.beta_min, self.beta_max, self.num_steps = beta_min, beta_max, num_steps

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _log_mean_coeff(self, t):
        return -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

    def marginal_prob(self, g0, t):
        log_mean_coeff = self._log_mean_coeff(t)
        while log_mean_coeff.dim() < g0.dim():
            log_mean_coeff = log_mean_coeff.unsqueeze(-1)
        mean = torch.exp(log_mean_coeff) * g0
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff)).clamp(min=1e-05)
        return mean, std


def forward_noise(g0, t, sde, ood_alpha=None):
    """(g_t, noise, std); ood_alpha amplifies the noise (Env-DM's OOD control, eq. 16)."""
    mean, std = sde.marginal_prob(g0, t)
    if ood_alpha is not None:
        std = std * (1.0 + ood_alpha)
    noise = torch.randn_like(g0)
    return mean + std * noise, noise, std


def reverse_step(g_t, t, dt, score, sde, guidance_grad=None, corrector_steps=1, snr=0.16, ood_alpha=None):
    """One Euler-Maruyama predictor step of the reverse SDE plus Langevin corrector steps."""
    beta_t = sde.beta(t)
    total_score = score if guidance_grad is None else score + guidance_grad
    diffusion_coef = math.sqrt(beta_t)
    if ood_alpha is not None:
        diffusion_coef = diffusion_coef * (1.0 + ood_alpha)
    drift = -0.5 * beta_t * g_t - diffusion_coef ** 2 * total_score
    noise = torch.randn_like(g_t)
    g_next = g_t - drift * dt + diffusion_coef * math.sqrt(abs(dt)) * noise
    for _ in range(corrector_steps):
        grad = score if guidance_grad is None else score + guidance_grad
        noise = torch.randn_like(g_next)
        grad_norm = grad.flatten(1).norm(dim=-1).mean()
        noise_norm = noise.flatten(1).norm(dim=-1).mean()
        step_size = (snr * noise_norm / (grad_norm + 1e-08)) ** 2 * 2
        g_next = g_next + step_size * grad + torch.sqrt(2 * step_size) * noise
    return g_next


# ------------------------------------------------------------------------------------------------ score networks
def timestep_embedding(t, dim):
    t = torch.as_tensor(t)
    if t.dim() == 0:
        t = t.view(1)
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (math.log(10000.0) / max(half - 1, 1)))
    args = t.float().view(-1, 1) * freqs.view(1, -1) * 1000.0
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def dense_gcn_propagate(A, h):
    return A @ h / A.sum(-1, keepdim=True).clamp(min=1.0)


class ScoreNetworkX(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64, num_layers=3, time_dim=32):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.in_proj = nn.Linear(feat_dim, hidden_dim)
        self.convs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_dim, feat_dim)

    def forward(self, X, A, t):
        h = self.in_proj(X)
        temb = self.time_mlp(timestep_embedding(t, self.time_dim))
        if h.dim() == 3 and temb.dim() == 2:
            temb = temb.unsqueeze(1)
        for conv in self.convs:
            h = F.silu(conv(dense_gcn_propagate(A, h)) + temb)
        return self.out_proj(h)


class ScoreNetworkA(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64, num_layers=3, time_dim=32):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.in_proj = nn.Linear(feat_dim, hidden_dim)
        self.convs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.edge_out = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, X, A, t):
        h = self.in_proj(X)
        temb = self.time_mlp(timestep_embedding(t, self.time_dim))
        if h.dim() == 3 and temb.dim() == 2:
            temb = temb.unsqueeze(1)
        for conv in self.convs:
            h = F.silu(conv(dense_gcn_propagate(A, h)) + temb)
        n = h.size(-2)
        h_i = h.unsqueeze(-2).expand(*h.shape[:-1], n, h.size(-1))
        h_j = h.unsqueeze(-3).expand(*h.shape[:-2], n, n, h.size(-1))
        score_A = self.edge_out(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)
        return (score_A + score_A.transpose(-1, -2)) / 2


class DiffusionModel(nn.Module):
    """Joint (X, A) score model on a VP-SDE."""

    def __init__(self, feat_dim, hidden_dim=64, num_layers=3, time_dim=32, ood_alpha=None):
        super().__init__()
        self.score_x = ScoreNetworkX(feat_dim, hidden_dim, num_layers, time_dim)
        self.score_a = ScoreNetworkA(feat_dim, hidden_dim, num_layers, time_dim)
        self.sde = VPSDE()
        self.ood_alpha = ood_alpha

    def score_matching_loss(self, X0, A0):
        t = torch.rand(1, device=X0.device).clamp(min=0.001)
        Xt, noise_x, std_x = forward_noise(X0, t, self.sde, self.ood_alpha)
        At, noise_a, std_a = forward_noise(A0, t, self.sde, self.ood_alpha)
        At = (At + At.transpose(-1, -2)) / 2
        loss_x = F.mse_loss(self.score_x(Xt, At, t), -noise_x / std_x)
        loss_a = F.mse_loss(self.score_a(Xt, At, t), -noise_a / std_a)
        return loss_x + loss_a

    @torch.no_grad()
    def sample(self, X_shape, A_shape, num_steps=50, guidance_fn=None, device='cpu'):
        X_t = torch.randn(X_shape, device=device)
        A_t = torch.randn(A_shape, device=device)
        A_t = (A_t + A_t.transpose(-1, -2)) / 2
        ts = torch.linspace(1.0, 0.001, num_steps, device=device)
        dt = -(1.0 - 0.001) / num_steps
        for t in ts:
            with torch.enable_grad():
                score_x = self.score_x(X_t, A_t, t)
                score_a = self.score_a(X_t, A_t, t)
            guidance_x = guidance_a = None
            if guidance_fn is not None:
                guidance_x, guidance_a = guidance_fn(X_t, A_t, t)
            X_t = reverse_step(X_t, t, dt, score_x, self.sde, guidance_grad=guidance_x, ood_alpha=self.ood_alpha)
            A_t = reverse_step(A_t, t, dt, score_a, self.sde, guidance_grad=guidance_a, ood_alpha=self.ood_alpha)
            A_t = (A_t + A_t.transpose(-1, -2)) / 2
            X_t = torch.nan_to_num(X_t, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            A_t = torch.nan_to_num(A_t, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        A_bin = (torch.sigmoid(A_t) > 0.5).float()
        if A_bin.dim() == 2:
            A_bin.fill_diagonal_(0)
        return X_t, A_bin


# ------------------------------------------------------------------------------------------------ attribute network
class AttributePredictionNetwork(nn.Module):
    """Noisy-graph classifier (eq. 19-20): dense GCN stack, gated readout, P(positive edge)."""

    def __init__(self, feat_dim, hidden_dim=64, num_gcn_layers=3):
        super().__init__()
        self.gcns = nn.ModuleList([nn.Linear(feat_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_gcn_layers)])
        cat_dim = feat_dim + hidden_dim * num_gcn_layers
        self.mlp_s_gate = nn.Sequential(nn.Linear(cat_dim, hidden_dim), nn.Sigmoid())
        self.mlp_t_gate = nn.Sequential(nn.Linear(cat_dim, hidden_dim), nn.Tanh())
        self.mlp_s_out = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, X_t, A_t):
        H = X_t
        Hs = [H]
        for gcn in self.gcns:
            H = torch.tanh(gcn(dense_gcn_propagate(A_t, H)))
            Hs.append(H)
        H_cat = torch.cat(Hs, dim=-1)
        pooled = torch.tanh(self.mlp_s_gate(H_cat) * self.mlp_t_gate(H_cat)).sum(dim=-2)
        return self.mlp_s_out(pooled).squeeze(-1)

    def guidance_grad(self, X_t, A_t, target_is_positive=True, strength=1.0):
        """strength * d log p(target | X_t, A_t) / d(X_t, A_t)."""
        X_t = torch.nan_to_num(X_t.detach(), nan=0.0, posinf=10000.0, neginf=-10000.0).requires_grad_(True)
        A_t = torch.nan_to_num(A_t.detach(), nan=0.0, posinf=10000.0, neginf=-10000.0).requires_grad_(True)
        p = torch.nan_to_num(self.forward(X_t, A_t), nan=0.5).clamp(1e-06, 1 - 1e-06)
        target = torch.ones_like(p) if target_is_positive else torch.zeros_like(p)
        log_prob = -F.binary_cross_entropy(p, target, reduction='sum')
        grad_x, grad_a = torch.autograd.grad(log_prob, [X_t, A_t])
        return strength * grad_x.detach(), strength * grad_a.detach()
