"""Encoders and link predictors; `build` creates them with the exact construction / initialisation order of the paper's
runs (construction initialises on the CPU generator, `.to(device)`, then `reset_parameters()` re-initialises on the
device generator; NCN and Refined-GAE are reset twice, as their trainer did)."""
import torch

from .gcn import unGCN
from .predictor import mlp_score


def build(backbone, input_channel, hp, device, num_nodes=None, ood_method=None):
    """(encoder, predictor) for backbone 'GCN' | 'NCN' | 'RGAE'. hp: dict with hidden_channels, num_layers,
    num_layers_predictor, dropout, edge_drop (GCN) and the backbone's own keys (ncn_*, rgae_*; defaults as the trainer)."""
    h = hp['hidden_channels']
    if backbone == 'NCN':
        from .ncn import CNLinkPredictor, NCN_GCN
        model = NCN_GCN(input_channel, h, h, hp['num_layers'], hp['dropout'], ln=hp.get('ncn_ln', False),
                        res=hp.get('ncn_res', False), max_x=-1, conv_fn=hp.get('ncn_conv', 'gcn'), jk=hp.get('ncn_jk', False),
                        edrop=hp.get('ncn_gnnedp', 0.0), xdropout=hp.get('ncn_xdp', 0.0), taildropout=hp.get('ncn_tdp', 0.0),
                        noinputlin=False).to(device)
        # returns LOGITS; every metric is rank based, so no sigmoid (saturated probabilities would tie)
        predictor = CNLinkPredictor(h, h, 1, hp['num_layers_predictor'], hp.get('ncn_predp', 0.0), hp.get('ncn_preedp', 0.0),
                                    hp.get('ncn_lnnn', False), use_xlin=hp.get('ncn_use_xlin', False),
                                    tailact=hp.get('ncn_tailact', False), twolayerlin=hp.get('ncn_twolayerlin', False),
                                    beta=hp.get('ncn_beta', 1.0)).to(device)
        model.reset_parameters(); predictor.reset_parameters()
    elif backbone == 'RGAE':
        from .rgae import RGAEEmb, RGAEEncoder, RGAEScore
        emb = hp.get('rgae_emb_hidden', 0)
        enc = RGAEEncoder(input_channel + emb, h, norm=hp.get('rgae_norm', False), prop_step=hp.get('rgae_prop_step', 4),
                          dropout=hp['dropout'], residual=hp.get('rgae_residual', 0.0), relu=hp.get('rgae_relu', False))
        model = (RGAEEmb(enc, num_nodes, emb) if emb > 0 else enc).to(device)
        predictor = RGAEScore(h, hp['dropout'], hp['num_layers_predictor'], res=hp.get('rgae_res', False),
                              norm=hp.get('rgae_norm', False), scale=hp.get('rgae_scale', False)).to(device)
        model.reset_parameters(); predictor.reset_parameters()
    elif backbone == 'GCN':
        model = unGCN(input_channel, h, h, hp['num_layers'], hp['dropout'], hp.get('edge_drop', 0.0), ood_method).to(device)
        predictor = mlp_score(h, h, 1, hp['num_layers_predictor'], hp['dropout']).to(device)
    else:
        raise ValueError(backbone)
    model.reset_parameters()
    predictor.reset_parameters()
    return model, predictor


def optimizer_for(backbone, model, predictor, hp):
    """Adam over encoder + predictor (NCN: the authors' separate encoder / predictor learning rates)."""
    lr, l2 = hp['lr'], hp.get('l2', 0.0)
    if backbone == 'NCN':
        g, p = hp.get('ncn_gnnlr', -1.0), hp.get('ncn_prelr', -1.0)
        return torch.optim.Adam([{'params': model.parameters(), 'lr': g if g > 0 else lr},
                                 {'params': predictor.parameters(), 'lr': p if p > 0 else lr}], lr=lr, weight_decay=l2)
    return torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=lr, weight_decay=l2)
