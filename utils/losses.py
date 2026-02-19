"""Loss function definitions for the Deep Clustering Survival Machines model

In this module we define the various losses for the censored and uncensored
instances of data corresponding to Weibull distribution.
These losses are optimized when training DCSM.

"""

import numpy as np
import torch
import torch.nn as nn


def _weibull_loss(model, t, e, risk='1'):
    shape, scale = model.get_shape_scale(risk)

    k_ = shape.expand(t.shape[0], -1)
    b_ = scale.expand(t.shape[0], -1)

    ll = 0.
    for g in range(model.k):
        k = k_[:, g]
        b = b_[:, g]

        s = - (torch.pow(torch.exp(b) * t, torch.exp(k)))
        f = k + b + ((torch.exp(k) - 1) * (b + torch.log(t)))
        f = f + s

        uncens = np.where(e.cpu().data.numpy() == int(risk))[0]
        cens = np.where(e.cpu().data.numpy() != int(risk))[0]
        ll += f[uncens].sum() + s[cens].sum()

    return -ll.mean()


def unconditional_loss(model, t, e, risk='1'):
    if model.dist == 'Weibull':
        return _weibull_loss(model, t, e, risk)
    else:
        raise NotImplementedError('Distribution: ' + model.dist +
                                  ' not implemented yet.')


def _conditional_weibull_loss(model, x, t, e, elbo=True, risk='1', load_balance_lambda=0.0):
    alpha = model.discount
    shape, scale, logits = model.forward(x, risk)

    k_ = shape
    b_ = scale

    lossf = []
    losss = []

    for g in range(model.k):
        k = k_[:, g]
        b = b_[:, g]

        s = - (torch.pow(torch.exp(b) * t, torch.exp(k)))
        f = k + b + ((torch.exp(k) - 1) * (b + torch.log(t)))
        f = f + s

        lossf.append(f)
        losss.append(s)

    losss = torch.stack(losss, dim=1)
    lossf = torch.stack(lossf, dim=1)

    if elbo:

        lossg = nn.Softmax(dim=1)(logits)
        losss = lossg * losss
        lossf = lossg * lossf
        losss = losss.sum(dim=1)
        lossf = lossf.sum(dim=1)

    else:

        lossg = nn.LogSoftmax(dim=1)(logits)
        losss = lossg + losss
        lossf = lossg + lossf
        losss = torch.logsumexp(losss, dim=1)
        lossf = torch.logsumexp(lossf, dim=1)

    uncens = np.where(e.cpu().data.numpy() == int(risk))[0]
    cens = np.where(e.cpu().data.numpy() != int(risk))[0]
    ll = lossf[uncens].sum() + alpha * losss[cens].sum()
    
    main_loss = -ll / float(len(uncens) + len(cens))
    
    # Add load balancing loss if using MoE and lambda > 0
    if load_balance_lambda > 0 and hasattr(model, 'use_moe') and model.use_moe:
        # Compute load balancing loss based on gate weights
        # This encourages even distribution of samples across experts
        gate_softmax = nn.Softmax(dim=1)(logits)  # [batch, k] - gate weights
        
        # Mean gate activation per expert
        expert_loads = gate_softmax.mean(dim=0)  # [k]
        
        # Load balancing loss: penalize deviation from uniform distribution
        # Target is uniform (1/num_experts)
        target_load = 1.0 / model.k
        lb_loss = torch.mean((expert_loads - target_load)**2)
        
        main_loss = main_loss + load_balance_lambda * lb_loss
    
    return main_loss


def conditional_loss(model, x, t, e, elbo=True, risk='1', load_balance_lambda=0.0):
    if model.dist == 'Weibull':
        return _conditional_weibull_loss(model, x, t, e, elbo, risk, load_balance_lambda)
    else:
        raise NotImplementedError('Distribution: ' + model.dist +
                                  ' not implemented yet.')


def _weibull_pdf(model, x, t_horizon, risk='1'):
    squish = nn.LogSoftmax(dim=1)

    shape, scale, logits = model.forward(x, risk)
    logits = squish(logits)

    k_ = shape
    b_ = scale

    t_horz = torch.tensor(t_horizon).double()
    t_horz = t_horz.repeat(shape.shape[0], 1)

    pdfs = []
    for j in range(len(t_horizon)):

        t = t_horz[:, j]
        lpdfs = []

        for g in range(model.k):
            k = k_[:, g]
            b = b_[:, g]
            s = - (torch.pow(torch.exp(b) * t, torch.exp(k)))
            f = k + b + ((torch.exp(k) - 1) * (b + torch.log(t)))
            f = f + s
            lpdfs.append(f)

        lpdfs = torch.stack(lpdfs, dim=1)
        lpdfs = lpdfs + logits
        lpdfs = torch.logsumexp(lpdfs, dim=1)
        pdfs.append(lpdfs.detach().numpy())

    return pdfs


def _weibull_cdf(model, x, t_horizon, risk='1'):
    squish = nn.LogSoftmax(dim=1)

    shape, scale, logits = model.forward(x, risk)

    logits = squish(logits)

    k_ = shape
    b_ = scale

    t_horz = torch.tensor(t_horizon).double()
    t_horz = t_horz.repeat(shape.shape[0], 1).cuda()

    cdfs = []
    for j in range(len(t_horizon)):

        t = t_horz[:, j]
        lcdfs = []

        for g in range(model.k):
            k = k_[:, g]
            b = b_[:, g]
            s = - (torch.pow(torch.exp(b) * t, torch.exp(k)))
            lcdfs.append(s)

        lcdfs = torch.stack(lcdfs, dim=1)
        lcdfs = lcdfs + logits
        lcdfs = torch.logsumexp(lcdfs, dim=1)
        # cdfs.append(lcdfs.detach().numpy())
        cdfs.append(lcdfs.detach().cpu().numpy())

    return cdfs


def _weibull_mean(model, x, risk='1'):
    squish = nn.LogSoftmax(dim=1)

    shape, scale, logits = model.forward(x, risk)
    logits = squish(logits)

    k_ = shape
    b_ = scale

    lmeans = []

    for g in range(model.k):
        k = k_[:, g]
        b = b_[:, g]

        one_over_k = torch.reciprocal(torch.exp(k))
        lmean = -(one_over_k * b) + torch.lgamma(1 + one_over_k)
        lmeans.append(lmean)

    lmeans = torch.stack(lmeans, dim=1)
    lmeans = lmeans + logits
    lmeans = torch.logsumexp(lmeans, dim=1)

    return torch.exp(lmeans).detach().cpu().numpy()


def predict_mean(model, x, risk='1'):
    torch.no_grad()
    if model.dist == 'Weibull':
        return _weibull_mean(model, x, risk)
    else:
        raise NotImplementedError('Mean of Distribution: ' + model.dist +
                                  ' not implemented yet.')


def predict_pdf(model, x, t_horizon, risk='1'):
    torch.no_grad()
    if model.dist == 'Weibull':
        return _weibull_pdf(model, x, t_horizon, risk)
    else:
        raise NotImplementedError('Distribution: ' + model.dist +
                                  ' not implemented yet.')


def predict_cdf(model, x, t_horizon, risk='1'):
    torch.no_grad()
    if model.dist == 'Weibull':
        return _weibull_cdf(model, x, t_horizon, risk)
    else:
        raise NotImplementedError('Distribution: ' + model.dist +
                                  ' not implemented yet.')


def compute_load_balance_loss(model):
    """Compute load balancing loss to encourage uniform expert usage.
    
    This loss encourages all experts to be used roughly equally across the batch,
    preventing routing collapse where all samples route to a few experts.
    
    Args:
        model: DCSM model
        
    Returns:
        load_balance_loss: scalar tensor representing the load imbalance
    """
    if not hasattr(model, 'use_moe') or not model.use_moe:
        return torch.tensor(0.0).cuda()
    
    # Find the MoE layer in the embedding
    moe_layer = None
    for module in model.embedding.modules():
        if hasattr(module, 'last_gate_weights'):
            moe_layer = module
            break
    
    if moe_layer is None or moe_layer.last_gate_weights is None:
        return torch.tensor(0.0).cuda()
    
    # Get gate weights from last forward pass: [batch_size, num_experts]
    gate_weights = moe_layer.last_gate_weights
    
    # Compute mean usage per expert across batch
    expert_usage = gate_weights.mean(dim=0)  # [num_experts]
    
    # Target is uniform distribution
    uniform_target = 1.0 / moe_layer.num_experts
    
    # Compute variance from uniform target (lower is better)
    load_balance_loss = ((expert_usage - uniform_target) ** 2).sum()
    
    return load_balance_loss
