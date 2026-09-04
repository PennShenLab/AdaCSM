"""Torch model definitons for the AdaCSM model.

This includes definitons for the AdaCSM module.
The main interface is the AdaCSMSurvivalMachinesTorch class which inherits
from torch.nn.Module.
"""

import torch.nn as nn
import torch


class MixtureOfExpertsLayer(nn.Module):
    """Mixture of Experts layer for AdaCSM.
    
    Parameters
    ----------
    inputdim : int
        Dimensionality of the input features.
    hidden : int
        Number of neurons in each expert's hidden layer.
    num_experts : int
        Number of expert networks.
    top_k : int or None
        If None, use all experts. If int, use only top-k experts by gating weight.
    dropout : float, optional
        Dropout rate for expert networks (default: 0.0, disabled).
    gate_dropout : float, optional
        Dropout rate for gating network (default: 0.0, disabled).
    temperature : float, optional
        Temperature for gating softmax (default: 1.0, no temperature scaling).
    routing_noise_std : float, optional
        Std of noise added to routing during training (default: 0.0, disabled).
    """
    
    def __init__(self, inputdim, hidden, num_experts=4, top_k=None,
                 dropout=0.0, gate_dropout=0.0, temperature=1.0, routing_noise_std=0.0):
        super(MixtureOfExpertsLayer, self).__init__()
        self.num_experts = num_experts
        self.hidden = hidden
        self.top_k = top_k if top_k is not None else num_experts  # Default: use all experts
        self.dropout = dropout
        self.gate_dropout = gate_dropout
        self.temperature = temperature
        self.routing_noise_std = routing_noise_std
        
        # Ensure top_k doesn't exceed num_experts
        if self.top_k > num_experts:
            self.top_k = num_experts
        
        # Create expert networks (with optional dropout)
        if dropout > 0:
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(inputdim, hidden, bias=False),
                    nn.ReLU6(),
                    nn.Dropout(dropout)
                ) for _ in range(num_experts)
            ])
        else:
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(inputdim, hidden, bias=False),
                    nn.ReLU6()
                ) for _ in range(num_experts)
            ])
        
        # Gating network (with optional dropout)
        if gate_dropout > 0:
            self.gate = nn.Sequential(
                nn.Dropout(gate_dropout),
                nn.Linear(inputdim, num_experts)
            )
        else:
            # Use simple Sequential with Softmax for default behavior (temperature=1.0)
            if temperature == 1.0 and routing_noise_std == 0.0:
                self.gate = nn.Sequential(
                    nn.Linear(inputdim, num_experts),
                    nn.Softmax(dim=-1)
                )
            else:
                # Use separate Linear when temperature or noise is enabled
                self.gate = nn.Linear(inputdim, num_experts)
    
    def forward(self, x):
        # Compute gate weights
        if isinstance(self.gate, nn.Sequential):
            # Simple case: gate includes Softmax (temperature=1.0, no noise)
            gate_weights = self.gate(x)  # [batch, num_experts]
        else:
            # Advanced case: apply temperature scaling and/or routing noise
            gate_logits = self.gate(x)  # [batch, num_experts]
            
            # Add routing noise during training (if enabled)
            if self.training and self.routing_noise_std > 0:
                noise = torch.randn_like(gate_logits) * self.routing_noise_std
                gate_logits = gate_logits + noise
            
            # Apply temperature scaling and softmax
            gate_weights = torch.softmax(gate_logits / self.temperature, dim=-1)
        
        # Apply top-k masking if needed
        if self.top_k < self.num_experts:
            # Get top-k indices and values
            topk_values, topk_indices = torch.topk(gate_weights, k=self.top_k, dim=-1)  # [batch, top_k]
            
            # Create mask and renormalize
            mask = torch.zeros_like(gate_weights)
            mask.scatter_(1, topk_indices, 1.0)
            gate_weights = gate_weights * mask
            gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-8)  # Renormalize
        
        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch, num_experts, hidden]
        
        # Weighted combination
        output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)  # [batch, hidden]
        
        return output


def create_representation(inputdim, layers, activation, use_moe=False, num_experts=4, top_k=None,
                         moe_dropout=0.0, gate_dropout=0.0, gate_temperature=1.0, routing_noise_std=0.0):
    r"""Helper function to generate the representation function for AdaCSM.

  AdaCSM learns a representation (\ Phi(X) \) for the input
  data. This representation is parameterized using a Non Linear Multilayer
  Perceptron (`torch.nn.Module`) or a Mixture of Experts. This is a helper function designed to
  instantiate the representation for AdaCSM.

  .. warning::
    Not designed to be used directly.

  Parameters
  ----------
  inputdim: int
      Dimensionality of the input features.
  layers: list
      A list consisting of the number of neurons in each hidden layer.
  activation: str
      Choice of activation function: One of 'ReLU6', 'ReLU' or 'SeLU'.
  use_moe: bool
      Whether to use Mixture of Experts instead of standard MLP.
  num_experts: int
      Number of experts to use if use_moe=True.
  top_k: int or None
      If not None and use_moe=True, only use top-k experts by gating weight.

  Returns
  ----------
  an MLP or MoE with torch.nn.Module with the specfied structure.

  """

    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()

    modules = []
    prevdim = inputdim

    for idx, hidden in enumerate(layers):
        if use_moe and idx == 0:  # Use MoE for first layer only
            modules.append(MixtureOfExpertsLayer(prevdim, hidden, num_experts, top_k=top_k,
                                                 dropout=moe_dropout, gate_dropout=gate_dropout,
                                                 temperature=gate_temperature,
                                                 routing_noise_std=routing_noise_std))
        else:
            modules.append(nn.Linear(prevdim, hidden, bias=False))
            modules.append(act)
        prevdim = hidden

    return nn.Sequential(*modules)


class AdaCSMSurvivalMachinesTorch(nn.Module):
    """A Torch implementation of AdaCSM model."""

    def _init_dcsm_layers(self, lastdim):
        if self.is_seed:
            print('random seed for torch model initialization is: ', self.random_state)
            torch.manual_seed(self.random_state)
        if self.dist in ['Weibull']:
            self.act = nn.SELU()
            if self.fix:
                self.shape = nn.ParameterDict({str(r + 1): nn.Parameter(torch.randn(self.k, requires_grad=True))
                                               for r in range(self.risks)})
                self.scale = nn.ParameterDict({str(r + 1): nn.Parameter(torch.randn(self.k, requires_grad=True))
                                               for r in range(self.risks)})
            else:
                self.shape = nn.ParameterDict({str(r + 1): nn.Parameter(-torch.ones(self.k))
                                               for r in range(self.risks)})
                self.scale = nn.ParameterDict({str(r + 1): nn.Parameter(-torch.ones(self.k))
                                               for r in range(self.risks)})
        else:
            raise NotImplementedError('Distribution: ' + self.dist + ' not implemented' +
                                      ' yet.')

        self.gate = nn.ModuleDict({str(r + 1): nn.Sequential(
            nn.Linear(lastdim, self.k, bias=False)
        ) for r in range(self.risks)})

        if self.fix == False:
            self.scaleg = nn.ModuleDict({str(r + 1): nn.Sequential(
                nn.Linear(lastdim, self.k, bias=True)
            ) for r in range(self.risks)})

            self.shapeg = nn.ModuleDict({str(r + 1): nn.Sequential(
                nn.Linear(lastdim, self.k, bias=True)
            ) for r in range(self.risks)})

    def __init__(self, inputdim, k, layers=None, dist='Weibull',
                 temp=1000., discount=1.0, optimizer='Adam',
                 risks=1, random_state=42, fix=False, is_seed=False,
                 use_moe=False, num_experts=4, top_k=None,
                 moe_dropout=0.0, gate_dropout=0.0, gate_temperature=1.0, routing_noise_std=0.0):
        super(AdaCSMSurvivalMachinesTorch, self).__init__()

        self.k = k
        self.dist = dist
        self.temp = float(temp)
        self.discount = float(discount)
        self.optimizer = optimizer
        self.risks = risks
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_dropout = moe_dropout
        self.gate_dropout = gate_dropout
        self.gate_temperature = gate_temperature
        self.routing_noise_std = routing_noise_std

        if layers is None:
            layers = []
        self.layers = layers

        if len(layers) == 0:
            lastdim = inputdim
        else:
            lastdim = layers[-1]

        self.random_state = random_state
        self.fix = fix
        self.is_seed = is_seed

        self._init_dcsm_layers(lastdim)
        self.embedding = create_representation(inputdim, layers, 'ReLU6',
                                              use_moe=use_moe, num_experts=num_experts, top_k=top_k,
                                              moe_dropout=moe_dropout, gate_dropout=gate_dropout,
                                              gate_temperature=gate_temperature,
                                              routing_noise_std=routing_noise_std)

    def forward(self, x, risk='1'):
        xrep = self.embedding(x)
        dim = x.shape[0]

        if self.fix:
            return (self.shape[risk].expand(dim, -1),
                    self.scale[risk].expand(dim, -1),
                    self.gate[risk](xrep) / self.temp)
        else:
            return (self.act(self.shapeg[risk](xrep)) + self.shape[risk].expand(dim, -1),
                    self.act(self.scaleg[risk](xrep)) + self.scale[risk].expand(dim, -1),
                    self.gate[risk](xrep) / self.temp)

    def get_shape_scale(self, risk='1'):
        return self.shape[risk], self.scale[risk]


AdaCSMMixtureOfExpertsLayer = MixtureOfExpertsLayer
