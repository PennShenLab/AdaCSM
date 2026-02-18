"""Torch model definitons for the Deep Clustering Survival Machines model

This includes definitons for the Deep Clustering Survival Machines module.
The main interface is the DeepClusteringSurvivalMachines class which inherits
from torch.nn.Module.

"""

import torch.nn as nn
import torch
import numpy as np


class MixtureOfExpertsLayer(nn.Module):
    """Mixture of Experts Layer for representation learning.
    
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
    """
    
    def __init__(self, inputdim, hidden, num_experts=4, top_k=None):
        super(MixtureOfExpertsLayer, self).__init__()
        self.num_experts = num_experts
        self.hidden = hidden
        self.top_k = top_k if top_k is not None else num_experts  # Default: use all experts
        
        # Ensure top_k doesn't exceed num_experts
        if self.top_k > num_experts:
            self.top_k = num_experts
        
        # Create expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inputdim, hidden, bias=False),
                nn.ReLU6()
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(inputdim, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        # Compute gate weights
        gate_weights = self.gate(x)  # [batch, num_experts]
        
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


def create_representation(inputdim, layers, activation, use_moe=False, num_experts=4, top_k=None):
    r"""Helper function to generate the representation function for DCSM.

  Deep Clustering Survival Machines learns a representation (\ Phi(X) \) for the input
  data. This representation is parameterized using a Non Linear Multilayer
  Perceptron (`torch.nn.Module`) or a Mixture of Experts. This is a helper function designed to
  instantiate the representation for Deep Clustering Survival Machines.

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
            modules.append(MixtureOfExpertsLayer(prevdim, hidden, num_experts, top_k=top_k))
        else:
            modules.append(nn.Linear(prevdim, hidden, bias=False))
            modules.append(act)
        prevdim = hidden

    return nn.Sequential(*modules)


class DeepClusteringSurvivalMachinesTorch(nn.Module):
    """A Torch implementation of Deep Clustering Survival Machines model.

  This is an implementation of Deep Clustering Survival Machines model in torch.
  It inherits from the torch.nn.Module class and includes references to the
  representation learning MLP, the parameters of the underlying distributions
  and the forward function which is called whenver data is passed to the
  module. Each of the parameters belongs to nn.Parameters and torch automatically
  keeps track and computes gradients for them.

  Parameters
  ----------
  inputdim: int
      Dimensionality of the input features.
  k: int
      The number of underlying parametric distributions.
  layers: list
      A list of integers consisting of the number of neurons in each
      hidden layer.
  init: tuple
      A tuple for initialization of the parameters for the underlying
      distributions. (shape, scale).
  activation: str
      Choice of activation function for the MLP representation.
      One of 'ReLU6', 'ReLU' or 'SeLU'.
      Default is 'ReLU6'.
  dist: str
      Choice of the underlying survival distributions.
      One of 'Weibull', 'LogNormal'.
      Default is 'Weibull'.
  temp: float
      The logits for the gate are rescaled with this value.
      Default is 1000.
  discount: float
      a float in [0,1] that determines how to discount the tail bias
      from the uncensored instances.
      Default is 1.

  """

    def _init_dcsm_layers(self, lastdim):

        if self.is_seed:  # if is_seed is true, means we use the random seed to fix the initialization
            print('random seed for torch model initialization is: ', self.random_state)
            torch.manual_seed(self.random_state)  # fix the initialization
        if self.dist in ['Weibull']:
            self.act = nn.SELU()
            if self.fix:  # means using fixed base distribution
                self.shape = nn.ParameterDict({str(r + 1): nn.Parameter(torch.randn(self.k, requires_grad=True))
                                               for r in range(self.risks)})  # .cuda()
                self.scale = nn.ParameterDict({str(r + 1): nn.Parameter(torch.randn(self.k, requires_grad=True))
                                               for r in range(self.risks)})  # .cuda()
            else:
                self.shape = nn.ParameterDict({str(r + 1): nn.Parameter(-torch.ones(self.k))
                                               for r in range(self.risks)})  # .cuda()
                self.scale = nn.ParameterDict({str(r + 1): nn.Parameter(-torch.ones(self.k))
                                               for r in range(self.risks)})  # .cuda()
        else:
            raise NotImplementedError('Distribution: ' + self.dist + ' not implemented' +
                                      ' yet.')

        self.gate = nn.ModuleDict({str(r + 1): nn.Sequential(
            nn.Linear(lastdim, self.k, bias=False)
        ) for r in range(self.risks)})  # .cuda()

        if self.fix == False:  # means using varied base distribution by discarding these parameters
            self.scaleg = nn.ModuleDict({str(r + 1): nn.Sequential(
                nn.Linear(lastdim, self.k, bias=True)
            ) for r in range(self.risks)})  # .cuda()

            self.shapeg = nn.ModuleDict({str(r + 1): nn.Sequential(
                nn.Linear(lastdim, self.k, bias=True)
            ) for r in range(self.risks)})  # .cuda()

    def __init__(self, inputdim, k, layers=None, dist='Weibull',
                 temp=1000., discount=1.0, optimizer='Adam',
                 risks=1, random_state=42, fix=False, is_seed=False,
                 use_moe=False, num_experts=4, top_k=None):
        super(DeepClusteringSurvivalMachinesTorch, self).__init__()

        self.k = k
        self.dist = dist
        self.temp = float(temp)
        self.discount = float(discount)
        self.optimizer = optimizer
        self.risks = risks
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k

        if layers is None: layers = []
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
                                              use_moe=use_moe, num_experts=num_experts, top_k=top_k)

    def forward(self, x, risk='1'):
        """The forward function that is called when data is passed through DCSM.

    Args:
      x:
        a torch.tensor of the input features.

    """
        xrep = self.embedding(x)
        dim = x.shape[0]

        if self.fix:  # means using fixed base distributions
            return (self.shape[risk].expand(dim, -1).cuda(),
                    self.scale[risk].expand(dim, -1).cuda(),
                    self.gate[risk](xrep) / self.temp)
        else:
            return (self.act(self.shapeg[risk](xrep)) + self.shape[risk].expand(dim, -1),
                    self.act(self.scaleg[risk](xrep)) + self.scale[risk].expand(dim, -1),
                    self.gate[risk](xrep) / self.temp)

    def get_shape_scale(self, risk='1'):
        return self.shape[risk], self.scale[risk]


def create_conv_representation(inputdim, hidden,
                               typ='ConvNet', add_linear=True):
    r"""Helper function to generate the representation function for DCSM.

  Deep Clustering Survival Machines learns a representation (\ Phi(X) \) for the input
  data. This representation is parameterized using a Convolutional Neural
  Network (`torch.nn.Module`). This is a helper function designed to
  instantiate the representation for Deep Clustering Survival Machines.

  .. warning::
    Not designed to be used directly.

  Parameters
  ----------
  inputdim: tuple
      Dimensionality of the input image.
  hidden: int
      The number of neurons in each hidden layer.
  typ: str
      Choice of convolutional neural network: One of 'ConvNet'

  Returns
  ----------
  an ConvNet with torch.nn.Module with the specfied structure.

  """

    if typ == 'ConvNet':
        embedding = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU6(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 3),
            nn.ReLU6(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.ReLU6(),
        )

    if add_linear:
        dummyx = torch.ones((10, 1) + inputdim)
        dummyout = embedding.forward(dummyx)
        outshape = dummyout.shape

        embedding.add_module('linear', torch.nn.Linear(outshape[-1], hidden))
        embedding.add_module('act', torch.nn.ReLU6())

    return embedding