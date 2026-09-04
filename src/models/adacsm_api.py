"""
AdaCSM API wrapper around torch implementations.
"""

from .adacsm_torch import AdaCSMSurvivalMachinesTorch
from utils import losses as losses
from utils.model_utils import train_dcsm
from utils.model_utils import _reshape_tensor_with_nans
import torch
import numpy as np


class AdaCSMBase:
    """Base class for AdaCSM."""

    def __init__(
        self,
        k=3,
        layers=None,
        distribution="Weibull",
        temp=1000.0,
        discount=1.0,
        random_state=42,
        fix=False,
        is_seed=False,
        use_moe=False,
        num_experts=4,
        top_k=None,
        moe_dropout=0.0,
        gate_dropout=0.0,
        gate_temperature=1.0,
        routing_noise_std=0.0,
        weight_decay=0.0,
        load_balance_lambda=0.0,
        progress_every=0,
    ):
        self.k = k
        self.layers = layers
        self.dist = distribution
        self.temp = temp
        self.discount = discount
        self.fitted = False
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_dropout = moe_dropout
        self.gate_dropout = gate_dropout
        self.gate_temperature = gate_temperature
        self.routing_noise_std = routing_noise_std
        self.weight_decay = weight_decay
        self.load_balance_lambda = load_balance_lambda
        self.progress_every = progress_every
        self.random_state = random_state
        self.fix = fix
        self.is_seed = is_seed
        self.device = self._resolve_device()
        self.tensor_dtype = torch.float32 if self.device.type == "mps" else torch.float64

    @staticmethod
    def _resolve_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _to_device_tensor(self, value):
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(device=self.device, dtype=self.tensor_dtype)
        return value.to(device=self.device, dtype=self.tensor_dtype)

    def _gen_torch_model(self, inputdim, optimizer, risks):
        return AdaCSMSurvivalMachinesTorch(
            inputdim,
            k=self.k,
            layers=self.layers,
            dist=self.dist,
            temp=self.temp,
            discount=self.discount,
            optimizer=optimizer,
            risks=risks,
            random_state=self.random_state,
            fix=self.fix,
            is_seed=self.is_seed,
            use_moe=self.use_moe,
            num_experts=self.num_experts,
            top_k=self.top_k,
            moe_dropout=self.moe_dropout,
            gate_dropout=self.gate_dropout,
            gate_temperature=self.gate_temperature,
            routing_noise_std=self.routing_noise_std,
        )

    def fit(
        self,
        x,
        t,
        e,
        vsize=0.15,
        val_data=None,
        iters=10000,
        learning_rate=1e-3,
        batch_size=100,
        elbo=True,
        optimizer="Adam",
        random_state=100,
        patience=100,
        early_stopping=True,
        weight_decay=0.0,
        load_balance_lambda=0.0,
        progress_every=0,
    ):
        x, x_test = x
        t, t_test = t
        e, e_test = e
        x_test = self._to_device_tensor(x_test)
        t_test = self._to_device_tensor(t_test)
        e_test = self._to_device_tensor(e_test)
        x_train, t_train, e_train, x_val, t_val, e_val = self._preprocess_training_data(
            x, t, e, vsize, val_data, random_state
        )
        inputdim = x_train.shape[-1]
        maxrisk = int(np.nanmax(e_train.cpu().numpy()))
        model = self._gen_torch_model(inputdim, optimizer, risks=maxrisk).to(
            device=self.device, dtype=self.tensor_dtype
        )
        model, _ = train_dcsm(
            model,
            x_train,
            t_train,
            e_train,
            x_test,
            t_test,
            e_test,
            n_iter=iters,
            lr=learning_rate,
            elbo=elbo,
            bs=batch_size,
            patience=patience,
            early_stopping=early_stopping,
            weight_decay=weight_decay,
            load_balance_lambda=load_balance_lambda,
            progress_every=progress_every,
        )
        self.torch_model = model.eval()
        self.fitted = True
        return self

    def compute_nll(self, x, t, e):
        if not self.fitted:
            raise Exception("Model has not been fitted yet.")
        _, _, _, x_val, t_val, e_val = self._preprocess_training_data(x, t, e, 0, None, 0)
        x_val, t_val, e_val = x_val, _reshape_tensor_with_nans(t_val), _reshape_tensor_with_nans(e_val)
        loss = 0
        for r in range(self.torch_model.risks):
            loss += float(
                losses.conditional_loss(self.torch_model, x_val, t_val, e_val, elbo=False, risk=str(r + 1))
                .detach()
                .cpu()
                .numpy()
            )
        return loss

    def _preprocess_test_data(self, x):
        return self._to_device_tensor(x)

    def _preprocess_training_data(self, x, t, e, vsize, val_data, random_state):
        idx = list(range(x.shape[0]))
        np.random.seed(random_state)
        np.random.shuffle(idx)
        x_train, t_train, e_train = x[idx], t[idx], e[idx]
        x_train = self._to_device_tensor(x_train)
        t_train = self._to_device_tensor(t_train)
        e_train = self._to_device_tensor(e_train)
        if val_data is None:
            vsize = int(vsize * x_train.shape[0])
            x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]
            x_train, t_train, e_train = x_train[:-vsize], t_train[:-vsize], e_train[:-vsize]
        else:
            x_val, t_val, e_val = val_data
            x_val = self._to_device_tensor(x_val)
            t_val = self._to_device_tensor(t_val)
            e_val = self._to_device_tensor(e_val)
        return (x_train, t_train, e_train, x_val, t_val, e_val)

    def predict_mean(self, x, risk=1):
        if not self.fitted:
            raise Exception("Model has not been fitted yet.")
        x = self._preprocess_test_data(x)
        return losses.predict_mean(self.torch_model, x, risk=str(risk))

    def predict_shape_scale(self, x, risk=1):
        if self.fitted:
            x = self._preprocess_test_data(x)
            shapes, scales, logits = self.torch_model.forward(x, risk=str(risk))
            shape = torch.sum(torch.mul(shapes, logits), dim=1) / torch.sum(logits, dim=1)
            scale = torch.sum(torch.mul(scales, logits), dim=1) / torch.sum(logits, dim=1)
            return shape.detach().cpu().numpy(), scale.detach().cpu().numpy()

    def predict_risk(self, x, t, risk=1):
        if self.fitted:
            return 1 - self.predict_survival(x, t, risk=str(risk))
        raise Exception("Model has not been fitted yet.")

    def predict_survival(self, x, t, risk=1):
        x = self._preprocess_test_data(x)
        if not isinstance(t, list):
            t = [t]
        if self.fitted:
            scores = losses.predict_cdf(self.torch_model, x, t, risk=str(risk))
            return np.exp(np.array(scores)).T
        raise Exception("Model has not been fitted yet.")

    def predict_pdf(self, x, t, risk=1):
        x = self._preprocess_test_data(x)
        if not isinstance(t, list):
            t = [t]
        if self.fitted:
            scores = losses.predict_pdf(self.torch_model, x, t, risk=str(risk))
            return np.exp(np.array(scores)).T
        raise Exception("Model has not been fitted yet.")

    def predict_phenotype(self, x, risk=1):
        x = self._preprocess_test_data(x)
        shape, scale, logits = self.torch_model.forward(x, risk=str(risk))
        cluster_tag = np.argmax(logits.detach().cpu().numpy(), axis=1)
        return cluster_tag, shape[0], scale[0]


class AdaCSMSurvivalMachines(AdaCSMBase):
    pass
