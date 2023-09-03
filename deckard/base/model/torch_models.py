import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.estimators.regression import PyTorchRegressor
from hydra.utils import instantiate


logger = logging.getLogger(__name__)

classifier_dict = {
    "pytorch": PyTorchClassifier,
    "torch": PyTorchClassifier,
}

regressor_dict = {
    "pytorch-regressor": PyTorchRegressor,
    "torch-regressor": PyTorchRegressor,
}

torch_dict = {**classifier_dict, **regressor_dict}
supported_models = list(torch_dict.keys())

__all__ = ["TorchInitializer", "TorchCriterion", "TorchOptimizer"]
dataclass


class TorchCriterion:
    name: str
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        self.name = kwargs.pop("_target_", name)
        self.kwargs = kwargs

    def __call__(self):
        logger.info(f"Initializing model {self.name} with kwargs {self.kwargs}")
        params = self.kwargs
        name = params.pop("_target_", self.name)
        dict_ = {"_target_": name}
        dict_.update(**params)
        obj = instantiate(dict_)
        return obj


@dataclass
class TorchOptimizer:
    name: str
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        self.name = kwargs.pop("_target_", name)
        self.kwargs = kwargs

    def __call__(self, model):
        logger.info(f"Initializing model {self.name} with kwargs {self.kwargs}")
        params = self.kwargs
        name = params.pop("_target_", self.name)
        dict_ = {"_target_": name}
        dict_.update(**params)
        dict_.update({"params": model.parameters()})
        obj = instantiate(dict_)
        return obj


@dataclass
class TorchInitializer:
    data: list
    model: str
    optimizer: TorchOptimizer = field(default_factory=TorchOptimizer)
    criterion: TorchCriterion = field(default_factory=TorchCriterion)
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, data, model, library, **kwargs):
        import torch

        self.data = data
        self.model = model
        assert str(library).lower().strip() in supported_models, ValueError("Unknown model library")
        self.library = library
        self.device = kwargs.pop(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.kwargs = kwargs

    def __call__(self):
        library = self.library
        model = self.model
        kwargs = deepcopy(self.kwargs)
        kwargs.update(**kwargs.pop("kwargs", {}))
        data = self.data

        if "art" in str(type(model)) and hasattr(model, "model"):
            model = model.model
        assert "optimizer" in kwargs, ValueError("Optimizer not specified. Please specify an optimizer dictionary.")
        optimizer = TorchOptimizer(**kwargs.pop("optimizer"))(model)
        kwargs.update({"optimizer": optimizer})
        assert "criterion" in kwargs, ValueError("Criterion not specified. Please specify a criterion dictionary.")
        criterion = TorchCriterion(**kwargs.pop("criterion"))()
        kwargs.update({"loss": criterion})
        if "input_shape" not in kwargs:
            kwargs.update({"input_shape": data[0].shape[1:]})
        if "nb_classes" not in kwargs:
            if len(data[2].shape) == 1:
                kwargs.update({"nb_classes": len(np.unique(data[2]))})
            else:
                kwargs.update({"nb_classes": data[2].shape[1]})
        kwargs.pop("library", None)
        model = torch_dict[library](model, **kwargs)
        model.model.to(self.device)
        return model
