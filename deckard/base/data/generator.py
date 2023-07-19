import logging
from sklearn.datasets import (
    make_classification,
    make_regression,
    make_blobs,
    make_moons,
    make_circles,
)
from typing import Literal
from dataclasses import dataclass, field
from ..utils import my_hash
import numpy as np
from art.utils import load_mnist, load_cifar10

__all__ = [
    "SklearnDataGenerator",
    "TorchDataGenerator",
    "KerasDataGenerator",
    "DataGenerator",
    "supported_sklearn_datasets",
    "supported_torch_datasets",
    "supported_keras_datasets",
    "supported_tf_datasets",
    "supported_datasets"
]

logger = logging.getLogger(__name__)

supported_sklearn_datasets = ["classification", "regression", "blobs", "moons", "circles"]
supported_torch_datasets = ["torch_mnist", "torch_cifar10", "torch_cifar"]
supported_keras_datasets = ["keras_mnist", "keras_cifar10", "keras_cifar"]
supported_tf_datasets = ["mnist", "cifar10", "cifar"]
supported_datasets = supported_sklearn_datasets + supported_torch_datasets + supported_keras_datasets + supported_tf_datasets
@dataclass
class SklearnDataGenerator:
    name: Literal["classification", "regression"] = "classification"
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        """Instantiates a data generator for sklearn datasets. Supported datasets are:
        :param name: Name of the dataset to generate
        :type name: str
        :param kwargs: Keyword arguments for the dataset generator
        :type kwargs: dict
        """
        logger.info(
            f"Instantiating {self.__class__.__name__} with name={name} and kwargs={kwargs}",
        )
        if str(name).lower().strip() in supported_sklearn_datasets:
            self.name = str(name).lower().strip()
        else:
            raise ValueError(f"Unknown dataset name {name}")
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def __call__(self) -> list:
        """Return the generated data
        :return: X, y
        :rtype: list
        """
        if str(self.name) == "classification":
            X, y = make_classification(**self.kwargs)
        elif str(self.name) == "regression":
            X, y = make_regression(**self.kwargs)
        elif str(self.name) == "blobs":
            X, y = make_blobs(**self.kwargs)
        elif str(self.name) == "moons":
            X, y = make_moons(**self.kwargs)
        elif str(self.name) == "circles":
            X, y = make_circles(**self.kwargs)
        return [X, y]

    def __hash__(self):
        return int(my_hash(self), 16)


@dataclass
class TorchDataGenerator:
    name: Literal["torch_mnist", "torch_cifar10"] = "torch_mnist"
    path = None
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name, path=None, **kwargs):
        """Instantiates a data generator for torch datasets
        :param name: Name of the dataset to generate. Supported datasets are: torch_mnist, torch_cifar10
        :type name: str
        :param path: Path to the dataset (optional)
        :type path: str
        :param kwargs: Keyword arguments for the dataset generator
        :type kwargs: dict
        """
        logger.info(
            f"Instantiating {self.__class__.__name__} with name={name} and kwargs={kwargs}",
        )
        if name in supported_torch_datasets:
            self.name = name
        else:
            raise ValueError(f"Unknown dataset name {name}")
        self.path = path
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def __call__(self) -> list:
        """Return the generated data
        :return: X, y
        :rtype: list
        """
        if str(self.name) == "torch_mnist":
            (X_train, y_train), (X_test, y_test), _, _ = load_mnist()
            # Append train and test data to create
            X_train = np.transpose(X_train, (0, 3, 1, 2)).astype(np.float32)
            X_test = np.transpose(X_test, (0, 3, 1, 2)).astype(np.float32)
            X = np.concatenate((X_train, X_test))
            y = np.concatenate((y_train, y_test))
        elif str(self.name) == "torch_cifar10" or str(self.name) == "torch_cifar":
            (X_train, y_train), (X_test, y_test), _, _ = load_cifar10()
            # Append train and test data to create
            X_train = np.transpose(X_train, (0, 3, 1, 2)).astype(np.float32)
            X_test = np.transpose(X_test, (0, 3, 1, 2)).astype(np.float32)
            X = np.concatenate((X_train, X_test))
            y = np.concatenate((y_train, y_test))
        return [X, y]

    def __hash__(self):
        return int(my_hash(self), 16)


@dataclass
class KerasDataGenerator:
    name: str = "keras_mnist"
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        """Instantiates a data generator for keras datasets
        :param name: Name of the dataset to generate. Supported datasets are: keras_mnist, keras_cifar10, keras_cifar
        :type name: str
        :param kwargs: Keyword arguments for the dataset generator
        :type kwargs: dict
        """
        logger.info(
            f"Instantiating {self.__class__.__name__} with name={name} and kwargs={kwargs}",
        )
        if name in supported_keras_datasets or name in supported_tf_datasets:
            self.name = name
        else:
            raise ValueError(f"Unknown dataset name {name}")
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def __call__(self) -> list:
        """Return the generated data
        :return: X, y
        :rtype: list
        """
        if "cifar" in self.name:
            (X_train, y_train), (X_test, y_test), _, _ = load_cifar10()
            X = np.concatenate((X_train, X_test))
            y = np.concatenate((y_train, y_test))
        elif "mnist" in self.name:
            (X_train, y_train), (X_test, y_test), _, _ = load_mnist()
            X = np.concatenate((X_train, X_test))
            y = np.concatenate((y_train, y_test))
        return [X, y]

    def __hash__(self):
        return int(my_hash(self), 16)


@dataclass
class DataGenerator:
    name: str = "classification"
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        """Instantiates a data generator for sklearn, torch and keras datasets
        :param name: Name of the dataset to generate. Supported datasets are: classification, regression, blobs, moons, circles, torch_mnist, torch_cifar10, torch_cifar, keras_mnist, keras_cifar10, keras_cifar
        :type name: str
        :param kwargs: Keyword arguments for the dataset generator
        :type kwargs: dict
        """
        self.name = name
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def __call__(self) -> list:
        """ Return the generated data
        :return: X, y
        :rtype: list
        """
        if str(self.name) in supported_sklearn_datasets:
            return SklearnDataGenerator(self.name, **self.kwargs)()
        elif str(self.name) in supported_torch_datasets:
            return TorchDataGenerator(self.name, **self.kwargs)()
        elif str(self.name) in supported_keras_datasets or str(self.name) in supported_tf_datasets:
            return KerasDataGenerator(self.name, **self.kwargs)()

    def __hash__(self):
        return int(my_hash(self), 16)
