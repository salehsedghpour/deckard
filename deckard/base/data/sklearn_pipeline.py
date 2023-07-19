import logging
from typing import Union
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from dataclasses import dataclass, asdict, field, is_dataclass
from copy import deepcopy
from ..utils import my_hash

__all__ = ["SklearnDataPipelineStage", "SklearnDataPipeline"]
logger = logging.getLogger(__name__)


@dataclass
class SklearnDataPipelineStage:
    name: str
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        logger.info(
            f"Instantiating {self.__class__.__name__} with name={name} and kwargs={kwargs}",
        )
        self.name = name
        self.kwargs = kwargs

    def __hash__(self):
        return int(my_hash(self), 16)

    def __call__(self, X_train, X_test, y_train, y_test):
        name = self.kwargs.pop("_target_", self.name)
        dict_ = {"_target_": name}
        dict_.update(**self.kwargs)
        obj = instantiate(dict_)
        assert hasattr(obj, "fit") and hasattr(obj, "transform")
        X_train = obj.fit(X_train).transform(X_train)
        X_test = obj.transform(X_test)
        return X_train, X_test, y_train, y_test


@dataclass
class SklearnDataPipeline:
    pipeline: Union[dict, None] = field(default_factory=dict)

    def __init__(self, **kwargs):
        """
        Return a pipeline of sklearn transformers
        :param kwargs: Pipeline stages (optional)
        :type kwargs: dict
        :param pipeline: Pipeline stages (optional)
        :type pipeline: dict
        """
        pipe = kwargs.pop("pipeline", {})
        pipe.update(**kwargs)
        for stage in pipe:
            name = pipe[stage].pop("name", pipe[stage].pop("_target_", stage))
            assert isinstance(pipe[stage], DictConfig)
            pipe[stage] = OmegaConf.to_container(pipe[stage])
            pipe[stage] = SklearnDataPipelineStage(name, **pipe[stage])
        self.pipeline = pipe

    def __getitem__(self, key):
        """Return the pipeline stage with the given key"""
        return self.pipeline[key]

    def __len__(self):
        """Return the number of stages in the pipeline"""
        return len(self.pipeline)

    def __hash__(self):
        return int(my_hash(self), 16)

    def __iter__(self):
        """Iterate over the pipeline stages"""
        return iter(self.pipeline)

    def __call__(self, X_train, X_test, y_train, y_test) -> list:
        """Runs the pipeline on the data
        :param X_train: Training data
        :type X_train: np.ndarray
        :param X_test: Testing data
        :type X_test: np.ndarray
        :param y_train: Training labels
        :type y_train: np.ndarray
        :param y_test: Testing labels
        :type y_test: np.ndarray
        :return: Transformed data
        :rtype: list
        """
        logger.info(
            "Calling SklearnDataPipeline with pipeline={}".format(self.pipeline),
        )
        pipeline = deepcopy(self.pipeline)
        for stage in pipeline:
            transformer = pipeline[stage]
            X_train, X_test, y_train, y_test = transformer(
                X_train, X_test, y_train, y_test,
            )
        return [X_train, X_test, y_train, y_test]
