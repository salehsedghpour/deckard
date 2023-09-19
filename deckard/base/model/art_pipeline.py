import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Dict, Union
from art.estimators import BaseEstimator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .keras_models import KerasInitializer, keras_dict
from .tensorflow_models import (
    TensorflowV1Initializer,
    TensorflowV2Initializer,
    tensorflow_dict,
)
from .torch_models import TorchInitializer, torch_dict
from .sklearn_pipeline import SklearnModelInitializer, sklearn_dict
from ..utils import my_hash, factory

__all__ = ["ArtPipelineStage", "ArtModelPipeline"]
logger = logging.getLogger(__name__)


all_models = {**sklearn_dict, **torch_dict, **keras_dict, **tensorflow_dict}
supported_models = all_models.keys()

__all__ = ["ArtPipelineStage", "ArtPipeline", "ArtInitializer"]


@dataclass
class ArtPipelineStage:
    """
    Represents a stage in an art pipeline.

    Args:
        name (Union[str, None], optional): The name of the pipeline stage. Defaults to None.
        kwargs (dict, optional): Additional keyword arguments for the pipeline stage. Defaults to an empty dictionary.
    Methods:
        __init__(self, name=None, **kwargs): Initializes the ArtPipelineStage object.
        __hash__(self): Computes the hash value of the ArtPipelineStage object.
        __call__(self): Executes the pipeline stage.
    """

    name: Union[str, None] = None
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name=None, **kwargs):
        """
        Initializes a new instance of the ArtPipelineStage class.

        Args:
            name (Union[str, None], optional): The name of the pipeline stage. Defaults to None.
            **kwargs: Additional keyword arguments for the pipeline stage.
        """
        logger.info(f"Creating pipeline stage: {name} kwargs: {kwargs}")
        self.name = name
        self.kwargs = kwargs

    def __hash__(self):
        """
        Computes the hash value of the ArtPipelineStage object.

        Returns:
            int: The hash value of the object.
        """
        return int(my_hash(self), 16)

    def __call__(self):
        """
        Executes the pipeline stage.

        Returns:
            tuple: A tuple containing the name of the pipeline stage and its keyword arguments.
        """
        return self.name, self.kwargs


@dataclass
class ArtInitializer:
    """
    Represents a model initializer for an art pipeline.

    Args:
        model (object): The model object to be used in the pipeline.
        library (str): The library used by the model.
        data (list, optional): The data to be used in the pipeline. Defaults to None.
        kwargs (dict, optional): Additional keyword arguments for the initializer. Defaults to an empty dictionary.
    
    Attributes:
        library (str): The library used by the model.
        data (list): The data to be used in the pipeline.
        model (object): The model object to be used in the pipeline.
        kwargs (dict): Additional keyword arguments for the initializer.
    
    Methods:

    """

    library: str = None
    data: list = None
    model: object = None
    kwargs: dict = field(default_factory=dict)

    def __init__(self, model: object, library: str, data=None, **kwargs):
        """
        Initializes a new instance of the ArtInitializer class.

        Args:
            model (object): The model object to be used in the pipeline.
            library (str): The library used by the model.
            data (list, optional): The data to be used in the pipeline. Defaults to None.
            **kwargs: Additional keyword arguments for the initializer.
        """
        assert (
            library in supported_models
        ), f"library must be one of {supported_models}. Got {library}"
        self.library = library
        self.data = data
        self.model = model
        self.kwargs = kwargs

    def __hash__(self):
        """
        Computes the hash value of the ArtInitializer object.
        """
        return int(my_hash(self), 16)

    def __call__(self):
        """
        Returns the art model object.
        Returns:
            BaseEstimator: The art model object.
        """
        library = self.library
        data = self.data
        model = self.model
        kwargs = self.kwargs
        if "torch" in str(library):
            model = TorchInitializer(
                data=data, model=model, library=library, **kwargs
            )()
        elif "keras" in str(library):
            try:
                model = KerasInitializer(
                    data=data, model=model, library=library, **kwargs
                )()
            except ValueError as e:
                if "disable eager execution" in str(e):
                    import tensorflow as tf

                    tf.compat.v1.disable_eager_execution()
                    if str(type(model)).startswith("<class 'art."):
                        model = model.model
                    model = KerasInitializer(
                        data=data, model=model, library=library, **kwargs
                    )()
                else:
                    raise e
        elif "sklearn" in str(library) or library is None:
            model = SklearnModelInitializer(
                data=data, model=model, library=library, **kwargs
            )()
        elif library in ["tf2", "tensorflowv2", "tensorflow", "tf", "tfv2"]:
            model = TensorflowV2Initializer(
                data=data, model=model, library=library, **kwargs
            )()
        elif library in ["tf1", "tensorflowv1", "tfv1"]:
            model = TensorflowV1Initializer(
                data=data, model=model, library=library, **kwargs
            )()
        else:
            raise ValueError(
                f"library must be one of {supported_models}. Got {library}",
            )
        assert hasattr(
            model,
            "fit",
        ), f"model must have a fit method. Got type {type(model)}"
        return model


@dataclass
class ArtPipeline:
    library: str = None
    pipeline: Dict[str, ArtPipelineStage] = field(default_factory=dict)
    name: str = None

    def __init__(self, library, name: str = None, **kwargs):
        """
        Creates a callable art pipeline object from the specified parameters.
        A class representing an Art Pipeline.

        Attributes:
            library (str): The library used for the pipeline.
            pipeline (Dict[str, ArtPipelineStage]): The stages of the pipeline.
            name (str): The name of the pipeline.

        Methods:
            __init__(self, library, name: str = None, **kwargs): Initializes the ArtPipeline object.
            __len__(self): Returns the number of stages in the pipeline.
            __hash__(self): Returns the hash value of the ArtPipeline object.
            __call__(self, model: object, library: str = None, data=None) -> BaseEstimator: Executes the pipeline on the given model.
        """
        self.library = library
        pipeline = deepcopy(kwargs.pop("pipeline", {}))
        pipeline.update(kwargs)
        pipeline.pop("data", None)
        pipeline.pop("model", None)
        for stage in pipeline:
            if isinstance(pipeline[stage], DictConfig):
                pipeline[stage] = OmegaConf.to_container(pipeline[stage])
            elif is_dataclass(pipeline[stage]):
                pipeline[stage] = asdict(pipeline[stage])
            else:
                if not isinstance(pipeline[stage], dict):
                    pipeline[stage] = (
                        {**pipeline[stage]} if pipeline[stage] is not None else {}
                    )
            while "kwargs" in pipeline[stage]:
                pipeline[stage].update(**pipeline[stage].pop("kwargs"))
            while "params" in pipeline[stage]:
                pipeline[stage].update(**pipeline[stage].pop("params"))
            name = pipeline[stage].pop("name", stage)
            params = pipeline[stage]
            params.pop("name", None)
            pipeline[stage] = ArtPipelineStage(name, **params)
        self.pipeline = pipeline
        self.name = kwargs.pop("name", my_hash(vars(self)))

    def __len__(self):
        if self.pipeline is not None:
            return len(self.pipeline)
        return 0

    def __hash__(self):
        return int(my_hash(self), 16)

    # def __iter__(self):
    #     return iter(self.pipeline)

    def __call__(self, model: object, library: str = None, data=None) -> BaseEstimator:
        """
        Executes the pipeline on the given model.
        
        Returns:
            BaseEstimator: The art model object with the defences applied.
        """
        if "initialize" in self.pipeline:
            if isinstance(self.pipeline["initialize"], DictConfig):
                params = OmegaConf.to_container(self.pipeline["initialize"])
                name = params.pop("name", None)
                kwargs = params.pop("kwargs", {})
            elif is_dataclass(self.pipeline["initialize"]):
                params = asdict(self.pipeline["initialize"])
                name = params.pop("name", None)
                kwargs = params.pop("kwargs", {})
            else:
                assert isinstance(self.pipeline["initialize"], dict)
                params = self.pipeline["initialize"]
                name = params.pop("name", None)
                kwargs = params.pop("kwargs", {})

        else:
            raise ValueError("Art Pipeline must have an initialize stage")
        pre_def = []
        post_def = []
        if data is None:
            data = model.data()
        if library is None:
            library = self.library
        else:
            assert (
                library in supported_models
            ), f"library must be one of {supported_models}. Got {library}"
        assert len(data) == 4, f"data must be a tuple of length 4. Got {data}"
        while "kwargs" in kwargs:
            kwargs.update(**kwargs.pop("kwargs"))
        if "preprocessor" in self.pipeline:
            if isinstance(self.pipeline["preprocessor"], DictConfig):
                params = OmegaConf.to_container(self.pipeline["preprocessor"])
                name = params.pop("name", None)
                sub_kwargs = params.pop("kwargs", {})
                while "kwargs" in sub_kwargs:
                    sub_kwargs.update(**sub_kwargs.pop("kwargs"))
            elif is_dataclass(self.pipeline["preprocessor"]):
                params = asdict(self.pipeline["preprocessor"])
                name = params.pop("name", None)
                sub_kwargs = params.pop("kwargs", {})
            else:
                assert isinstance(self.pipeline["preprocessor"], dict)
                params = self.pipeline["preprocessor"]
                name = params.pop("name", None)
                sub_kwargs = params.pop("kwargs", {})
            config = {
                "_target_": name,
            }
            config.update(**sub_kwargs)
            obj = instantiate(config)
            pre_def.append(obj)
            kwargs.update({"preprocessing_defences": pre_def})
        if "postprocessor" in self.pipeline:
            # if isinstance(self.pipeline["postprocessor"], DictConfig):
            #     params = OmegaConf.to_container(self.pipeline["postprocessor"])
            #     name = params.pop("name", "_target_")
            # elif is_dataclass(self.pipeline["postprocessor"]):
            params = asdict(self.pipeline["postprocessor"])
            name = params.pop("name", "_target_")
            # else:
            #     assert isinstance(self.pipeline["postprocessor"], dict)
            #     params = self.pipeline["postprocessor"]
            #     name = params.pop("name", "_target_")
            config = {
                "_target_": name,
            }
            while "kwargs" in params:
                params.update(**params.pop("kwargs"))
            config.update(**params)
            obj = instantiate(config)
            post_def.append(obj)
            kwargs.update({"postprocessing_defences": post_def})
        
        model = ArtInitializer(model=model, data=data, **kwargs, library=library)()
        if "transformer" in self.pipeline:
            name, sub_kwargs = self.pipeline["transformer"]()
            obj = factory(name, model, **sub_kwargs)
            model = obj(transformed_classifier = model, x=data[0])
        return model
