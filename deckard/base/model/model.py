import logging
import pickle
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from time import process_time_ns
from typing import Union, Dict
from omegaconf import OmegaConf, DictConfig, ListConfig
from copy import deepcopy
import numpy as np
from sklearn.exceptions import NotFittedError
from ..data import Data
from ..utils import my_hash, factory

from .art_pipeline import ArtPipeline

from .sklearn_pipeline import SklearnModelPipeline

__all__ = ["Model"]
logger = logging.getLogger(__name__)


@dataclass
class ModelInitializer:
    name: str
    kwargs: Union[dict, None] = field(default_factory=dict)
    pipeline: Union[Dict[str, dict], None] = None

    def __init__(self, name, pipeline={}, **kwargs):
        self.name = kwargs.pop("_target_", name)
        if pipeline is not None and len(pipeline) > 0:
            self.pipeline = SklearnModelPipeline(**pipeline)
        else:
            self.pipeline = None
        self.kwargs = kwargs

    def __call__(self):
        logger.info(f"Initializing model {self.name} with kwargs {self.kwargs}")
        if "kwargs" in self.kwargs:
            kwargs = self.kwargs.pop("kwargs", {})
            params = self.kwargs
            params.update(**kwargs)
        else:
            params = self.kwargs
        if "input_dim" in params:
            if isinstance(params["input_dim"], list):
                params["input_dim"] = tuple(params["input_dim"])
            elif isinstance(params["input_dim"], int):
                params["input_dim"] = params["input_dim"]
            elif isinstance(params["input_dim"], ListConfig):
                input_dim_list = tuple(
                    OmegaConf.to_container(params["input_dim"], resolve=True),
                )
                if len(input_dim_list) == 1:
                    params["input_dim"] = input_dim_list[0]
                else:
                    params["input_dim"] = tuple(input_dim_list)
            else:
                raise ValueError(
                    f"input_dim must be a list or tuple. Got {type(params['input_dim'])}",
                )
        if "output_dim" in params:
            if isinstance(params["output_dim"], list):
                params["output_dim"] = tuple(params["output_dim"])
            elif isinstance(params["output_dim"], int):
                params["output_dim"] = params["output_dim"]
            elif isinstance(params["output_dim"], ListConfig):
                output_dim_list = OmegaConf.to_container(
                    params["output_dim"],
                    resolve=True,
                )
                if len(output_dim_list) == 1:
                    params["output_dim"] = output_dim_list[0]
                else:
                    params["output_dim"] = tuple(output_dim_list)
            else:
                raise ValueError(
                    f"output_dim must be a list or tuple. Got {type(params['output_dim'])}",
                )
        name = params.pop("name", self.name)
        if self.pipeline is not None:
            pipeline = deepcopy(self.pipeline)
            obj = factory(name, **params)
            if isinstance(pipeline, DictConfig):
                pipeline = OmegaConf.to_container(pipeline, resolve=True)
            elif isinstance(pipeline, dict):
                pipeline = pipeline
            elif is_dataclass(pipeline):
                pipeline = asdict(pipeline)
            else:
                raise ValueError(
                    f"Pipeline must be a dict or DictConfig or dataclass. Got {type(pipeline)}",
                )
            pipe_conf = SklearnModelPipeline(**pipeline["pipeline"])
            model = pipe_conf(obj)
        else:
            model = factory(name, **params)
        return model

    def __hash__(self):
        return int(my_hash(self), 16)


@dataclass
class ModelTrainer:
    kwargs: dict = field(default_factory=dict)

    def __init__(self, **kwargs):
        logger.info(f"Initializing model trainer with kwargs {kwargs}")
        self.kwargs = kwargs

    def __hash__(self):
        return int(my_hash(self), 16)

    def __call__(self, data: list, model: object, library=None):
        logger.info(f"Training model {model} with fit params: {self.kwargs}")

        trainer = self.kwargs
        if library == "sklearn" or library is None:
            pass
        elif library in ["torch", "pytorch"]:
            import torch

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            X_train, X_test, y_train, y_test = data
            X_train = torch.FloatTensor(X_train).to(device)
            X_test = torch.FloatTensor(X_test).to(device)
            y_train = torch.LongTensor(y_train).to(device)
            y_test = torch.LongTensor(y_test).to(device)
        elif library in ["tensorflow", "tf"]:
            import tensorflow as tf

            tf.config.run_functions_eagerly(True)
        else:
            raise NotImplementedError(f"Training library {library} not implemented")
        try:
            start = process_time_ns()
            model.fit(data[0], data[2], **trainer)
            end = process_time_ns() - start
        except np.AxisError:
            from art.utils import to_categorical

            data[2] = to_categorical(data[2])
            start = process_time_ns()
            model.fit(data[0], data[2], **trainer)
            end = process_time_ns() - start
        except ValueError as e:
            if "Shape of labels" in str(e):
                from art.utils import to_categorical

                nb_classes = len(np.unique(data[2]))
                data[2] = to_categorical(data[2], nb_classes=nb_classes)
                start = process_time_ns()
                model.fit(data[0], data[2], **trainer)
                end = process_time_ns() - start
            else:
                raise e
        except AttributeError as e:
            logger.warning(f"AttributeError: {e}. Trying to fit model anyway.")
            try:
                data[0] = np.array(data[0])
                data[2] = np.array(data[2])
                start = process_time_ns()
                model.fit(data[0], data[2], **trainer)
                end = process_time_ns() - start
            except Exception as e:
                raise e
        except RuntimeError as e:
            if "eager mode" in str(e):
                import tensorflow as tf

                tf.config.run_functions_eagerly(True)
                start = process_time_ns()
                model.fit(data[0], data[2], **trainer)
                end = process_time_ns() - start
            else:
                raise e
        time_dict = {
            "train_time": end / 1e9,
            "train_time_per_sample": end / (len(data[0]) * 1e9),
        }
        return model, time_dict


@dataclass
class Model:
    data: Data = field(default_factory=Data)
    init: ModelInitializer = field(default_factory=ModelInitializer)
    trainer: ModelTrainer = field(default_factory=ModelTrainer)
    art: Union[ArtPipeline, None] = field(default_factory=ArtPipeline)
    library: Union[str, None] = None
    name: str = None

    def __init__(
        self,
        data,
        init,
        trainer=None,
        art=None,
        library=None,
        name: str = None,
        **kwargs,
    ):
        self.data = data
        if isinstance(init, ModelInitializer):
            self.init = init
        elif isinstance(init, dict):
            self.init = ModelInitializer(**init)
        elif isinstance(init, DictConfig):
            init_dict = OmegaConf.to_container(init, resolve=True)
            self.init = ModelInitializer(**init_dict)
        else:
            raise ValueError(
                f"Init {init} is not a dictionary or ModelInitializer object. It is of type {type(init)}",
            )
        assert isinstance(self.init, ModelInitializer)
        self.library = str(init.name).split(".")[0] if library is None else library
        if isinstance(trainer, dict):
            self.trainer = ModelTrainer(**trainer)
        elif isinstance(trainer, type(None)):
            self.trainer = ModelTrainer()
        elif isinstance(trainer, ModelTrainer):
            self.trainer = trainer
        elif isinstance(trainer, DictConfig):
            train_dict = OmegaConf.to_container(trainer, resolve=True)
            self.trainer = ModelTrainer(**train_dict)
        else:
            raise ValueError(
                f"Trainer {trainer} is not a dictionary or ModelTrainer object. It is of type {type(trainer)}",
            )

        while "kwargs" in kwargs:
            kwargs.update(**kwargs.pop("kwargs", {}))
        kwargs.pop("library", None)
        kwargs.pop("data", None)
        kwargs.pop("init", None)
        kwargs.pop("trainer", None)
        name = kwargs.pop("name", None)
        if isinstance(art, ArtPipeline):
            art_dict = asdict(art)
            art_dict.update(**kwargs)
            art_dict.update({"library": self.library})
        elif isinstance(art, type(None)):
            art_dict = None
        elif isinstance(art, dict):
            art_dict = deepcopy(kwargs)
            art_dict.update({"library": self.library})
        elif isinstance(art, DictConfig):
            art_dict = OmegaConf.to_container(art, resolve=True)
            art_dict.update(**kwargs)
            art_dict.update({"library": self.library})
        else:
            raise ValueError(
                f"Art {art} is not a dictionary or ArtPipeline object. It is of type {type(art)}",
            )
        if art_dict is not None:
            self.art = ArtPipeline(**art_dict)
        else:
            self.art = None
        self.name = my_hash(self) if name is None else str(name)
        logger.info(
            f"Initializing model with data {self.data}, init {self.init}, trainer {self.trainer}, art {self.art}",
        )

    def __hash__(self):
        return int(my_hash(self), 16)

    def __call__(
        self,
        data=None,
        model=None,
        data_file=None,
        model_file=None,
        predictions_file=None,
        probabilities_file=None,
        time_dict_file=None,
        loss_file=None,
    ):
        result_dict = {}
        if isinstance(data, Data):
            data = data.initialize()
        elif isinstance(data, type(None)):
            data = self.data.initialize()
        elif isinstance(data, (str, Path)):
            data = self.load(data)
        assert isinstance(
            data,
            (type(None), list, tuple),
        ), f"Data {data} is not a list. It is of type {type(data)}."
        assert len(data) == 4, f"Data {data} is not a tuple of length 4."
        result_dict["data"] = data
        if isinstance(model, Model):
            data, model = model.initialize(data)
        elif isinstance(model, type(None)):
            data, model = self.initialize(data)
            assert len(data) == 4, f"Data {data} is not a tuple of length 4."
        elif isinstance(model, (str, Path)):
            model = self.load(model)
        elif hasattr(model, "fit"):
            assert hasattr(model, "predict") or hasattr(
                model,
                "predict_proba",
            ), f"Model {model} does not have a predict or predict_proba method."
        else:
            raise ValueError(f"Model {model} is not a valid model.")
        result_dict["model"] = model

        if predictions_file is not None and Path(predictions_file).exists():
            preds = self.data.load(predictions_file)
            result_dict["predictions"] = preds
        if probabilities_file is not None and Path(probabilities_file).exists():
            probs = self.data.load(probabilities_file)
            result_dict["probabilities"] = probs
        if loss_file is not None and Path(loss_file).exists():
            loss = self.data.load(loss_file)
            result_dict["loss"] = loss
        if time_dict_file is not None and Path(time_dict_file).exists():
            time_dict = self.data.load(time_dict_file)
        if [
            predictions_file,
            probabilities_file,
            time_dict_file,
            loss_file,
            model_file,
        ].count(None) != 5:
            time_dict = locals().get("time_dict", {})
            result_dict["time_dict"] = time_dict
            # Fitting
            if model_file is None:
                model, fit_time_dict = self.fit(
                    data=data,
                    model=model,
                    model_file=model_file,
                )
                time_dict.update(**fit_time_dict)
                result_dict["model"] = model
                result_dict["data"] = data
                result_dict["time_dict"].update(**fit_time_dict)
            elif Path(model_file).exists():
                model = self.load(model_file)
                result_dict["model"] = model
            else:
                model, fit_time_dict = self.fit(
                    data=data,
                    model=model,
                    model_file=model_file,
                )
                result_dict["model"] = model
                result_dict["data"] = data
                result_dict["time_dict"].update(**fit_time_dict)
            # Predicting
            if predictions_file is not None and not Path(predictions_file).exists():
                preds, pred_time_dict = self.predict(
                    data=data,
                    model=model,
                    predictions_file=predictions_file,
                )
                result_dict["time_dict"].update(**pred_time_dict)
                result_dict["predictions"] = preds
            elif predictions_file is not None and Path(predictions_file).exists():
                preds = self.data.load(predictions_file)
                result_dict["predictions"] = preds
            else:
                preds, pred_time_dict = self.predict(
                    data=data,
                    model=model,
                    predictions_file=predictions_file,
                )
                result_dict["time_dict"].update(**pred_time_dict)
                result_dict["predictions"] = preds
            # Predicting probabilities
            if probabilities_file is not None:
                probs, prob_time_dict = self.predict_proba(
                    data=data,
                    model=model,
                    probabilities_file=probabilities_file,
                )
                result_dict["probabilities"] = probs
                result_dict["time_dict"].update(**prob_time_dict)
            elif probabilities_file is not None and Path(probabilities_file).exists():
                probs, prob_time_dict = self.data.load(probabilities_file)
                result_dict["probabilities"] = probs
                result_dict["time_dict"].update(**prob_time_dict)
            else:
                probs, prob_time_dict = self.predict_proba(
                    data=data,
                    model=model,
                    probabilities_file=probabilities_file,
                )
                result_dict["probabilities"] = probs
                result_dict["time_dict"].update(**prob_time_dict)
            # Predicting loss
            if loss_file is not None:
                loss, loss_time_dict = self.predict_log_loss(
                    data=data,
                    model=model,
                    loss_file=loss_file,
                )
                time_dict.update(**loss_time_dict)
                result_dict["loss"] = loss
                result_dict["time_dict"].update(**loss_time_dict)
            elif loss_file is not None and Path(loss_file).exists():
                loss = self.data.load(loss_file)
                result_dict["loss"] = loss
            else:
                loss, loss_time_dict = self.predict_log_loss(
                    data=data,
                    model=model,
                    loss_file=loss_file,
                )
                time_dict.update(**loss_time_dict)
                result_dict["loss"] = loss
                result_dict["time_dict"].update(**loss_time_dict)
            if time_dict_file is not None:
                if Path(time_dict_file).exists():
                    old_time_dict = self.data.load(time_dict_file)
                    old_time_dict.update(**result_dict["time_dict"])
                    time_dict = old_time_dict
                self.data.save(time_dict, time_dict_file)
                result_dict["time_dict"] = time_dict
        if data_file is not None:
            self.data.save(data, data_file)
        if model_file is not None:
            self.save(model, model_file)
        return result_dict

    def initialize(self, data=None, model=None):
        """Initializes the model with the data and returns the data and model.

        :param data: The data to initialize the model with.
        :type data: list, str, Path, Data
        :param model: The model to initialize.
        :type model: str, Path, Model
        Returns:
            tuple: The data and model as Data and Model objects.
        """
        if isinstance(data, Data):
            data = data.initialize()
        elif isinstance(data, (str, Path)):
            data = self.data(data)
        elif isinstance(data, type(None)):
            data = self.data.initialize()
        assert isinstance(
            data,
            (type(None), list),
        ), f"Data {data} is not a list. It is of type {type(data)}."
        if isinstance(model, (str, Path)) and Path(model).exists():
            model = self.load(model)
        else:
            try:
                model = self.init()
            except RuntimeError as e:
                if "disable eager execution" in str(e):
                    logger.warning("Disabling eager execution for Tensorflow.")
                    import tensorflow as tf

                    tf.compat.v1.disable_eager_execution()
                    model = self.init()
                elif "eager" in str(e):
                    logger.warning("Enabling eager execution for Tensorflow.")
                    import tensorflow as tf

                    tf.config.run_functions_eagerly(True)
                    model = self.init()
                else:
                    raise e
            if self.art is not None:
                model = self.art(model=model, data=data)
            else:
                pass
        assert hasattr(model, "fit"), f"Model {model} does not have a fit method."
        return data, model

    def fit(self, data, model, model_file=None):
        """Fits the model the data and returns the average time per sample.
        :param data: The data to fit the model to.
        :type data: tuple
        :return: The fitted model and the average time per sample.
        """
        if isinstance(data, Data):
            data = data.initialize()
        elif isinstance(data, (str, Path)):
            data = self.data(data)
        assert isinstance(data, (type(None), list)), f"Data {data} is not a tuple."
        if isinstance(model, Model):
            data, model = model.initialize(data)
        elif isinstance(model, (str, Path)):
            model = self.load(model)
        assert hasattr(model, "fit"), f"Model {model} does not have a fit method."
        model, time_dict = self.trainer(data, model)
        assert hasattr(model, "fit"), f"Model {model} does not have a fit method."
        if model_file is not None:
            self.save(model, model_file)
        return model, time_dict

    def predict(self, data=None, model=None, predictions_file=None):
        """Predicts on the data and returns the average time per sample.
        :param model: The model to use for prediction.
        :type model: object
        :param data: The data to predict on.
        """
        if isinstance(data, Data):
            data = data.initialize()
        elif isinstance(data, (str, Path)):
            data = self.data(data)
        elif isinstance(data, type(None)):
            data = self.data.initialize()
        assert isinstance(data, (type(None), list)), f"Data {data} is not a tuple."
        if isinstance(model, (Model)):
            data, model = model.initialize(data)
        elif isinstance(model, (str, Path)):
            model = self.load(model)
        elif isinstance(model, type(None)):
            data, model = self.initialize(data)
        assert hasattr(
            model,
            "predict",
        ), f"Model {model} does not have a predict method."
        try:
            start = process_time_ns()
            predictions = model.predict(data[1])
        except NotFittedError as e:
            logger.warning(e)
            logger.warning(f"Model {model} is not fitted. Fitting now.")
            self.fit(data=data, model=model)
            start = process_time_ns()
            predictions = model.predict(data[1])
        except TypeError as e:
            if "np.float32" in str(e):
                data[1] = data[1].astype(np.float32)
                start = process_time_ns()
                predictions = model.predict(data[1])
            else:
                raise e
        except Exception as e:
            logger.error(e)
            raise e
        end = process_time_ns() - start
        if predictions_file is not None:
            self.data.save(predictions, predictions_file)
        return (
            predictions,
            {
                "predict_time": end / 1e9,
                "predict_time_per_sample": end / (len(data[0]) * 1e9),
            },
        )

    def predict_proba(self, data=None, model=None, probabilities_file=None):
        """Predicts on the data and returns the average time per sample.
        :param model: The model to use for prediction.
        :type model: object
        :param data: The data to predict on.
        :type data: tuple
        :return: The predictions and the average time per sample.
        """
        if isinstance(data, Data):
            data = data.initialize()
        elif isinstance(data, type(None)):
            data = self.data.initialize()
        elif isinstance(data, (str, Path)):
            data = self.data(data)
        else:
            assert isinstance(data, list), f"Data {data} is not a list."
        if isinstance(model, Model):
            data, model = model.initialize(data)
        elif isinstance(model, (str, Path)):
            model = self.load(model)
        if (
            str("art") in str(type(model))
            and "sklearn" in str(type(model))
            and hasattr(model.model, "predict_proba")
        ):
            model = model.model
            logger.warning(
                "Predicting probabilities on ART sklearn models is not supported. Using the underlying model instead.",
            )
        elif hasattr(model, "predict_proba"):
            start = process_time_ns()
            predictions = model.predict_proba(data[1])
            end = process_time_ns() - start
        else:
            start = process_time_ns()
            predictions = model.predict(data[1])
            end = process_time_ns() - start
        if probabilities_file is not None:
            self.data.save(predictions, probabilities_file)
        return (
            predictions,
            {
                "predict_proba_time": end / 1e9,
                "predict_proba_time_per_sample": end / (len(data[0]) * 1e9),
            },
        )

    def predict_log_loss(self, data, model, loss_file=None):
        """Predicts on the data and returns the average time per sample.
        :param model: The model to use for prediction.
        :type model: object
        :param data: The data to predict on.
        :type data: tuple
        :return: The predictions and the average time per sample.
        """
        if isinstance(data, Data):
            data = data.initialize()
        elif isinstance(data, (str, Path)):
            data = self.data(data)
        assert isinstance(data, (type(None), list)), f"Data {data} is not a tuple."
        if isinstance(model, Model):
            data, model = model.initialize(data)
        elif isinstance(model, (str, Path)):
            model = self.load(model)
        if str("art") in str(type(model)) and (
            hasattr(model.model, "predict_log_proba")
            or hasattr(model.model, "predict_proba")
        ):
            model = model.model
            logger.warning(
                "Predicting probabilities on ART models is not supported. Using the underlying model instead.",
            )
        if hasattr(model, "predict_log_proba"):
            start = process_time_ns()
            predictions = model.predict_log_proba(data[1])
            end = process_time_ns() - start
        elif hasattr(model, "predict_proba"):
            start = process_time_ns()
            predictions = model.predict_proba(data[1])
            end = process_time_ns() - start
        elif hasattr(model, "predict"):
            start = process_time_ns()
            predictions = model.predict(data[1])
            end = process_time_ns() - start
        else:
            raise ValueError(
                f"Model {model} does not have a predict_log_proba or predict_proba method.",
            )
        if loss_file is not None:
            self.data.save(predictions, loss_file)
        return (
            predictions,
            {
                "predict_log_proba_time": end / 1e9,
                "predict_log_proba_time_per_sample": end / (len(data[0]) * 1e9),
            },
        )

    def load(self, filename):
        """Loads a model from a file."""
        suffix = Path(filename).suffix
        if suffix in [".pkl", ".pickle"]:
            with open(filename, "rb") as f:
                model = pickle.load(f)
        elif suffix in [".pt", ".pth"]:
            import torch as t

            model = t.load(filename)
            model.load_state_dict(
                t.load(Path(filename).with_suffix(f".optimizer{suffix}")),
            )
            model = self.art(model=model, data=self.data())
        elif suffix in [".wt", ".h5"]:
            import keras as k

            model = k.models.load_model(filename)
            model = self.art(model=model, data=self.data())
        elif suffix in [".tf", "_tf"]:
            import tensorflow as tf

            model = tf.keras.models.load_model(filename)
        else:
            raise ValueError(f"Unknown file type {suffix}")
        return model

    def save(self, model, filename):
        suffix = Path(filename).suffix
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        if suffix in [".pickle", ".pkl"]:
            with open(filename, "wb") as f:
                pickle.dump(model, f)
        elif suffix in [".pt", ".pth"]:
            import torch as t

            while hasattr(model, "model"):
                model = model.model
            t.save(model, filename)
            t.save(
                model.state_dict(),
                Path(filename).with_suffix(f".optimizer{suffix}"),
            )
        elif suffix in [".h5", ".wt"]:
            import keras as k

            while hasattr(model, "model"):
                model = model.model
            try:
                k.models.save_model(model, filename)
            except NotImplementedError as e:
                logger.warning(e)
                logger.warning(
                    f"Saving model to {suffix} is not implemented. Using model.save_weights instead.",
                )
                model.save_weights(filename)
        elif suffix in [".tf", "_tf"]:
            import keras as k

            while hasattr(model, "model"):
                model = model.model
            k.models.save_model(model, filename, save_format="tf")
        else:
            raise NotImplementedError(
                f"Saving model to {suffix} is not implemented. You can add support for your model by adding a new method to the class {self.__class__.__name__} in {__file__}",
            )
