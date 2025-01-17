import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ..attack import Attack
from ..data import Data
from ..files import FileConfig
from ..model import Model
from ..scorer import ScorerDict
from ..utils import to_dict
from ..utils.hashing import my_hash

__all__ = ["Experiment"]
logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    data: Data = field(default_factory=Data)
    model: Union[Model, None] = field(default_factory=Model)
    attack: Union[Attack, None] = field(default_factory=Attack)
    scorers: Union[ScorerDict, None] = field(default_factory=ScorerDict)
    files: Union[FileConfig, None] = field(default_factory=FileConfig)
    name: Union[str, None] = field(default_factory=str)
    stage: Union[str, None] = field(default_factory=str)
    optimizers: Union[list, None] = field(default_factory=list)
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(
        self,
        data: Data,
        model: Model,
        scorers: ScorerDict,
        files: list,
        attack: Attack = None,
        name=None,
        stage=None,
        optimizers=None,
        **kwargs,
    ):
        if isinstance(data, dict):
            self.data = Data(**data)
        elif isinstance(data, DictConfig):
            data_dict = OmegaConf.to_container(data)
            data_dict.update({"_target_": "deckard.base.data.Data"})
            self.data = instantiate(data_dict)
        elif isinstance(data, Data):
            self.data = data
        elif isinstance(data, type(None)):
            raise ValueError("data cannot be None.")
        else:
            raise ValueError("data must be a dict, DictConfig, or Data object.")
        assert isinstance(self.data, Data), f"data is {type(self.data)}"
        if isinstance(model, dict):
            self.model = Model(**model)
        elif isinstance(model, DictConfig):
            model_dict = OmegaConf.to_container(model)
            self.model = Model(**model_dict)
        elif isinstance(model, Model):
            self.model = model
        elif isinstance(model, type(None)):
            self.model = None
        else:
            raise ValueError("model must be a dict, DictConfig, or Model object.")
        assert isinstance(self.model, (Model, type(None)))
        if isinstance(scorers, dict):
            self.scorers = ScorerDict(**scorers)
        elif isinstance(scorers, DictConfig):
            scorer_dict = OmegaConf.to_container(scorers)
            self.scorers = ScorerDict(**scorer_dict)
        elif isinstance(scorers, ScorerDict):
            self.scorers = scorers
        else:
            raise ValueError(
                "scorers must be a dict, DictConfig, or ScorerDict object.",
            )
        assert isinstance(self.scorers, ScorerDict)
        if isinstance(files, dict):
            self.files = FileConfig(**files)
        elif isinstance(files, DictConfig):
            file_dict = OmegaConf.to_container(files)
            self.files = FileConfig(**file_dict)
        elif isinstance(files, FileConfig):
            self.files = files
        else:
            raise ValueError("files must be a dict, DictConfig, or FileConfig object.")
        assert isinstance(self.files, FileConfig)
        if attack is None or (hasattr(attack, "__len__") and len(attack) == 0):
            self.attack = None
        elif isinstance(attack, dict):
            self.attack = Attack(**attack)
        elif isinstance(attack, DictConfig):
            attack_dict = OmegaConf.to_container(attack)
            self.attack = Attack(**attack_dict)
        elif isinstance(attack, Attack):
            self.attack = attack
        elif isinstance(attack, type(None)):
            self.attack = None
        else:
            raise ValueError("attack must be a dict, DictConfig, or Attack object.")
        assert isinstance(self.attack, (Attack, type(None)))
        self.stage = stage
        self.optimizers = optimizers
        self.kwargs = kwargs
        self.name = name if name is not None else self._set_name()
        logger.info("Instantiating Experiment with id: {}".format(self.get_name()))

    def __hash__(self):
        return int(self.name, 16)

    def __call__(self):
        """Runs the experiment. If the experiment has already been run, it will load the results from disk. If scorer is not None, it will return the score for the specified scorer. If scorer is None, it will return the score for the first scorer in the ScorerDict.
        :param scorer: The scorer to return the score for. If None, the score for the first scorer in the ScorerDict will be returned.
        :type scorer: str
        :return: The score for the specified scorer or the status of the experiment if scorer=None (default).
        """
        logger.info("Running experiment with id: {}".format(self.get_name()))
        old_name = self.get_name()
        old_hash = my_hash(self)
        # Setup files, data, and model
        files = deepcopy(self.files).get_filenames()
        # Save params file
        if "params_file" in files and files["params_file"] is not None:
            self.save_params_file()
        # Check status of files
        assert (
            "score_dict_file" in files
        ), f"score_dict_file must be in files. Got {files.keys()}"
        if (
            files["score_dict_file"] is not None
            and Path(files["score_dict_file"]).exists()
        ):
            score_dict = self.data.load(files["score_dict_file"])
        else:
            score_dict = {}
        results = {}
        results["score_dict_file"] = score_dict
        #########################################################################
        data_files = {
            "data_file": files.get("data_file", None),
            "train_labels_file": files.get("train_labels_file", None),
            "test_labels_file": files.get("test_labels_file", None),
            # "time_dict_file": files.get("score_dict_file", None),
            # TODO data_score_file
        }
        data = self.data(**data_files)
        #########################################################################
        if self.model is not None:
            model_files = {
                "model_file": files.get("model_file", None),
                "predictions_file": files.get("predictions_file", None),
                "probabilities_file": files.get("probabilities_file", None),
                "time_dict_file": files.get("score_dict_file", None),
                # TODO train_score_file
                # TODO test_score_file
            }
            model_results = self.model(data, **model_files)
            score_dict.update(**model_results.pop("time_dict", {}))
            score_dict.update(**model_results.pop("score_dict", {}))
            model = model_results["model"]
            # Prefer probabilities, then loss_files, then predictions
            if (
                "probabilities" in model_results
                and model_results["probabilities"] is not None
            ):
                probs = model_results["probabilities"]
                if not hasattr(probs, "shape"):
                    probs = np.array(probs)
                logger.debug(f"probs shape: {probs.shape}")
            if (
                "predictions" in model_results
                and model_results["predictions"] is not None
            ):
                preds = model_results["predictions"]
                if not hasattr(preds, "shape"):
                    preds = np.array(preds)
                logger.debug(f"preds shape: {preds.shape}")
            if "losses" in model_results and model_results["losses"] is not None:
                losses = model_results["losses"]
                if not hasattr(losses, "shape"):
                    losses = np.array(losses)
                logger.debug(f"losses shape: {losses.shape}")
        else:
            preds = data[2]
        ##########################################################################
        if self.attack is not None:
            adv_results = self.attack(
                data,
                model,
                attack_file=files.get("attack_file", None),
                adv_predictions_file=files.get("adv_predictions_file", None),
                adv_probabilities_file=files.get("adv_probabilities_file", None),
                adv_losses_file=files.get("adv_losses_file", None),
            )
            if "adv_predictions" in adv_results:
                adv_preds = adv_results["adv_predictions"]
                if not hasattr(adv_preds, "shape"):
                    adv_preds = np.array(adv_preds)
                logger.debug(f"adv_preds shape: {adv_preds.shape}")
            if "adv_losses" in adv_results:
                adv_preds = adv_results["adv_losses"]
                if not hasattr(adv_preds, "shape"):
                    adv_preds = np.array(adv_preds)
                logger.debug(f"adv_losses shape: {adv_preds.shape}")
            if "adv_probabilities" in adv_results:
                adv_preds = adv_results["adv_probabilities"]
                if not hasattr(adv_preds, "shape"):
                    adv_preds = np.array(adv_preds)
                logger.debug(f"adv_probabilities shape: {adv_preds.shape}")
            if "time_dict" in adv_results:
                adv_time_dict = adv_results["time_dict"]
                score_dict.update(**adv_time_dict)
        # #########################################################################
        if self.scorers is not None:
            if "probs" in locals() and "preds" not in locals():
                preds = probs
            elif "losses" in locals() and "preds" not in locals():
                preds = losses
            if "preds" in locals() and self.model is not None:
                if not hasattr(data[3], "shape"):
                    ground_truth = np.array(data[3])
                else:
                    pass
                ground_truth = data[3][: len(preds)]
                if not hasattr(preds, "shape"):
                    preds = np.array(preds)
                logger.debug(f"preds shape: {preds.shape}")
                logger.debug(f" len(preds) : {len(preds)}")
                if self.model is not None:
                    preds = preds[: len(ground_truth)]
                    ground_truth = ground_truth[: len(preds)]
                else:
                    preds = data[0][: len(ground_truth)]
                    ground_truth = data[1][: len(preds)]
                logger.debug(f" len(preds) : {len(preds)}")
                new_score_dict = self.scorers(ground_truth, preds)
                score_dict.update(**new_score_dict)
                results["score_dict_file"] = score_dict
            if "adv_preds" in locals():
                ground_truth = data[3][: len(adv_preds)]
                adv_preds = adv_preds[: len(ground_truth)]
                adv_score_dict = self.scorers(ground_truth, adv_preds)
                adv_score_dict = {
                    f"adv_{key}": value for key, value in adv_score_dict.items()
                }
                score_dict.update(**adv_score_dict)
                results["score_dict_file"] = score_dict
            # # Save results
            if "score_dict_file" in files and files["score_dict_file"] is not None:
                if Path(files["score_dict_file"]).exists():
                    old_score_dict = self.data.load(files["score_dict_file"])
                    old_score_dict.update(**score_dict)
                    score_dict = old_score_dict
                self.data.save(score_dict, files["score_dict_file"])
        else:
            raise ValueError("Scorer is None. Please specify a scorer.")
        #########################################################################
        # Returns score if scorer is not None, otherwise returns status
        if self.optimizers is not None and self.optimizers != []:
            if not isinstance(self.optimizers, list):
                self.optimizers = [self.optimizers]
            scores = {}
            for scorer in self.optimizers:
                if isinstance(scorer, list):
                    assert (
                        len(scorer) == 1
                    ), f"scorer must be a string or a list of length 1, not {scorer}."
                    scorer = scorer[0]
                try:
                    score = score_dict[scorer]
                except KeyError:
                    raise KeyError(
                        f"Scorer {scorer} not found in score_dict. Available self.optimizers: {score_dict.keys()}",
                    )
                scores[scorer] = score
                logger.info(f"Score for id : {self.get_name()} : {scorer}: {score}")
        else:
            scores = score_dict
            logger.info(f"Score for id : {self.get_name()}: {score_dict}")
        logger.info("Finished running experiment with id: {}".format(self.get_name()))
        new_name = self.get_name()
        assert (
            old_name == new_name
        ), f"Experiment hash() name changed from {old_name} to {new_name}."
        logger.debug(
            f"Experiment deckard hash changed from {old_hash} to {my_hash(self)}.",
        )
        return scores

    def _set_name(self):
        if self.files.name is not None:
            name = self.files.name
        else:
            name = my_hash(vars(self))
        return name

    def get_name(self):
        return self.name

    def save_params_file(self):
        files = self.files.get_filenames()
        params_file = files["params_file"]
        Path(params_file).parent.mkdir(parents=True, exist_ok=True)
        params = to_dict(self)
        with open(params_file, "w") as f:
            yaml.dump(params, f)
