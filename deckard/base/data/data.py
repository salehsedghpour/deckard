import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from pandas import DataFrame, read_csv, read_excel

from ..utils import my_hash
from .generator import DataGenerator
from .sampler import SklearnDataSampler
from .sklearn_pipeline import SklearnDataPipeline

__all__ = ["Data"]
logger = logging.getLogger(__name__)


@dataclass
class Data:
    """Data class for generating and sampling data. If the data is generated, then generate the data and sample it. When called, the data is loaded from file if it exists, otherwise it is generated and saved to file. Returns X_train, X_test, y_train, y_test as a list of arrays, typed according to the framework."""

    generate: Union[DataGenerator, None] = field(default_factory=DataGenerator)
    sample: Union[SklearnDataSampler, None] = field(default_factory=SklearnDataSampler)
    sklearn_pipeline: Union[SklearnDataPipeline, None] = field(
        default_factory=SklearnDataPipeline,
    )
    target: Union[str, None] = None
    name: Union[str, None] = None

    def __init__(
        self,
        name: str = None,
        generate: DataGenerator = None,
        sample: SklearnDataSampler = None,
        sklearn_pipeline: SklearnDataPipeline = None,
        target: str = None,
    ) -> list:
        """Initialize the data object. If the data is generated, then generate the data and sample it. If the data is loaded, then load the data and sample it.
        :param name: Name of the data file
        :type name: str
        :param generate: Data generator
        :type generate: DataGenerator
        :param sample: Data sampler
        :type sample: SklearnDataSampler
        :param sklearn_pipeline: Data pipeline
        :type sklearn_pipeline: SklearnDataPipeline
        :param target: Target column name
        :type target: str
        :return: X_train, X_test, y_train, y_test
        :rtype: list
        """
        logger.info(
            f"Instantiating {self.__class__.__name__} with name={name} and generate={generate} and sample={sample} and sklearn_pipeline={sklearn_pipeline} and target={target}",
        )
        if generate is not None:
            self.generate = (
                generate
                if isinstance(generate, (DataGenerator))
                else DataGenerator(**generate)
            )
        else:
            self.generate = None
        if sample is not None:
            self.sample = (
                sample
                if isinstance(sample, (SklearnDataSampler))
                else SklearnDataSampler(**sample)
            )
        else:
            self.sample = SklearnDataSampler()
        if sklearn_pipeline is not None:
            self.sklearn_pipeline = (
                sklearn_pipeline
                if isinstance(sklearn_pipeline, (SklearnDataPipeline, type(None)))
                else SklearnDataPipeline(**sklearn_pipeline)
            )
        else:
            self.sklearn_pipeline = None
        self.target = target
        if name is None:
            self.name = my_hash(self)
        else:
            assert Path(name).exists(), ValueError(f"Data file {name} does not exist")
        self.name = name 
        logger.debug(f"Instantiating Data with id: {self.get_name()}")

    def get_name(self):
        """Get the name of the data object."""
        return str(self.name)

    def __hash__(self):
        """Get the hash of the data object."""
        return int(my_hash(self), 16)

    def initialize(self):
        """Initialize the data object. If the data is generated, then generate the data and sample it. If the data is loaded, then load the data and sample it.
        :return: X_train, X_test, y_train, y_test
        """
        if self.generate is not None:
            result = self.generate()
            assert len(result) == 2, f"Data is not generated: {self.name}"
            
        else:
            result = self.load(self.name)
        result = self.sample(*result)
        assert len(result) == 4
        if self.sklearn_pipeline is not None:
            result = self.sklearn_pipeline(*result)
        assert len(result) == 4
        return result

    def load(self, filename) -> list:
        """
        Loads data from a file
        :param filename: Absolute path to the file
        :type filename: str
        :return: a list of arrays
        :rtype: list
        """
        suffix = Path(filename).suffix
        if suffix in [".json"]:
            with open(filename, "r") as f:
                data = json.load(f)
        elif suffix in [".csv"]:
            filename = Path(Path().absolute(), filename).as_posix()
            data = read_csv(filename)
            if "X_train" in data.columns:
                assert "X_test" in data.columns
                assert "y_train" in data.columns
                assert "y_test" in data.columns
                X_train = data["X_train"].to_numpy()
                X_test = data["X_test"].to_numpy()
                y_train = data["y_train"].to_numpy()
                y_test = data["y_test"].to_numpy()
                data = [X_train, X_test, y_train, y_test]
            else:
                y = data[self.target]
                X = data.drop(self.target, axis=1)
                X = X.to_numpy()
                y = y.to_numpy()
                data = [X, y]
        elif suffix in [".pkl", ".pickle"]:
            with open(filename, "rb") as f:
                data = pickle.load(f)
        elif suffix in [".xls", ".xlsx"]:
            data = read_excel(filename)
            y = data[self.target]
            X = data.drop(self.target, axis=1)
            X = X.to_numpy()
            y = y.to_numpy()
            data = [X, y]
        else:
            raise ValueError(f"Unknown file type {suffix}")
        return data

    def save(self, data, filename):
        """Save data to a file
        :param data: DataFrame
        :param filename: str
        """
        if filename is not None:
            logger.info(f"Saving data to {filename}")
            suffix = Path(filename).suffix
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            if suffix in [".json"]:
                data = [x.tolist() for x in data]
                with open(filename, "w") as f:
                    json.dump(data, f)
            elif suffix in [".csv"]:
                x_train = data[0].tolist()
                x_test = data[1].tolist()
                y_train = data[2].tolist()
                y_test = data[3].tolist()
                df = DataFrame([x_train, x_test, y_train, y_test]).T
                df.columns=["X_train", "X_test", "y_train", "y_test"]
                df.to_csv(filename)
            elif suffix in [".pkl", ".pickle"]:
                with open(filename, "wb") as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"Unknown file type {type(suffix)} for {suffix}")
            assert Path(filename).exists()

    def __call__(
        self, data_file=None, train_labels_file=None, test_labels_file=None,
    ) -> list:
        """Loads data from file if it exists, otherwise generates data and saves it to file. Returns X_train, X_test, y_train, y_test as a list of arrays, typed according to the framework.
        :param filename: str
        :return: list
        """
        if data_file is not None and Path(data_file).exists():
            data = self.load(data_file)
            assert len(data) == 4, f"Some data is missing: {self.name}"
        else:
            data = self.initialize()
            assert len(data) == 4, f"Some data is missing: {self.name}"
        self.save(data, data_file)
        if train_labels_file is not None:
            self.save(data[2], train_labels_file)
            assert Path(
                train_labels_file,
            ).exists(), f"Error saving train labels to {train_labels_file}"
        if test_labels_file is not None:
            self.save(data[3], test_labels_file)
            assert Path(
                test_labels_file,
            ).exists(), f"Error saving test labels to {test_labels_file}"
        return data