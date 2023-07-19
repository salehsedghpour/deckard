import unittest
from pathlib import Path
import os
import numpy as np
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

from deckard.base.data.generator import (
    DataGenerator,
    SklearnDataGenerator,
    TorchDataGenerator,
    KerasDataGenerator,
    supported_sklearn_datasets,
    supported_torch_datasets,
    supported_keras_datasets,
    supported_tf_datasets,
    supported_datasets,
)


this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()
config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
config_file = "classification.yaml"


class testDataGenerator(unittest.TestCase):
    def setUp(self, config_dir=config_dir, config_file=config_file):
        with initialize_config_dir(
            config_dir=Path(config_dir).resolve().as_posix(), version_base="1.3",
        ):
            cfg = compose(config_name=config_file)
        self.cfg = cfg
        self.data = instantiate(config=self.cfg)

    def test_init(self):
        self.assertTrue(isinstance(self.data.generate, DataGenerator))

    def test_call(self):
        data = self.data.generate()
        self.assertIsInstance(data, list)
        self.assertIsInstance(data[0], np.ndarray)
        self.assertIsInstance(data[1], np.ndarray)
        self.assertEqual(data[0].shape[0], data[1].shape[0])
        self.assertEqual(len(data), 2)

    def test_hash(self):
        old_hash = hash(self.data.generate)
        self.assertIsInstance(old_hash, int)
        new_cfg = self.cfg
        new_data = instantiate(config=new_cfg)
        new_hash = hash(new_data.generate)
        self.assertEqual(old_hash, new_hash)

    def tearDown(self) -> None:
        pass


class testSklearnDataGenerator(unittest.TestCase):
    def setUp(self):
        self.names = ["classification", "regression", "blobs", "moons", "circles", "sasdaf0235"]
        self.kwarg_dict ={
            "blobs" : {"n_samples": 100, "n_features": 2, "centers": 2},
            "circles" : {"n_samples": 100, "noise": 0.1, "factor": 0.5},
            "classification" : {"n_samples": 100, "n_features": 2, "n_redundant": 0, "n_informative": 2, "random_state": 0, "n_clusters_per_class": 1},
            "moons" : {"n_samples": 100, "noise": 0.1},
        }

    def test_init(self):
        for name in self.names:
            if name in supported_sklearn_datasets:
                data = SklearnDataGenerator(name=name, **self.kwarg_dict.get(name, {}))
                self.assertTrue(isinstance(data, SklearnDataGenerator))
            else:
                self.assertRaises(ValueError, SklearnDataGenerator, name=name)

    def test_call(self):
        for name in self.names:
            if name in supported_sklearn_datasets:
                data = SklearnDataGenerator(name=name, **self.kwarg_dict.get(name, {}))()
                self.assertIsInstance(data, list)
                self.assertIsInstance(data[0], np.ndarray)
                self.assertIsInstance(data[1], np.ndarray)
                self.assertEqual(data[0].shape[0], data[1].shape[0])
                self.assertEqual(len(data), 2)
            else:
                self.assertRaises(ValueError, SklearnDataGenerator, name=name)
                
    def test_hash(self):
        for name in self.names:
            if name in supported_sklearn_datasets:
                data = SklearnDataGenerator(name=name, **self.kwarg_dict.get(name, {}))
                old_hash = hash(data)
                self.assertIsInstance(old_hash, int)
                new_data = SklearnDataGenerator(name=name, **self.kwarg_dict.get(name, {}))
                new_hash = hash(new_data)
                self.assertEqual(old_hash, new_hash)
            else:
                self.assertRaises(ValueError, SklearnDataGenerator, name=name)

    def tearDown(self) -> None:
        pass


class testTorchDataGenerator(unittest.TestCase):
    def setUp(self):
        self.names = ["torch_mnist", "torch_cifar", "sasdaf0235"]

    def test_init(self):
        for name in self.names:
            if name in supported_torch_datasets:
                data = TorchDataGenerator(name=name)
                self.assertTrue(isinstance(data, TorchDataGenerator))
            else:
                self.assertRaises(ValueError, TorchDataGenerator, name=name)

    def test_hash(self):
        for name in self.names:
            if name in supported_torch_datasets:
                data = TorchDataGenerator(name=name)
                old_hash = hash(data)
                self.assertIsInstance(old_hash, int)
                new_data = TorchDataGenerator(name=name)
                new_hash = hash(new_data)
                self.assertEqual(old_hash, new_hash)
            else:
                self.assertRaises(ValueError, TorchDataGenerator, name=name)

    def test_call(self):
        for name in self.names:
            if name in supported_torch_datasets:
                data = TorchDataGenerator(name=name)()
                self.assertIsInstance(data, list)
                self.assertIsInstance(data[0], np.ndarray)
                self.assertIsInstance(data[1], np.ndarray)
                self.assertEqual(data[0].shape[0], data[1].shape[0])
                self.assertEqual(len(data), 2)
            else:
                self.assertRaises(ValueError, TorchDataGenerator, name=name)


class testKerasDataGenerator(unittest.TestCase):
    def setUp(self):
        self.names = ["mnist", "cifar", "sasdaf0235"]
    
    def test_init(self):
        for name in self.names:
            if name in supported_tf_datasets or name in supported_keras_datasets:
                data = KerasDataGenerator(name=name)
                self.assertTrue(isinstance(data, KerasDataGenerator))
            else:
                self.assertRaises(ValueError, KerasDataGenerator, name=name)
    
    def test_hash(self):
        for name in self.names:
            if name in supported_tf_datasets or name in supported_keras_datasets:
                data = KerasDataGenerator(name=name)
                old_hash = hash(data)
                self.assertIsInstance(old_hash, int)
                new_data = KerasDataGenerator(name=name)
                new_hash = hash(new_data)
                self.assertEqual(old_hash, new_hash)
            else:
                self.assertRaises(ValueError, KerasDataGenerator, name=name)
    
    def test_call(self):
        for name in self.names:
            if name in supported_tf_datasets or name in supported_keras_datasets:
                data = KerasDataGenerator(name=name)()
                self.assertIsInstance(data, list)
                self.assertIsInstance(data[0], np.ndarray)
                self.assertIsInstance(data[1], np.ndarray)
                self.assertEqual(data[0].shape[0], data[1].shape[0])
                self.assertEqual(len(data), 2)
            else:
                self.assertRaises(ValueError, KerasDataGenerator, name=name)
                