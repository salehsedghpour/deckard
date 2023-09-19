import unittest
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree
import os
import numpy as np
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

from deckard.base.data import Data

this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()


class testSklearnData(unittest.TestCase):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "classification.yaml"
    data_type = ".pkl"
    data_file = "data"
    test_labels_file = None
    train_labels_file = None

    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(),
            version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.data = instantiate(config=self.cfg)
        self.directory = mkdtemp()

    def test_init(self):
        self.assertTrue(isinstance(self.data, Data))

    def test_call(self):
        filename = Path(self.directory, self.data_file + self.data_type).as_posix()
        if self.train_labels_file is not None:
            train_labels_file = Path(self.directory, self.train_labels_file).as_posix()
        else:
            train_labels_file = self.train_labels_file
        if self.test_labels_file is not None:
            test_labels_file = Path(self.directory, self.test_labels_file).as_posix()
        else:
            test_labels_file = self.test_labels_file
        X_train, X_test, y_train, y_test = self.data(data_file=filename, train_labels_file=train_labels_file, test_labels_file=test_labels_file)
        self.assertIsInstance(X_train, (np.ndarray, list))
        self.assertIsInstance(X_test, (np.ndarray, list))
        self.assertIsInstance(y_train, (np.ndarray, list))
        self.assertIsInstance(y_test, (np.ndarray, list))
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertTrue(Path(filename).exists())

    def test_hash(self):
        old_hash = hash(self.data)
        self.assertIsInstance(old_hash, int)
        new_cfg = self.cfg
        new_data = instantiate(config=new_cfg)
        new_hash = hash(new_data)
        self.assertEqual(old_hash, new_hash)
        new_data()
        hash_after_call = hash(new_data)
        self.assertEqual(old_hash, hash_after_call)

    def test_initialize(self):
        data = self.data.initialize()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 4)

    def test_load(self):
        data_file = Path(self.directory, self.data_file + self.data_type).as_posix()
        _ = self.data(data_file=data_file)
        data = self.data.load(data_file)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 4)

    def test_save(self):
        data_file = Path(self.directory, self.data_file + self.data_type).as_posix()
        data = self.data(data_file)
        self.assertTrue(Path(data_file).exists())
        self.data.save(data, data_file)
        self.assertTrue(Path(data_file).exists())

    def tearDown(self) -> None:
        rmtree(self.directory)

class testLoadTwice(testSklearnData):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "titanic.yaml"
    data_type = ".pkl"
    data_file = "data"
    
    def test_load(self):
        data_file = Path(self.directory, self.data_file + self.data_type).as_posix()
        data = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]
        _ = self.data.save(data, data_file)
        new_hash = hash(self.data)
        self.assertTrue(Path(data_file).exists())
        _ = self.data(data_file = data_file)
        old_hash = hash(self.data)
        self.assertEqual(old_hash, new_hash)


class testKerasData(testSklearnData):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "keras_mnist.yaml"
    data_type = ".pkl"
    data_file = "data"


class testTorchData(testSklearnData):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "torch_mnist.yaml"
    data_type = ".pkl"
    data_file = "data"


class testTensorflowData(testSklearnData):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "tensorflow_mnist.yaml"
    data_type = ".pkl"
    data_file = "data"
    
class testCSVData(testSklearnData):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "titanic.yaml"
    data_type = ".csv"
    data_file = "data"

class testJSONData(testSklearnData):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "titanic.yaml"
    data_type = ".json"
    data_file = "data"

class testExcelData(testSklearnData):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "titanic_xlsx.yaml"
    data_type = ".json"
    data_file = "data"

class testTrainLabels(testSklearnData):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "titanic.yaml"
    data_type = ".json"
    data_file = "data"
    train_labels_file = "train_labels.pkl"

class testTestLabels(testSklearnData):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "titanic.yaml"
    data_type = ".json"
    data_file = "data"
    test_labels_file = "test_labels.pkl"




class testInitValueError(unittest.TestCase):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "classification.yaml"
    data_type = ".pkl"
    data_file = "data"
    test_labels_file = None
    train_labels_file = None

    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(), version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.data = instantiate(config=self.cfg)
        self.directory = mkdtemp()

    def test_init(self):
        self.assertRaises(ValueError, self.data, data_file="foo")

class testDataTypeError(unittest.TestCase):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "classification.yaml"
    data_type = ".pkl"
    data_file = "data"
    test_labels_file = None
    train_labels_file = None

    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(), version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.data = instantiate(config=self.cfg)
        self.directory = mkdtemp()
        self.object = unittest.TestCase()
        

    def test_save(self):
        self.assertRaises(TypeError, self.data.save, data=self.object, filename="foo.json")
        
class testDataTypeListError(unittest.TestCase):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "classification.yaml"
    data_type = ".pkl"
    data_file = "data"
    test_labels_file = None
    train_labels_file = None

    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(), version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.data = instantiate(config=self.cfg)
        self.directory = mkdtemp()
        self.object = [unittest.TestCase()]
        

    def test_save(self):
        self.assertRaises(TypeError, self.data.save, data=self.object, filename="foo.json")

class testSaveNumpy(unittest.TestCase):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "classification.yaml"
    data_type = ".pkl"
    data_file = "data"
    test_labels_file = None
    train_labels_file = None

    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(), version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.data = instantiate(config=self.cfg)
        self.directory = mkdtemp()
        self.object = np.array(unittest.TestCase())
        

    def test_save(self):
        self.assertRaises(TypeError, self.data.save, data=self.object, filename="foo.json")


class testLoadError(unittest.TestCase):
    config_dir = Path(this_dir, "../../conf/data").resolve().as_posix()
    config_file = "titanic.yaml"
    data_type = ".dill"
    data_file = "data"
    
    def setUp(self):
        with initialize_config_dir(
            config_dir=Path(self.config_dir).resolve().as_posix(), version_base="1.3",
        ):
            cfg = compose(config_name=self.config_file)
        self.cfg = cfg
        self.data = instantiate(config=self.cfg)
        self.directory = mkdtemp()
    
    def test_load(self):
        data_file = Path(self.directory, self.data_file + self.data_type).as_posix()
        self.assertRaises(ValueError, self.data.load, data_file)        