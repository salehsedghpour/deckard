import unittest
from pathlib import Path
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig, ListConfig
from deckard.base.utils import to_dict, my_hash
import os

this_dir = Path(os.path.realpath(__file__)).parent.resolve().as_posix()


@dataclass
class testClass:
    C: int = 1

# @dataclass
# class testClass2:
#     data : testClass = testClass(C = 1)

class testHashing(unittest.TestCase):
    param_dict: dict = {"C": 1}
    ordered_dict: OrderedDict = OrderedDict({"C": 1})
    named_tuple: namedtuple = namedtuple("named_tuple", ["C"])(1)
    data_class: dataclass = testClass()
    dict_config: DictConfig = OmegaConf.create({"C": 1})
    list_config: ListConfig = OmegaConf.create([{"C": 1}])
    tuple_config: tuple = ("C", 1)
    configs : list = ["ordered_dict", "data_class", "named_tuple", "dict_config", "list_config", "tuple_config"]
    
    
    def test_to_dict(self):
        old_dict = to_dict(self.param_dict)
        self.assertIsInstance(old_dict, dict)
        for thing in self.configs:
            try:
                new_dict = to_dict(getattr(self, thing))
                self.assertIsInstance(new_dict, dict)
                self.assertDictEqual(old_dict, new_dict)
            except:
                print("*"*80)
                print(f"Failed on {thing}")
                print("*"*80)
                raise

    def test_my_hash(self):
        old_hash = my_hash(self.param_dict)
        self.assertIsInstance(old_hash, str)
        for thing in self.configs:
            new_hash = my_hash(getattr(self, thing))
            self.assertIsInstance(new_hash, str)
            self.assertEqual(old_hash, new_hash)

class testDictHashing(testHashing):
    param_dict = {"data": {"C": 1}}
    ordered_dict = OrderedDict({"data": {"C": 1}})
    named_tuple = namedtuple("named_tuple", ["data"])({"C": 1})
    dict_config = OmegaConf.create({"data": {"C": 1}})
    list_config = OmegaConf.create([{"data": {"C": 1}}])
    list_config_config = OmegaConf.create({"data" : OmegaConf.create({"C": 1})})
    tuple_config = ("data", {"C": 1})
    configs = ["ordered_dict",  "named_tuple", "dict_config", "list_config", "list_config_config", "tuple_config"]
    
class testListHashing(testHashing):
    param_dict = {"data": [{"A": 1}, {"B": 2}]}
    ordered_dict = OrderedDict({"data": [{"A": 1}, {"B": 2}]})
    named_tuple = namedtuple("named_tuple", ["data"])([{"A": 1}, {"B": 2}])
    dict_config = OmegaConf.create({"data": [{"A": 1}, {"B": 2}]})
    list_config = OmegaConf.create([{"data": [{"A": 1}, {"B": 2}]}])
    tuple_config = ("data", [{"A": 1}, {"B": 2}])
    list_config_config = OmegaConf.create({"data" : OmegaConf.create([{"A": 1}, {"B": 2}])})
    configs = ["ordered_dict",  "named_tuple", "dict_config", "list_config", "tuple_config", "list_config_config"]


class testNestedHashing(testHashing):
    param_dict = OmegaConf.create({"data" : OmegaConf.create([{"A": 1}, {"B": 2}]), "model" : {"C": 1}})
    ordered_dict = OrderedDict({"data": [{"A": 1}, {"B": 2}], "model" : {"C": 1}})
    configs = ["ordered_dict", "param_dict"]
    
    