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


class errorClass:
    def __init__(self):
        self.C = 1

class testFactory(unittest.TestCase):
    param_dict: dict = {"C": 1}
    ordered_dict: OrderedDict = OrderedDict({"C": 1})
    data_class: dataclass = testClass()
    dict_config: DictConfig = OmegaConf.create({"C": 1})

    def setUp(self):
        self.things = {
                "params" : self.param_dict,
                "dataclass" : self.data_class,
                "dict config" : self.dict_config,
        }

    def test_to_dict(self):
        old_dict = to_dict(OmegaConf.create(self.param_dict))
        new_dict = to_dict(OmegaConf.create(self.things['params']))
        self.assertEqual(old_dict, new_dict)

    
    def test_complicated_config(self):
        things = {}
        things['list'] = [1,2,3]
        things['list config'] = OmegaConf.create([1,OmegaConf.create(["D","2"]),OmegaConf.create({'E':3})])
        things['None'] = None
        things['str'] = 'string'
        things['tuple'] = (1,2,3)
        things['dict'] = {'C':1, 'D':2}
        things.update(self.things)
        # things = OmegaConf.create(things)
        old_dict = to_dict(things)
        old_hash = my_hash(things)
        assert isinstance(old_dict, dict), f"old_dict is {type(old_dict)} {old_dict}"
        self.assertIsInstance(old_dict, dict)
        self.assertIsInstance(old_hash, str)
    
    def test_error_hashing(self):
        self.assertRaises(ValueError, my_hash, errorClass())
        things = {"error": errorClass()}
        self.assertRaises(ValueError, to_dict, things)
                
    def test_my_hash(self):
        old_hash = my_hash(OmegaConf.create(self.param_dict))
        new_hash = my_hash(OmegaConf.create(self.things['params']))
        self.assertEqual(old_hash, new_hash)