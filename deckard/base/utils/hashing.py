from hashlib import md5
from collections import OrderedDict
from typing import NamedTuple, Union
from dataclasses import asdict, is_dataclass
from omegaconf import DictConfig, OmegaConf, SCMode, ListConfig
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


def to_dict(obj: Union[dict, OrderedDict, NamedTuple]) -> dict:
    new = {}
    if hasattr(obj, "_asdict"):
        obj = obj._asdict()
    elif is_dataclass(obj):
        obj = deepcopy(asdict(obj))
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    if isinstance(obj, dict) and not isinstance(obj, OrderedDict):
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    elif isinstance(obj, OrderedDict):
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    elif isinstance(obj, (DictConfig, OmegaConf)):
        obj = dict(
            deepcopy(
                OmegaConf.to_container(
                    obj,
                    resolve=True,
                    structured_config_mode=SCMode.DICT,
                ),
            ),
        )
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    elif isinstance(obj, (ListConfig, list)):
        if isinstance(obj, (ListConfig)):
            obj = deepcopy(OmegaConf.to_container(obj, resolve=True))
        keys = []
        values = []
        for entry in obj:
            if isinstance(entry, dict):
                keys.extend(list(entry.keys()))
                values.extend(list(entry.values()))
            else: # pragma: no cover
                raise ValueError(f"entry {entry} is not a key:value pair")
        sorted_keys = keys
        obj = OrderedDict(zip(keys, values))
    elif isinstance(obj, str):
        obj = obj
        sorted_keys = []
    # elif isinstance(obj, type(None)):
    #     obj = None
    #     sorted_keys = []
    elif isinstance(obj, (tuple)):
        sorted_keys = [obj[0]]
        value = obj[1]
        key = obj[0]
        obj = {key : value}
    else: # pragma: no cover
        raise ValueError(
            f"obj must be a Dict, namedtuple or OrderedDict. It is {type(obj)}",
        )
    for key in sorted_keys:
        if obj[key] is None:
            continue
        if isinstance(obj[key], (list, tuple)):
            new[key] = [to_dict(x) for x in obj[key]] #
        elif isinstance(obj[key], (str, float, int, bool, tuple, list)):
            new[key] = obj[key]
        elif isinstance(obj[key], (DictConfig)):
            new[key] = to_dict(obj[key]) #
        elif isinstance(obj[key], (ListConfig)):
            new[key] = to_dict(obj[key]) #
        elif isinstance(obj[key], (dict)):
            new[key] = to_dict(obj[key])
        elif is_dataclass(obj[key]):
            new[key] = asdict(obj[key])
        else: # pragma: no cover
           raise TypeError(f"obj[{key}] is of type {type(obj[key])}")
    return new


def my_hash(obj: Union[dict, OrderedDict, NamedTuple]) -> str:
    return md5(str(to_dict(obj)).encode("utf-8")).hexdigest()
