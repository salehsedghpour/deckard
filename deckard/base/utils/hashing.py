from hashlib import md5
from collections import OrderedDict
from typing import NamedTuple, Union
from dataclasses import asdict, is_dataclass
from omegaconf import OmegaConf, SCMode, DictConfig, ListConfig
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


def to_dict(obj: Union[dict, DictConfig, ListConfig]) -> dict:
    """
    Recursively convert a DictConfig to a dict.
    Args:
        obj (Union[dict, DictConfig, ListConfig]): DictConfig to convert
    Returns:
        dict: Converted dict
    """
    new = {}
    if isinstance(obj, dict):
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    elif is_dataclass(obj):
        obj = asdict(obj)
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    elif isinstance(obj, (DictConfig, ListConfig)):
        obj = OmegaConf.to_container(
                obj, resolve=True
            ),
        sorted_keys = range(len(obj))
    else:
        raise ValueError(
            f"obj must be a Dict, namedtuple or OrderedDict. It is {type(obj)}",
        )
    for key in sorted_keys:
        if obj[key] is None:
            continue
        elif isinstance(obj[key], (str, float, int, bool, tuple, list)):
            new[key] = obj[key]
        elif is_dataclass(obj[key]):
            new[key] = asdict(obj[key])
        elif isinstance(obj[key], (DictConfig)):
            new[key] = to_dict(obj[key])
        elif isinstance(obj[key], (ListConfig)):
            new[key] = OmegaConf.to_container(obj[key], resolve=True)
        elif isinstance(obj[key], (dict)):
            new[key] = to_dict(obj[key])
        else:
            raise ValueError(
                f"obj[{key}] must be a Dict, namedtuple or OrderedDict. It is {type(obj[key])}",
            )
    return new


def my_hash(obj: Union[dict]) -> str:
    """
    Calculate the md5 hash of a sorted dict encoded as a 'utf-8' in hexademical.
    Args:
        obj (Union[dict]): Dict to hash
    Returns:
        str: Hash as utf-8 hexademical
    """
    return md5(str(to_dict(obj)).encode("utf-8")).hexdigest()
