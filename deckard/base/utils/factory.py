import logging
from itertools import product
from importlib import import_module

__all__ = ["flatten_dict", "unflatten_dict", "make_grid", "factory"]
logger = logging.getLogger(__name__)


def flatten_dict(dictionary: dict, separator: str = ".", prefix: str = ""):
    """
    Args:
        dictionary (dict): Dictionary to flatten
        separator (str, optional): Separator to use when flattening. Defaults to ".".
        prefix (str, optional): Prefix to use when flattening. Defaults to "".
    Returns:
        dict: Flattened dictionary
    """
    stack = [(dictionary, prefix)]
    flat_dict = {}
    while stack:
        cur_dict, cur_prefix = stack.pop()
        for key, val in cur_dict.items():
            new_key = cur_prefix + separator + key if cur_prefix else key
            if isinstance(val, dict):
                logger.debug(f"Flattening {val} into {new_key}")
                stack.append((val, new_key))
            else:
                logger.debug(f"Adding {val} to {new_key}")
                flat_dict[new_key] = val
    return flat_dict


def factory(module_class_string, *args, super_cls: str = None, **kwargs) -> object:
    """
    Factory function for instantiating classes.
    Args:
        module_class_string (str): Module and class name to instantiate
        *args: Arguments to pass to the class
        super_cls (type, optional): Super class to check inheritance. Defaults to None.
        **kwargs: Keyword arguments to pass to the class
    """
    try:
        module_name, class_name = module_class_string.rsplit(".", 1)
    except Exception as e:  # noqa E722
        raise ValueError(f"Invalid module_class_string: {module_class_string}")
    module = import_module(module_name)
    assert hasattr(module, class_name), "class {} is not in {}".format(
        class_name,
        module_name,
    )
    logger.debug("reading class {} from module {}".format(class_name, module_name))
    cls = getattr(module, class_name)
    if super_cls is not None:
        assert issubclass(cls, super_cls), "class {} should inherit from {}".format(
            class_name,
            super_cls.__name__,
        )
    logger.debug("initialising {} with params {}".format(class_name, kwargs))
    try:
        obj = cls(*args, **kwargs)
    except Exception as e:
        logger.info(f"Error with args {args} and kwargs {kwargs}: {e}")
        raise e
    return obj


def unflatten_dict(dictionary: dict, separator: str = ".") -> dict:
    """Unflattens a dictionary into a nested dictionary.
    Args:
        dictionary (dict): Dictionary to unflatten
        separator (str, optional): Separator to use when unflattening. Defaults to ".".
    Returns:
        dict: Unflattened dictionary
    """
    result = {}
    for key, val in dictionary.items():
        parts = key.split(separator)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        logger.debug(f"Adding {val} to {parts[-1]}")
        d[parts[-1]] = val
    return result


def make_grid(dict_list: list) -> list:
    """
    Makes a grid of parameters from a list of dictionaries.
    Args:
        dict_list (list): List of dictionaries
    Returns:
        list: List of dictionaries
    """
    big = []
    assert isinstance(dict_list, list), f"dictionary is {type(dictionary)}"
    for dictionary in dict_list:
        for k, v in dictionary.items():
            if isinstance(v, list):
                logger.debug(f"{v} is a list.")
                dictionary[k] = v
            else:
                logger.debug(f"{v} is type {type(v)}")
                dictionary[k] = [v]
        keys = dictionary.keys()
        combinations = product(*dictionary.values())
        big.extend(combinations)
    return [dict(zip(keys, cc)) for cc in big]
