import logging
from itertools import product
from importlib import import_module

__all__ = ["flatten_dict", "unflatten_dict", "make_grid", "factory"]
logger = logging.getLogger(__name__)


def flatten_dict(dictionary: dict, separator: str = ".", prefix: str = ""):
    """
    Flattens a dictionary into a list of dictionarieswith keys separated by the separator.
    :param dictionary: The dictionary to flatten.
    :param separator: The separator to use when flattening the dictionary.
    :param prefix: The prefix to use when flattening the dictionary.
    :return: A flattened dictionary.
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


def factory(module_class_string, *args, super_cls: type = None, **kwargs) -> object:
    """
    :param module_class_string: full name of the class to create an object of
    :param super_cls: expected super class for validity, None if bypass
    :param kwargs: parameters to pass
    :return:
    """
    try:
        module_name, class_name = module_class_string.rsplit(".", 1)
    except Exception as e:  # noqa E722
        raise ValueError(f"Invalid module_class_string: {module_class_string}")
    module = import_module(module_name)
    assert hasattr(module, class_name), "class {} is not in {}".format(
        class_name, module_name,
    )
    logger.debug("reading class {} from module {}".format(class_name, module_name))
    cls = getattr(module, class_name)
    if super_cls is not None:
        assert issubclass(cls, super_cls), "class {} should inherit from {}".format(
            class_name, super_cls.__name__,
        )
    logger.debug("initialising {} with params {}".format(class_name, kwargs))
    try:
        obj = cls(*args, **kwargs)
    except Exception as e:
        raise ValueError(f"Error with args {args} and kwargs {kwargs}: {e}")
    return obj


def unflatten_dict(dictionary: dict, separator: str = ".") -> dict:
    """Unflattens a dictionary into a nested dictionary.
    :param dictionary: The dictionary to unflatten.
    :param separator: The separator to use when unflattening the dictionary.
    :param prefix: The prefix to use when unflattening the dictionary.
    :return: An unflattened dictionary.
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
    :param dict_list: The list of dictionaries to make a grid from.
    :return: A list of dictionaries with all possible combinations of parameters.
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
