import logging
from dataclasses import dataclass, asdict
from copy import deepcopy
from sklearn.model_selection import train_test_split
from ..utils import my_hash

__all__ = ["SklearnDataSampler"]
logger = logging.getLogger(__name__)


@dataclass
class SklearnDataSampler:
    test_size: float = 0.2
    train_size: float = 0.8
    random_state: int = 0
    shuffle: bool = True
    stratify: bool = False
    time_series: bool = False

    def __init__(
        self,
        test_size=0.2,
        train_size=0.8,
        random_state=0,
        shuffle=True,
        stratify=False,
        time_series=False,
    ):
        """
        Return a sampler for sklearn data
        :param test_size: Test size (optional)
        :type test_size: float
        :param train_size: Train size (optional)
        :type train_size: float
        :param random_state: Random state (optional)
        :type random_state: int
        :param shuffle: Shuffle (optional)
        :type shuffle: bool
        :param stratify: Stratify (optional)
        :type stratify: bool
        :param time_series: Time series (optional)
        :type time_series: bool
        """
        logger.info(
            f"Instantiating {self.__class__.__name__} with params {asdict(self)}",
        )
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        self.time_series = time_series

    def __call__(self, X, y) -> list:
        """
        Return the sampled data
        :param X: Input data
        :type X: np.ndarray
        :param y: Target data
        :type y: np.ndarray
        :return: X_train, X_test, y_train, y_test
        :rtype: list
        """
        logger.info(f"Calling SklearnDataSampler with params {asdict(self)}")
        params = deepcopy(asdict(self))
        stratify = params.pop("stratify", False)
        if stratify is True:
            stratify = y
        else:
            stratify = None
        test_size = params.pop("test_size")
        train_size = params.pop("train_size")
        time_series = params.pop("time_series")
        if time_series is not True:
            X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    train_size=train_size,
                    stratify=stratify,
                    **params,
                    test_size=test_size,
                )
        else:
            if isinstance(train_size, float):
                train_size = int(train_size * len(X))
            if test_size is None:
                test_size = len(X) - train_size
            elif isinstance(test_size, float):
                test_size = int(test_size * len(X))
            X_train = X[:train_size]
            X_test = X[train_size : train_size + test_size]  # noqa E203
            y_train = y[:train_size]
            y_test = y[train_size : train_size + test_size]  # noqa E203

        return [X_train, X_test, y_train, y_test]

    def __hash__(self):
        return int(my_hash(self), 16)
