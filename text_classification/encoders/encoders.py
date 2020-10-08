from abc import abstractmethod


class BaseEncoder:
    @abstractmethod
    def __call__(self):
        return

    @property
    @abstractmethod
    def attributes(self):
        return {}
