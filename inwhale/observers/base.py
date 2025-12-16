from abc import ABC, abstractmethod


class Observer(ABC):

    @abstractmethod
    def observe(self, x):
        pass

    @abstractmethod
    def get_range(self):
        pass
