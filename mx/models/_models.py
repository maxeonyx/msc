import abc
from dataclasses import dataclass


@dataclass
class MxModel(abc.ABC):
    
    @abc.abstractmethod
    def call(self, inputs, training: bool):
        pass
