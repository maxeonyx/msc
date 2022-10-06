

import abc
from dataclasses import dataclass
from mx.models import MxModel

@dataclass
class Embedding(abc.ABC):

    @abc.abstractmethod
    def to(self, model: MxModel):
        pass
