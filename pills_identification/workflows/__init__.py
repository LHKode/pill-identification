from abc import ABC, abstractmethod
from typing import List


class IWorkflow(ABC):
    @abstractmethod
    def __call__(self, **kwargs):
        return kwargs


class IWorkflowStep(ABC):
    @abstractmethod
    def __call__(self, **kwargs):
        return kwargs
