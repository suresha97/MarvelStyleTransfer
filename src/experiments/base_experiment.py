from abc import ABC, abstractmethod


class BaseExperiment(ABC):
    def __init__(self):
        return

    @abstractmethod
    def run_experiment(self):
        pass
