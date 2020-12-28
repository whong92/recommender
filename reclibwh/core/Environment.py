import copy
from typing import Dict, Callable, Any, Optional, List
from abc import ABC, abstractmethod

# forward declare
class Environment: pass

class EvalProto(ABC):

    @abstractmethod
    def evaluate(self):
        pass

class RecAlgo(ABC):

    @abstractmethod
    def recommend(self, user):
        pass

class Algorithm(ABC):

    @abstractmethod
    def fit(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, **kwargs):
        raise NotImplementedError


class UpdateAlgo(ABC):

    @abstractmethod
    def update_user(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def make_update_data(self, data):
        raise NotImplementedError

class UpdateAlgoStub(UpdateAlgo):

    def update_user(self, **kwargs):
        print("stub update called, doing nothing ... ")


class Environment:

    def __init__(
            self, path, model, data, state: Optional[Dict]=None
    ):
        self.__path = path
        self.__model = model
        self.__data = data
        self.__state = {'environment_path': path, 'model': model, 'data': data}
        if state: self.__state.update(state)

    def environment_add_object(self, obj, k):
        obj.set_env(self)
        self.set_state({k: obj})

    def __getitem__(self, k):
        return self.__state[k]

    def get_state(self):
        return self.__state
    
    def set_state(self, s: Dict[str, Any]):
        self.__state.update(s)
