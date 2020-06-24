import copy
from typing import Dict, Callable, Any, Optional
from inspect import signature
from functools import wraps
from abc import ABC, abstractmethod
import os, json

"""
class EnvironmentObject(Protocol):

    @abstractmethod
    def env_restore(self, env: Environment):
        raise NotImplementedError

    @abstractmethod
    def env_save(self, env: Environment):
        raise NotImplementedError
"""

# forward declare
class Algorithm: pass

class Environment: pass

def isEnvObject(x: Any):
    if hasattr(x, "env_restore") and hasattr(x, "env_save"):
        return callable(getattr(x, "env_restore")) and callable(getattr(x, "env_save"))
    return False

class Environment:

    def __init__(self, path, model, data, algo: Algorithm, state: Optional[Dict]=None, callbacks: Optional[Dict]=None):
        self.model = model
        self.data = data
        self.state = {}
        self.state.update({'model': self.model, 'data': self.data, 'algo': algo, 'environment_path': path})
        if state: self.state.update(state)
        self.callbacks = callbacks if callbacks else {}
    
    def set_callbacks(self, callbacks: Dict[str, Callable[[Environment], Any]]):
        self.callbacks.update(callbacks)
    
    def get_state(self):
        return self.state
    
    def set_state(self, s: Dict[str, Any]):
        self.state.update(s)
    
    def get_callbacks(self):
        return self.callbacks
    
    def run_callback(self, callback: str, context: Optional[dict]=None):
        assert callback in self.callbacks, "callback with name {:s} not set".format(callback)
        if context: return self.callbacks[callback](self, context)
        return self.callbacks[callback](self)
    
    # convenience functions
    def run_train_cb(self):
        return self.run_callback('train')
    
    def run_predict_cb(self):
        return self.run_callback('predict')
    
    def run_eval_cb(self):
        return self.run_callback('evaluate')
    
    def env_restore(self):
        prev_state = self.state.copy() # shallow copy
        for k, v in prev_state.items():
            # if isinstance(v, EnvironmentObject): v.env_restore(self)
            if isEnvObject(v): v.env_restore(self)
        if not os.path.exists(os.path.join(self.state['environment_path'], 'state.json')):
            print("state not updated: cannot find state.json")
            return
        with open(os.path.join(self.state['environment_path'], 'state.json'), 'r') as fp:
            S = json.load(fp)
            self.state.update(S)

    def env_save(self):
        """
        [
            checks if there is a EnvObject implementation in the top level, otherwise
            try to json-ize it, if fails, skip
        ]
        """

        S = {}
        prev_state = self.state.copy() # shallow copy
        for k, v in prev_state.items():
            # if isinstance(v, EnvironmentObject): v.env_save(self)
            if isEnvObject(v): v.env_save(self)
            else:
                try:
                    json.dumps({k: v}) 
                    S[k] = v
                except:
                    print("warning: failed to serialize object {}: {}". format(k, v))
                    pass
        with open(os.path.join(self.state['environment_path'], 'state.json'), 'w') as fp:
            json.dump(S, fp)

def map_state_to_props(func: Callable):
    
    @wraps(func)
    def env_func(env: Environment, context: Optional[dict]=None):
    
        state = env.get_state()
        sig = signature(func)
        kwargs = {}
    
        # fetch the corresponding state
        for k in sig.parameters.keys(): 
            if k in state: kwargs[k] = state[k]
    
        # fetch the corresponding context, keys in context override state
        if context:
            for k in sig.parameters.keys(): 
                if k in state: kwargs[k] = context[k]
    
        # call function with compiled kwargs
        return func(**kwargs)
    
    return env_func

def update_env_with_ret(func: Callable):
    
    @wraps(func)
    def update_func(env: Environment, context: Optional[dict]=None):
        if context: ret = func(env, context)
        else: ret = func(env)
        # update state with return values
        env.set_state(ret)
        return ret
    
    return update_func

def run_algo_fit(env: Environment):
    algo = env.get_state()['algo']
    algo.fit(env)

class Algorithm(ABC):

    def __init__(self, config):
        self.config = config
        pass

    @abstractmethod
    def fit(self, env: Environment):
        raise NotImplementedError

    @abstractmethod
    def env_restore(self, env: Environment):
        raise NotImplementedError

    @abstractmethod
    def env_save(self, env: Environment):
        raise NotImplementedError
