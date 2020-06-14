import json
import torch
from torch.utils import data
import logging

class Config():
    
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

    def init_obj(self, name, module, *args, **kwargs):

        module_name = self.dict[name]['type']
        module_args = dict(self.dict[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

def togpu(batch, device):

    newbatch = []
    for item in batch:
        if isinstance(item, torch.Tensor):
            newbatch.append(item.to(device))
        else:
            newbatch.append(item)
    
    return tuple(newbatch)

def setdevice():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device is set to: {}".format(device))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

