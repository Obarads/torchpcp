import os, sys

import warnings
from os.path import join as opj 
import omegaconf
import numpy as np
import random
import pathlib
import subprocess

import torch
from torch.utils.data import Subset, random_split

### load cfg
def get_configs(yaml_file):
    """
    get command line args and configs

    Parameters
    ----------
    yaml_file: str
        yaml file path for configs
    
    Returns
    -------
    cfg: omegaconf.DictConfig
        configs with a yaml file and CLI args (CLI args overwrite settings on 
        a yaml file.)
    cfg_yaml: omegaconf.DictConfig
        settings on a yaml file.
    cfg_cli: omegaconf.DictConfig
        CLI args.
    """

    cfg_yaml = omegaconf.OmegaConf.load(yaml_file)
    cfg_cli = omegaconf.OmegaConf.from_cli()
    cfg = omegaconf.OmegaConf.merge(cfg_yaml, cfg_cli)

    return cfg, cfg_yaml, cfg_cli

def save_configs(yaml_file, cfg):
    """
    save cfg to yaml file.

    Parameters
    ----------
    yaml_file: str
        yaml file path
    cfg: omegaconf.DictConfig
        cfg
    """

    omegaconf.OmegaConf.save(config=cfg, f=yaml_file)

def make_folders(odir):
    """
    Create folders.

    Parameters
    ----------
    odir: str
        folder path
    """
    if not os.path.exists(odir):
        os.makedirs(odir)

def is_absolute(path:str)->bool:
    path_pl = pathlib.Path(path)
    return path_pl.is_absolute()

def fix_path_in_configs(base_path, cfg, dictkey_list):
    """
    Join to base_path relative path

    Parameters
    ----------
    base_path : str
        a path to join to head of relative path
    cfg : omega.DictConfig
        configs
    dictkey_list : list
        key list to fix relative paths

    Return
    ------
    modified_cfg
        config with modified paths

    Examples
    --------
    import omegaconf

    dict_cfg = {
        "o1" : 0,
        "p1" : {
            "p2" : "/path/to/a",
            "p3" : "to/b"
        },
        "q1" : {
            "q2" : "to/c",
            "q3" : "to/d"
        },
        "r1" : {"r1" : "to/e"}
    }
    cfg = omegaconf.OmegaConf.create(dict_cfg)
    dictkey_list = [["p1","p2"],["q1","q2"]]
    base_path = "/other/path"
    fixed_cfg = fix_path_in_configs(base_path, cfg, dictkey_list)
    print(cfg)
    print(fixed_cfg)
    """

    def input_value(i, key_list, bq, _cfg):
        if len(key_list) > i:
            d = {key_list[i] : input_value(i+1, key_list, bq, _cfg[key_list[i]])}
        else:
            d = os.path.join(bq, _cfg) # If _cfg is absolute path, d = _cfg.
        return d

    for keys in dictkey_list:
        da = input_value(0, keys, base_path, cfg)
        da = omegaconf.OmegaConf.create(da)
        cfg = omegaconf.OmegaConf.merge(cfg, da)

    return cfg

def get_git_commit_hash():
    cmd = "git rev-parse --short HEAD"
    hash_code = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    return hash_code

class PytorchTools:
    def __init__(self):
        print("This class is for staticmethod.")
    
    @staticmethod
    def create_subset(dataset, subset):
        """
        Get dataset subset.
        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            torch dataset
        subset: list or int
            Data index or number of data used in a subset
        
        Returns
        -------
        subset of dataset:
            subset of torch dataset
        subset_number_list:
            dataset index list used in subset
        """
        if type(subset) is int:
            subset_number_list = np.random.randint(0,len(dataset)-1,subset)
        elif type(subset) is list:
            subset_number_list = subset
        else:
            NotImplementedError()
        return Subset(dataset,subset_number_list), subset_number_list
    
    @staticmethod
    def split_dataset(dataset, ratio:float, seed=0):
        """
        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            dataset
        ratio: float
            ratio for number of data [0,1]

        Examples
        --------
        dataset = Dataset()
        new_dataset = (dataset, 0.75)
        train_dataset = new_dataset[0] # 0.75
        test_dataset = new_dataset[1] # 0.25
        """
        dataset_lenght = len(dataset)
        lenght_1 = int(dataset_lenght*ratio)
        lenght_2 = dataset_lenght - lenght_1
        dataset = random_split(dataset, [lenght_1, lenght_2])
        return dataset

    @staticmethod
    def set_seed(seed, cuda=True, consistency=False):
        """
        Set seeds in all frameworks (random, numpy, torch).
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda: 
            torch.cuda.manual_seed(seed)
        if cuda and torch.cuda.is_available() and not consistency:
            torch.backends.cudnn.enabled = True # use cuDNN
        else:
            torch.backends.cudnn.enabled = False

    @staticmethod
    def select_device(device_name):
        """
        This function correct orthographic variants for device selection.
        Parameters
        ----------
        device_name: {"cpu","gpu","cuda","N",N} (N=-1 or available gpu number)
            device name or number used on torch

        Returns
        -------
        device: str
            If device_name is {"cpu","-1",-1}, device is "cpu".
            If device_name is {"cuda","N",N}, device is "cude".
        """
        if type(device_name) is str:
            if device_name in ["cpu", "-1"]:
                device = "cpu"
            elif device_name in ["cuda", "gpu","0"]:
                device = "cuda"
            elif device_name in ["tpu"]:
                raise NotImplementedError()
            else:
                raise NotImplementedError("1 Unknow device: {}".format(device_name))
        elif type(device_name) is int:
            if device_name < 0:
                device = "cpu"
            elif device_name >= 0:
                device = "cuda"
            else:
                raise NotImplementedError("2 Unknow device: {}".format(device_name))
        else:
            raise NotImplementedError("0 Unknow device: {}".format(device_name))
        return device

    @staticmethod
    def fix_model(model):
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def load_data(path):
        """
        Load a check point.
        """
        print("-> loading data '{}'".format(path))
        # https://discuss.pytorch.org/t/out-of-memory-error-when-resume-training-even-though-my-gpu-is-empty/30757
        checkpoint = torch.load(path, map_location='cpu')
        return checkpoint

    @staticmethod
    def resume(checkpoint, model, optimizer, scheduler):
        """
        return: model, optimizer, scheduler
        """
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint["scheduler"])
        return model, optimizer, scheduler

    @staticmethod
    def t2n(torch_tensor):
        return torch_tensor.cpu().detach().numpy()


