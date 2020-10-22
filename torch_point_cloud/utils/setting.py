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

def get_git_commit_hash():
    cmd = "git rev-parse --short HEAD"
    hash_code = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    return hash_code

def download_and_unzip(www, output_path):
    zip_file = os.path.basename(www)
    if not os.path.exists(zip_file):
        os.system('wget %s --no-check-certificate' % (www))
    folder_name = zip_file[:-4]
    make_folders(folder_name)
    os.system("unzip %s -d %s" % ('"'+zip_file+'"', "'"+folder_name+"'"))
    os.system('mv %s %s' % ('"'+folder_name+'"', '"'+output_path+'"'))
    os.system('rm %s' % ('"'+zip_file+'"'))

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


