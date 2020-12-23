import numpy as np
import random

import torch
from torch.utils.data import Subset, random_split

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

def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False

def load_data(path):
    """
    Load a check point.
    """
    print("-> loading data '{}'".format(path))
    # https://discuss.pytorch.org/t/out-of-memory-error-when-resume-training-even-though-my-gpu-is-empty/30757
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint

def resume(checkpoint, model, optimizer, scheduler):
    """
    return: model, optimizer, scheduler
    """
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return model, optimizer, scheduler

def t2n(torch_tensor):
    """
    Convert torch.tensor to np.ndarray.
    """
    return torch_tensor.cpu().detach().numpy()


