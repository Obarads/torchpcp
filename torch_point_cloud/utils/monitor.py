import os
import omegaconf
import time
from decimal import Decimal

###
### log service
###

def dict2tensorboard(log_dict, writer, step):
    for key in log_dict:
        if isinstance(log_dict[key], list):
            _log = log_dict[key][0]
            _step = log_dict[key][1]
        else:
            _log = log_dict[key]
            _step = step
        writer.add_scalar(key, _log, _step)

def dict2wandb(log_dict, writer, step):
    writer.log(log_dict, step=step)

def dict2logger(log_dict, writer, step, log_service):
    # assert log_service in ["wandb","tensorboardX"]
    if log_service == "wandb":
        dict2wandb(log_dict, writer, step)
    elif log_service == "tensorboardX":
        dict2tensorboard(log_dict, writer, step)
    else:
        raise NotImplementedError("Unknown log_service: {}".format(log_service))

###
### configs with omegaconfig
###

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

###
### Wathcer
###

class TimeWatcher:
    def __init__(self):
        self.reset()
    def reset(self):
        r"""
        Reset params.
        """
        self.__start_time = 0
        self.__previous_time = 0
    def start(self):
        r"""
        Start timer and return start time.
        This function do time reset when called.
        """
        self.reset()
        now_time = time.time()
        self.__start_time = now_time
        self.__previous_time = now_time
        return now_time
    def lap_and_split(self):
        r"""
        Return lap time and split time.
        """
        now_time = time.time()
        split_time = now_time - self.__start_time
        lap_time = now_time - self.__previous_time
        self.__previous_time = now_time
        return lap_time, split_time

def timecheck(start=None, publisher="time"):
    if start is not None:
        print("{}: {}s".format(publisher, Decimal(time.time()-start)))
    return time.time()


