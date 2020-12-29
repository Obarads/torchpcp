import sys,os
import yaml

class YAMLLogWriter():
    """
    Logger with a yaml file.

    How to use
    ----------
    root = "log_dir/log.yaml"
    for i in range(5):
        logger = LogWriter(root)
        log_dict = {"log1": i, "log2": -i}
        logger.update(log_dict)

    If you want to load LogWriter yaml file....
    root = "log_dir/log.yaml"
    new_root = "log_dir/new_log.yaml"
    logger = LogWriter(new_root, root)

    or

    logger = LogWriter(new_root)
    logger.load_log(root)
    """
    def __init__(self, root:str, loading_log_root:str=None):

        self.root = root
        self.status = []

        if loading_log_root is not None:
            self.load_log(loading_log_root)

    def load_log(self, loading_log_root:str):
        with open(loading_log_root, mode="r") as f:
            self.status = yaml.load(f, Loader=yaml.FullLoader)

    def update(self, log):
        self.status.append(log)
        with open(self.root, mode="w") as f:
            yaml.dump(self.status, f, indent=2)
