import os
import pathlib
import subprocess

def is_absolute(path:str)->bool:
    path_pl = pathlib.Path(path)
    return path_pl.is_absolute()

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

def get_git_commit_hash():
    cmd = "git rev-parse --short HEAD"
    hash_code = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    return hash_code

