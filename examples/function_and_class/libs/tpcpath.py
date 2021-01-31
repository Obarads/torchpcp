# add path of torchpcp
# If you installed torchpcp using setup.py, don't need to import this module.
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../../../"))) # for package path

