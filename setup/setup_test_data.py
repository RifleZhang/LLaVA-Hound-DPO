import os, sys, time, os.path as osp
import math
import numpy as np
import glob
import json
from collections import defaultdict 
from tqdm.autonotebook import tqdm

import pandas as pd
import fire

import pathlib
from typing import Any, Dict, List, Optional, Union

from hf_utils import download_file, download_repo

def main(repo_id, local_dir, repo_type):
    download_repo(repo_id=repo_id,
                  local_dir=local_dir,
                  repo_type=repo_type)
    
if __name__ == "__main__":
    fire.Fire(main)