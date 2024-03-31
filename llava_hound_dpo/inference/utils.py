import os, sys, time, os.path as osp
import math
import numpy as np
import glob
import json
from collections import defaultdict 
from tqdm import tqdm
from collections import Counter

import pandas as pd
import re

import pathlib
from typing import Any, Dict, List, Optional, Union
import fire
from logzero import logger
from data_processing.utils import load_jsonl, save_jsonl, load_json, save_json, load_pickle
from data_processing.utils import parse_single_file

def get_score(response):
    try:
        # score can be float
        pattern = r"Score: (\d+\.?\d*)"
        score = re.search(pattern, response).group(1)
        return float(score)
    except:
        # Fetch all float or integer values
        pattern = r"\b\d+\.?\d*\b"
        scores = re.findall(pattern, response)
        scores = [float(score) for score in scores]

        if scores:
            # Majority voting
            score_counts = Counter(scores)
            most_common_scores = score_counts.most_common()
            highest_frequency = most_common_scores[0][1]

            # Check for ties
            candidates = [score for score, count in most_common_scores if count == highest_frequency]

            # Strategy in case of a tie or clear majority
            if len(candidates) == 1:
                # Clear majority
                return candidates[0]
            else:
                # Tie or no scores found, apply a default strategy
                # For simplicity, returning the average in case of a tie
                return sum(candidates) / len(candidates)
        else:
            # Default score if no numbers are found
            return 1