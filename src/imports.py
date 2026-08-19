# useful built-in modules
import os
import sys
from pathlib import Path
import warnings
import time
import datetime
from zoneinfo import ZoneInfo
import tqdm
import re
import shutil
import math
import argparse
from itertools import product
from concurrent.futures import ProcessPoolExecutor

# type annotation
from typing import Literal, Any
from collections.abc import Callable, Sequence
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# file handling
import yaml
import pickle

# numpy weeeee
import numpy

# high-level stuff
import torch
import pandas
import sklearn
from scipy import signal
from sklearn.decomposition import PCA

# snntorch, because it's stupid.
import snntorch
from snntorch import functional
from snntorch import surrogate

# visualization
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import seaborn
from snntorch import spikeplot

# some very global constants
NOW = datetime.datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%d-%H%M%S")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
NUMPY_RNG = numpy.random.default_rng(42)
TORCH_RNG = torch.manual_seed(42)

# some paths
DEFAULT_CONFIG_PATH = os.path.join(
    Path(__file__).parent.parent,
    "config.yml"
)
ROOT = Path(__file__).parent.parent