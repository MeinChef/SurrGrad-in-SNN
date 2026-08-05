# useful built-in modules
import os
from pathlib import Path
import warnings
import time
import datetime
import gc
import tqdm
import re
import shutil
import math
import timeit
import argparse
from itertools import product
from concurrent.futures import ProcessPoolExecutor

# type annotation
import typing
from typing import Literal
from collections.abc import Callable, Sequence
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# file handling
import yaml
import pickle

# numpy weeeee
import numpy

# high-level stuff
import tonic
import torch
import torchvision
import sklearn
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter

# snntorch, because it's stupid.
import snntorch
from snntorch import functional
from snntorch import surrogate

# visualization
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import seaborn
from snntorch import spikeplot
import torchinfo

# some very global constants
NOW = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
NUMPY_RNG = numpy.random.default_rng(42)
TORCH_RNG = torch.manual_seed(42)