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

# type annotation
import typing
from typing import Literal, Callable, Sequence
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

# snntorch, because it's stupid.
import snntorch
from snntorch import functional
from snntorch import surrogate

# visualization
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from snntorch import spikeplot
import torchinfo