import logging
import time

import numpy as np
from hermes.aeriel.client import InferenceClient
from tqdm import tqdm

from infer.data import Sequence
from infer.postprocess import Postprocessor