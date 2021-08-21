import pandas as pd
import numpy as np
import os

from helpers import *
from process import *

img_instance = Image_process()
X_train, X_cv, y_train, y_cv = img_instance.create_model_inputs()

