import os
import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.datasets imoprtload_breast_cancer
from predict import make_prediction

MODEL_PATH="best_model.joblib"

#FIXTURE1:Load the trained model
