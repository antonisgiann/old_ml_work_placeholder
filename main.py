# %%
# # # IMPORTS # # #
# General
import numpy as np
import pandas as pd
import scipy as sp
import os
import opendatasets as od
from joblib import load,dump
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve

# Proprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Model selection
from sklearn.model_selection import (
    cross_val_predict,
    StratifiedShuffleSplit, 
    StratifiedKFold,
    RandomizedSearchCV,
    GridSearchCV,
)
# Models
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
# Datasets
from sklearn.datasets import fetch_openml
# Custom
from utils import get_my_logger

# %%
logger = get_my_logger(__file__)

DATA_DIR = "datasets"
DATA_NAME = "mnist_784.pkl"
DATASET_PATH = os.path.join(os.getcwd(),DATA_DIR, DATA_NAME.split("_")[0], DATA_NAME)

# %%

# Get data
if os.path.exists(DATASET_PATH):
    df = load(DATASET_PATH)
else:
    if not os.path.exists(os.path.dirname(DATASET_PATH)):
        os.makedirs(os.path.dirname(DATASET_PATH))
    df = fetch_openml("mnist_784", version=1)
    with open(DATASET_PATH, "wb") as f:
        dump(df, f)
logger.info("Dataset loaded successfully")

X,y = df["data"], df["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# %%
logger.info("Running cross validation")
# Binary classification for 5s
y_train_5, y_test_5 = (y_train.astype(int) == 5), (y_test.astype(int) == 5)
logger.info("Running linear model...")
linear_preds = cross_val_predict(SGDClassifier(), X_train, y_train_5, cv=5, method="decision_function", n_jobs=-1, verbose=1)
logger.info("Running tree model...")
tree_preds = cross_val_predict(DecisionTreeClassifier(), X_train, y_train_5, cv=5, method="predict_proba", n_jobs=-1, verbose=1)
logger.info("Running forest model...")
forest_preds = cross_val_predict(RandomForestClassifier(), X_train, y_train_5, cv=5, method="predict_proba", n_jobs=-1, verbose=1)
logger.info("Predictions are ready for all models")
# %%