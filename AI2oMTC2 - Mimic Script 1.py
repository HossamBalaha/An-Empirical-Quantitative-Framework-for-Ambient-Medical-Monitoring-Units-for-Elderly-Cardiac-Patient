import math
import numpy as np
import os
import pandas as pd
import random
import warnings
from lightgbm import LGBMClassifier
from sklearn.metrics import (
  accuracy_score, auc, classification_report, confusion_matrix, f1_score,
  multilabel_confusion_matrix, precision_score,
  recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
  cross_val_predict, cross_val_score, GridSearchCV, KFold, RepeatedKFold,
  RepeatedStratifiedKFold, StratifiedKFold,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier
from CurrentDatasets import *

warnings.filterwarnings("ignore")


def AquilaOptimizer(X, Fs, Ps, D, lb, ub, t, T, fitnessFunction=None):
  bestIndex = np.argmax(Fs)
  bestSolution, bestFitness = X[bestIndex], Fs[bestIndex]
  newX = np.copy(X)

  G1 = 2.0 * np.random.uniform(1) - 1.0  # Equation 16.
  G2 = 2.0 * (1.0 - t / T)  # Equation 17.
  alpha = 0.1
  delta = 0.1
  D1 = np.array(list(range(1, D + 1)))
  omega = 0.005
  U = 0.00565
  r1 = 10.0
  r = r1 + U * D1
  theta1 = 3.0 * np.pi / 2.0
  theta = -omega * D1 + theta1
  x = r * np.sin(theta)  
  y = r * np.cos(theta)  
  QF = np.power(t, ((2.0 * np.random.uniform(1) - 1.0) / np.power((1.0 - T), 2)))  # Equation 15.

  for i in range(0, Ps):
    if (t <= (2.0 / 3.0) * T):
      if (np.random.uniform(1) < 0.5):
        # High soar with a vertical stoop.
        # Equation 4.
        avg = np.average(X)
        # Equation 3.
        tempSolution = bestSolution * (1.0 - t / T) + (avg - bestSolution * np.random.uniform(1))
      else:
        Xr = X[np.random.randint(0, Ps), :].copy()
        rand = np.random.uniform(1)
        levy = LevyChecked(D)
        tempSolution = bestSolution * levy + Xr + (y - x) * rand
    else:
      if (np.random.uniform(1) < 0.5):
        # A low flight with a slow descent attack.
        # Equation 4.
        avg = np.average(X)
        rand1 = np.random.uniform(1)
        rand2 = np.random.uniform(1)
        tempSolution = (bestSolution - avg) * alpha - rand1 + ((ub - lb) * rand2 + lb) * delta
      else:
        rand1 = np.random.uniform(1)
        rand2 = np.random.uniform(1)
        levy = LevyChecked(D)
        # Equation 14.
        tempSolution = (QF * bestSolution) - (G1 * X[i, :] * rand1) - (G2 * levy) + (rand2 * G1)

    tempSolution = np.clip(tempSolution, lb, ub)
    currentScore = fitnessFunction(tempSolution)
    if (currentScore > bestFitness):
      newX[i, :], bestFitness = tempSolution.copy(), currentScore

  newX = np.clip(newX, lb, ub)
  return newX, bestSolution, bestFitness


def FitnessFunction(solution):
  print(".", end="")
  X, y, solutionSize, datasetName = LoadDataset(DATASET_NAME)
  encoded = np.array(solution) >= 0.5
  if (np.sum(encoded) == 0):
    return 0.0
  newX = X[:, encoded].copy()
  score = PerformClassification(newX, y)
  return score


def PerformClassification(X, y):
  classifier = CLASSIFIER()
  steps = [
    ('scaler', None),
    ('classifier', classifier),
  ]
  pipe = Pipeline(steps=steps)
  if (APPLY_HYPER_OPT):
    params = PARAMS
  else:
    params = {}

  cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=1, random_state=0)
  grid = GridSearchCV(estimator=pipe, param_grid=params, cv=cv, refit=True, verbose=0, n_jobs=1)
  grid.fit(X, y)
  score = cross_val_score(grid.best_estimator_, X, y, cv=cv)
  # print("Cross Validation Scores:", score)
  # print("Best Hyperparameters:", grid.best_params_)
  # print("Cross Validation Scores Mean:", score.mean())
  # yPred = cross_val_predict(grid.best_estimator_, X, y, cv=cv)
  # cm = confusion_matrix(y, yPred)
  # print("Confusion Matrix:", cm)
  return score.mean()