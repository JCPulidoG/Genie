########################################   FUNCTIONS  ########################################

## IMPORT LIBRERIES
import pandas as pd
import holidays
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs

from datetime import datetime
from matplotlib import cycler
from statsmodels.tools.tools import add_constant # type: ignore
import statsmodels.api as sm # type: ignore

import contextlib
import sys
import os
import io


## FUNCTIONS TO DROP MESSAGE FROM YAHOO FINANCE
@contextlib.contextmanager
def fnSuppressStdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


##  FUNCTION TO ACTIVATE DARK MODE FOR VISUALIZATION
def fn_DarkMode():
    colors = cycler('color', ['#669FEE', '#66EE91', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
    plt.rc('figure', facecolor = '#313233')
    plt.rc('axes', facecolor = "#313233", edgecolor = 'none', axisbelow = True, grid = True, prop_cycle = colors, labelcolor = 'gray')
    plt.rc('grid', color = '474A4A', linestyle = 'solid')
    plt.rc('xtick', color = 'gray')
    plt.rc('ytick', direction = 'out', color = 'gray')
    plt.rc('legend', facecolor = "#313233", edgecolor = "#313233")
    plt.rc("text", color = "#C9C9C9")
    plt.rc('figure', facecolor = '#313233')


## FUNCTION TO BUILD THE BEST MODEL THROUGHT FORWARD SELECTION METHOD: FIRST INCLUDE VARIABLES ONE BY ONE UNTIL ALL ARE ADDED AND ANALIZED
def fnForwardSelection(data, target, significance_level = 0.05):
    initial_features = []
    remaining_features = list(data.columns)
    best_features = []
    while remaining_features:
        remaining_p_values = pd.Series(index=remaining_features)
        for feature in remaining_features:
            model = sm.OLS(target, add_constant(data[initial_features + [feature]])).fit()
            remaining_p_values[feature] = model.pvalues[feature]
        min_p_value = remaining_p_values.min()
        if min_p_value < significance_level:
            best_feature = remaining_p_values.idxmin()
            initial_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_features.append(best_feature)
        else:
            break
    return best_features


## FUNCTION TO BUILD THE BEST MODEL THROUGHT BACKWARD SELECTION METHOD: FIRST INCLUDE ALL VARIABLES THEN DROP ONE BY ONE UNTIL ONE IS LEFT
def fnBackwardSelection(data, target, significance_level = 0.05):
    features = list(data.columns)
    while len(features) > 0:
        model = sm.OLS(target, add_constant(data[features])).fit()
        max_p_value = model.pvalues.max()  # Obtener el valor de p más alto
        if max_p_value > significance_level:
            excluded_feature = model.pvalues.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features