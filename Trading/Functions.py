########################################   FUNCTIONS  ########################################

## IMPORT LIBRERIES
import pandas as pd
import holidays
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib import cycler
import scipy.stats as scs

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