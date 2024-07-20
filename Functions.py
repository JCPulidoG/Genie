########################################   FUNCIONES PARA PORTAFOLIOS  ########################################

## IMPORTAR LIBRERÍAS
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


# FUNCIÓN PARA SUPRIMIR EL MENSAJE DE YAHOO FINANCE
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


## FUNCIÓN PARA ACTIVAR EL MODO OBSCURO EN LOS GRÁFICOS
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


## FUNCIÓN PARA OBTENER EL PRIMER/ÚLTIMO DÍA HÁBIL DEL AÑO
def fnGetDay(Year, Type, Country = 'MX'):
    
    if Type == "First":
        Dates = pd.date_range(start = f'{Year}-01-01', end = f'{Year}-01-31', freq = 'B')
    elif Type == "Last":
        Dates = pd.date_range(start = f'{Year}-12-01', end = f'{Year}-12-31', freq = 'B')
    Holidays = holidays.CountryHoliday(Country, years = [Year])
    Business_Days = [day for day in Dates if day not in Holidays]
    Day = np.where(Type == "First", Business_Days[0], Business_Days[-1]).tolist()
    Day = datetime.strftime(Day, '%Y-%m-%d')
    
    return Day


## FUNCIÓN PARA OBTENER EL SIGUIENTE DÍA HÁBIL DADA UNA FECHA
def fnGetNextDay(Date, Country = 'MX'):
    
    Date = pd.to_datetime(Date)
    Year = Date.year
    Holidays = holidays.CountryHoliday(Country, years = [Year, Year + 1])
    Next_Day = Date + pd.DateOffset(1)
    while Next_Day.weekday() >= 5 or Next_Day in Holidays:
        Next_Day += pd.DateOffset(1)
    Next_Day = datetime.strftime(Next_Day, '%Y-%m-%d')

    return Next_Day


## FUNCIÓN PARA OBTENER EL PRECIO DE UN ACTIVO DADA UNA FECHA
def fnGetPrice(Symbol, Date):

    Next_Day = fnGetNextDay(Date)
    with fnSuppressStdout():
        df_Price = yf.download(Symbol, start = Date, end = Next_Day)
    price = df_Price['Adj Close'].iloc[0]
    
    return price


## FUNCIÓN PARA OBTENER LAS ESTADÍSTICAS DE UNA VARIABLE
def fn_Statistics(Array):

    sta = scs.describe(Array)
    print('%14s %15s' % ('statistic', 'value'))
    print(30 * '-')
    print('%14s %15.5f' % ('size', sta[0]))
    print('%14s %15.5f' % ('min', sta[1][0]))
    print('%14s %15.5f' % ('max', sta[1][1]))
    print('%14s %15.5f' % ('mean', sta[2]))
    print('%14s %15.5f' % ('std', np.sqrt(sta[3])))
    print('%14s %15.5f' % ('skew', sta[4]))
    print('%14s %15.5f' % ('kurtosis', sta[5]))


# FUNCIÓN PARA OBTENER EL RETORNO ESPERADO ANUALIZADO DEL PORTAFOLIO
def fnPortfolioReturn(Weights, Returns):

    Return = np.sum(Returns.mean() * Weights) * 252

    return Return


# FUNCIÓN PARA OBTENER LA VOLATILIDAD ESPERADA ANUALIZADA DEL PORTAFOLIO
def fnPortfolioVolatility(Weights, Returns):

    Volatility = np.sqrt(np.dot(Weights.T, np.dot(Returns.cov() * 252, Weights)))

    return Volatility


# FUNCIÓN PARA MINIMIZAR LA VOLATILIDAD DEL PORTAFOLIO
def fnMinFuncSharpe(Weights, Returns):

    return -fnPortfolioReturn(Weights, Returns) / fnPortfolioVolatility(Weights, Returns)