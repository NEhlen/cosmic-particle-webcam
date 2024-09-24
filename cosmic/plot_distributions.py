import numpy as np
from scipy.ndimage import label
from scipy import ndimage
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import time
import math
from scipy.special import factorial
from scipy import stats
from scipy.optimize import minimize, curve_fit


df = pd.read_parquet("data/dist/disttest_cam4.parquet")


def poisson(k: np.ndarray, lamb: float = 1):
    return lamb**k * np.exp(-lamb) / (factorial(k))


def fit_poisson(params: list, series: pd.Series):
    return stats.poisson.pmf(series, params[0]).sum()


params = []

for column in df:
    data = df[column]
    bins = data.index
    popt, cov_matrix = curve_fit(
        poisson,
        bins,
        data / data.sum(),
    )
    params.append(popt[0])

plt.plot(params)

plt.figure()
for i in np.arange(0, 1200, 150):
    (line,) = plt.plot(df.iloc[:, i].index, df.iloc[:, i] / df.iloc[:, i].sum())
    plt.plot(poisson(df.iloc[:, i].index, params[i]), "+", color=line.get_color())


plt.imshow(df, aspect="auto", interpolation="none")
