import pandas as pd
import numpy as np
import hvplot.pandas  # noqa
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

pd.set_option('display.precision', 10)


def harmonic(t, a, w, p):
    return a * np.cos(w * t + p)

def harmonic_fit(df, x='t', y='x', a0=3, w0=1.1, p0=0):

    fit_params, covariances = curve_fit(harmonic, df[x], df[y], p0=[a0, w0, p0], bounds=([0, 0, -2 * np.pi], [np.inf, np.inf, 2 * np.pi]))

    fit = harmonic(df['t'], *fit_params)

    f1 =  plt.figure()
    plt.plot(df[x], fit)
    plt.plot(df[x], df[y])

    return fit_params

def find_phase_shift(df, p1, p2, w):
    t = df['t']

    t0 = abs(p1 - p2) / w
    difference_array = np.absolute(t-t0)

    # find the index of minimum element from the array
    n = difference_array.argmin()
    print(f"to index is {n}")
    return n

def fix_phase_shift(df ,n):
    x = list(df['x'][n:])
    y = list(df['y'][:len(df) - n])
    return x, y
    

def polyarea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def area(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in segments(p)))

def segments(p):
    return zip(p, p[1:] + [p[0]])

def findzeros(xaxis,yaxis):
    negativezero = [xaxis[5],yaxis[5]]
    positivezero = [xaxis[5],yaxis[5]]
    for k in range(len(yaxis)):
        if abs(yaxis[k]) <= abs(negativezero[1]) and xaxis[k] <= 0:
            negativezero[1] = yaxis[k]
            negativezero[0] = xaxis[k]
        elif abs(yaxis[k]) <= abs(negativezero[1]) and xaxis[k] >= 0:
            positivezero[1] = yaxis[k]
            positivezero[0] = xaxis[k]
    return (positivezero,negativezero)