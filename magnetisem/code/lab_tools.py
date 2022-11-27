import pandas as pd
import numpy as np
import hvplot.pandas  # noqa
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


# pd.set_option('display.precision', 10)

def show_multiple_graphs(d):
    fig, axs = plt.subplots(1, 4, figsize=(9, 3), sharey=True)
    for n, (k, v) in enumerate(d.items()):
        axs[n - 1].plot(v['x'], v['y'])
        axs[n - 1].set_title(k)
        plt.grid()
        plt.xlabel(r'$V_{x} \propto H [V]^$')
        plt.ylabel(r'$V_x$')


def a():
    pass


def convert_number_to_other_base(n, base):
    """
    Converts a number to another base
    n - the number to convert
    base - the base to convert to
    """

    if n < base:
        return [n]
    else:
        return convert_number_to_other_base(n // base, base) + [n % base]


def load_data(path):
    df = pd.read_csv(path, header=None, usecols=[3, 4, 10], names=['t', 'x', 'y'])
    return df


def harmonic(t, a, w, p):
    return a * np.cos(w * t + p)


def harmonic_fit(df, x='t', y='x', a0=3, w0=1.1, p0=0):
    fit_params, covariances = curve_fit(harmonic, df[x], df[y], p0=[a0, w0, p0],
                                        bounds=([0, 0, -2 * np.pi], [np.inf, np.inf, 2 * np.pi]))

    fit = harmonic(df['t'], *fit_params)

    f1 = plt.figure()
    plt.plot(df[x], fit)
    plt.plot(df[x], df[y])

    return fit_params


def find_phase_shift(df, p1, p2, w):
    t = df['t']

    t0 = abs(p1 - p2) / w
    difference_array = np.absolute(t - t0)

    # find the index of minimum element from the array
    n = difference_array.argmin()
    print(f"to index is {n}")
    return n


def fix_phase_shift(df, n):
    x = list(df['x'][n:])
    y = list(df['y'][:len(df) - n])
    return x, y


def polyarea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def area(p):
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for ((x0, y0), (x1, y1)) in segments(p)))


def segments(p):
    return zip(p, p[1:] + [p[0]])


def findzeros(xaxis, yaxis):
    negativezero = [xaxis[5], yaxis[5]]
    positivezero = [xaxis[5], yaxis[5]]
    for k in range(len(yaxis)):
        if abs(yaxis[k]) <= abs(negativezero[1]) and xaxis[k] <= 0:
            negativezero[1] = yaxis[k]
            negativezero[0] = xaxis[k]
        elif abs(yaxis[k]) <= abs(negativezero[1]) and xaxis[k] >= 0:
            positivezero[1] = yaxis[k]
            positivezero[0] = xaxis[k]
    return (positivezero, negativezero)


def find_hardness(xaxis, yaxis):
    positivezero, negativezero = findzeros(xaxis, yaxis)
    return abs(positivezero[0] - negativezero[0]) / 2


def read_to_dict(folder):
    csv_files = os.listdir(folder)
    data_dict = {}
    for x in csv_files:
        path = os.path.join(folder, x)
        data_dict[os.path.splitext(os.path.basename(path))[0]] = load_data(path)

    return data_dict
