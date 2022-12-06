import pandas as pd
import numpy as np
import hvplot.pandas  # noqa
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os
import uncertainties as unc
import logging
import pickle
import math

import plotly.express as px
from uncertainties import unumpy
from uncertainties import ufloat
from scipy.constants import mu_0 as mu_0


logging.basicConfig(level=logging.WARNING)

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


def harmonic(t, a, w, p, c):
    return a * np.cos(w * t + p) + c 


def harmonic_fit(df, x='t', y='x', a0=None, w0=1.1, p0=0, c0=0, display=False):
    if a0 is None:
        a0 = np.abs(np.min(df[y]) - np.max(df[y])) / 20

    try:
        fit_params, covariances = curve_fit(harmonic, df[x], df[y], p0=[a0, w0, p0, c0],
                                            bounds=([0, 0, -2 * np.pi, -np.inf], [np.inf, np.inf, 2 * np.pi, np.inf]))
    except RuntimeError:
        logging.warning(f"fit failed for frequency {w0}")
        return None
        
    fit = harmonic(df[x], *fit_params)
    r_squred = r2_score(df[y], fit)

    if display:

        f1 = plt.figure()
        plt.plot(df[x], fit)
        plt.plot(df[x], df[y])
        plt.show()

        print(f'Fitted parameters: a = {fit_params[0]:.3f}, w = {fit_params[1]:.3f}, p = {fit_params[2]:.3f}, c = {fit_params[3]:.3f}')
    
    if r_squred < 0.90:
        logging.warning(f"R^2 is {r_squred}: for frequency{w0}")
        return None

    return fit_params


def find_phase_shift_index(df, p1, p2, w):
    t = df['t']

    t0 = abs(p1 - p2) / w
    difference_array = np.absolute(t - t0)

    # find the index of minimum element from the array
    n = difference_array.argmin()
    print(f"to index is {n}")
    return n

def extract_data_from_fit(func_dict, df, w, a0_1=1, p0_1=0, c0_1=0, a0_2=1, p0_2=0, c0_2=0, display=True, limit=None):
    
    params_1 = harmonic_fit(df, x='t', y='x', a0=a0_1, w0=2*np.pi*w, p0=p0_1, c0=c0_1, display=display)
    params_2 = harmonic_fit(df, x='t', y='y', a0=a0_2, w0=2*np.pi*w, p0=p0_2, c0=c0_1, display=display)

    if params_1 is None or params_2 is None:
        print(f"fit failed for frequency {w}")
        return None

    a1, w1, p1, c1  = tuple(params_1)
    a2, w2, p2, c2  = tuple(params_2)

    if np.abs(w1 - w2) / w2 > 0.05:
        print(f"Warning: frequencies are not the same for frequenciy {w}")
        return None

    result_dict = {}

    for name, func in func_dict.items():
        
        val  = func(w,a1,p1,c1,a2,p2,c2)
        
        if limit and name in limit:
            if val > limit[name][1] or val < limit[name][0]:
                logging.warning(f"{name} is out of range for frequency {w}")
                return None

        result_dict[name] = val

    return result_dict



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
        try:
            data_dict[os.path.splitext(os.path.basename(path))[0]] = load_data(path)
        except:
            print(f"Error reading file {path}")

    return data_dict


def find_hardness(xaxis, yaxis):
    positivezero, negativezero = findzeros(xaxis, yaxis)
    return abs(positivezero[0] - negativezero[0]) / 2


def find_peak(y_axis, x_axis):
    peak = [x_axis[1], y_axis[1]]
    for k in range(len(y_axis)):
        if y_axis[k] >= peak[1]:
            peak[1] = y_axis[k]
            peak[0] = x_axis[k]
    return peak

def main():
    # print(os.getcwd())
    # path4 = '/Users/user/Documents/semster_c/courses/lab/magnetisem/extension2/first/'

    # path = '../extension2/first/'
    # path2 = '../extension2/second/'
    # path3 = '../extension2/all/'

    # # d = lab_tools.read_to_dict(path)
    # d = read_to_dict(path4)
    # phases = []
    # frequencies = []
    # amplitudes = []
    # faild_fits = {}

    # def find_phase(w,a1,p1,c1,a2,p2,c2):
    #     f = 2 * np.pi / w 
    #     return f * (p1 - p2)

    # def find_z(w,a1,p1,c1,a2,p2,c2):
    #     return a2 / a1 

    # funcs = {'phase': find_phase, 'z': find_z}
    # limits = {'z': [0, 10000]}

    # for freq, df in d.items():
    #     freq = float(freq)
    #     logging.info(f"frequency is {freq}")
    #     result = extract_data_from_fit(funcs, df, freq, display=False)

    #     if result:
    #         frequencies.append(float(freq))
    #         phases.append(result['phase'])
    #         amplitudes.append(result['z'])
        
    #     else:
    #         faild_fits[freq] = df

    # failed = pickle.load(open('/Users/user/Documents/semster_c/courses/lab/magnetisem/code/pickles/failed.pickle', 'rb'))
    # harmonic_fit(failed['1836600'], w0=10000000, display=True)
    pass


def get_temperature_from_file_names(d):
    temperatures = []

    for x in d.keys():
        temperatures.append(-float(x))
    return temperatures
    
def find_amp(d):
    df = pd.DataFrame(columns=['temperature', 'amplitude'])
    amplitude = []
    for x in d.keys():
        amp_out = abs((max(d[x]['x']) - min(d[x]['x'])))
        amplitude.append(Constants.R * (Constants.Termo.a_in / amp_out))

    return amplitude

def load_pickle(name):
    with open(os.path.join('pickles',f'{name}.pickle'), 'rb') as handle:
        return pickle.load(handle)

def save_pickle(name):
    with open(os.path.join('pickles',f'{name}.pickle'), 'wb') as handle:
        return pickle.dump(handle)

class Constants:
    L = ufloat(0.0125,0.00025)
    R = ufloat(19.2,0.096)
    r = ufloat(0.03,0.003)
    A = 2*math.pi*(r**2)
    l = 2*r
    N_square = (L*l)/(A*mu_0)
    
    # the errors from the signal generator are really small
    class Termo:
        a_in = 5
        f_in = 507517
    

def uplot(x_ufloats, y_ufloats):
    plt.errorbar( unumpy.nominal_values(x_ufloats), unumpy.nominal_values(y_ufloats),
      xerr=unumpy.std_devs(x_ufloats), yerr= unumpy.std_devs(y_ufloats), fmt='o')  