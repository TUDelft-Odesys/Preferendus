"""
Extract forces from OpenFAST output files (outb)
"""
import os
import pathlib
from random import shuffle
from time import perf_counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyFAST.input_output import FASTOutputFile
from windrose import WindroseAxes

# determine where the script and results are located
HERE = pathlib.Path(__file__).parent
results_dir = None

# all parameters for which OpenFAST has run
water_depths = np.arange(120, 151, 5)
env_conditions = ['DLC1.6', 'SLC']
mooring_conf = ['catenary', 'taut']
directions = [0, 30, 60]
misalignments = [-8, 0, 8]
runs = [1, 2, 3, 4, 5, 6]

# try making directories if they are not yet there
try:
    for wd in water_depths:
        os.mkdir(f'results/{wd}m/')
        for ec in env_conditions:
            os.mkdir(f'results/{wd}m/{ec}/')
            for mc in mooring_conf:
                os.mkdir(f'results/{wd}m/{ec}/{mc}/')
except FileExistsError:
    pass

# get all files in the results folder
list_dir = os.listdir(results_dir)
list_dir.sort()
try:
    list_dir.remove('.DS_Store')
except ValueError:  # DS_Store is not always present
    pass


def break_lists(lst, size):
    """
    Compact the list so the runs with the same parameters (six in total) are combined in one list
    """
    ret = list()
    for i in range(0, len(lst), size):
        ret.append(lst[i:i + size])
    return ret


file_list_broken = break_lists(list_dir, 6)

# define the empty lists to store eventually in a DataFrame
col_wd = list()
col_ec = list()
col_mc = list()
col_ma = list()
col_d = list()
col_max_a1_single = list()
col_max_a2_single = list()
col_max_a3_single = list()
col_max_shared = list()
col_dir_shared = list()
col_max_shared_2a = list()
col_max_t1_shared = list()
col_max_t2_shared = list()
col_max_t3_shared = list()
seq_t2 = list()
seq_t3 = list()


def extract(file_to_extract):
    """
    Extract force data from outb-files
    """
    data_frame = FASTOutputFile(f'{results_dir}/{file_to_extract}').toDataFrame()

    if mc == 'catenary':
        f1 = data_frame['FAIRTEN1_[N]'].values.tolist()
        f2 = data_frame['FAIRTEN2_[N]'].values.tolist()
        f3 = data_frame['FAIRTEN3_[N]'].values.tolist()

        a1 = data_frame['ANCHTEN1_[N]'].values.tolist()
        a2 = data_frame['ANCHTEN2_[N]'].values.tolist()
        a3 = data_frame['ANCHTEN3_[N]'].values.tolist()

    else:
        f1 = data_frame['FAIRTEN3_[N]'].values.tolist()
        f2 = data_frame['FAIRTEN6_[N]'].values.tolist()
        f3 = data_frame['FAIRTEN9_[N]'].values.tolist()

        a1 = data_frame['ANCHTEN1_[N]'].values.tolist()
        a2 = data_frame['ANCHTEN4_[N]'].values.tolist()
        a3 = data_frame['ANCHTEN7_[N]'].values.tolist()

    return [f1, f2, f3, a1, a2, a3]


def strip_begin_end(arr):
    """
    Strip initialization phase of catenary moorings
    :return:
    """
    loc_end = list()
    loc_start = list()
    arr_new = arr.copy()
    for itt, ls in enumerate(arr):
        mean_ls = np.mean(ls)
        for i in range(len(ls) - 1, 0, -1):
            if ls[i - 1] <= mean_ls <= ls[i]:
                loc_end.append(i)
                break
        else:
            loc_end.append(len(ls) - 1)

        for i in range(int(60 / 0.025), len(ls)):
            if ls[i - 1] <= mean_ls <= ls[i]:
                loc_start.append(i)
                break
        else:
            loc_start.append(0)

        arr_new[itt] = ls[loc_start[itt]:loc_end[itt] + 1]

    return arr_new, loc_start, loc_end


# run through all files
index = 0
for runs_list in file_list_broken:
    # check if runs are sequential in file list
    run_nr_check = 1
    for file in runs_list:
        run = int(file[-6:-5])
        if run != run_nr_check:
            raise ValueError('Aborted because run numbers are non-sequential')
        run_nr_check += 1
        if run_nr_check > 6:
            run_nr_check = 1

    # define list per turbine (3 turbines for shared moorings)
    seq_list_t1 = [1, 2, 3, 4, 5, 6]
    seq_list_t2 = [1, 2, 3, 4, 5, 6]
    seq_list_t3 = [1, 2, 3, 4, 5, 6]

    # get random sequence of runs for turbine 2 & 3
    mask_t2 = np.array(seq_list_t1) == np.array(seq_list_t2)
    while mask_t2.any():
        shuffle(seq_list_t2)
        mask_t2 = np.array(seq_list_t1) == np.array(seq_list_t2)

    mask_t3_1 = np.array(seq_list_t1) == np.array(seq_list_t3)
    mask_t3_2 = np.array(seq_list_t1) == np.array(seq_list_t3)
    mask_t3 = np.min([mask_t3_1, mask_t3_2], axis=0)
    while mask_t3.any():
        shuffle(seq_list_t3)
        mask_t3_1 = np.array(seq_list_t1) == np.array(seq_list_t3)
        mask_t3_2 = np.array(seq_list_t1) == np.array(seq_list_t3)
        mask_t3 = np.min([mask_t3_1, mask_t3_2], axis=0)

    # define empty lists for forces
    fair_ten_1 = list()
    fair_ten_2 = list()
    fair_ten_3 = list()
    anch_ten_1 = list()
    anch_ten_2 = list()
    anch_ten_3 = list()

    s = list()
    e = list()

    # get the first file name to extract data about the parameters and extract the parameters
    f = runs_list[0]
    tic = perf_counter()

    wd = f[7:10]  # water depth
    col_wd.append(wd)

    if 'SLC' in f:  # design condition
        ec = 'SLC'
    else:
        ec = 'DLC1.6'
    col_ec.append(ec)

    if 'taut' in f:  # mooring config
        mc = 'taut'
    else:
        mc = 'catenary'
    col_mc.append(mc)

    if '_ma-8' in f:  # mis-alignment
        ma = -8
    elif '_ma8' in f:
        ma = 8
    else:
        ma = 0
    col_ma.append(ma)

    if '_d60' in f:  # direction
        d = 60
    elif '_d30' in f:
        d = 30
    else:
        d = 0
    col_d.append(d)

    for f in runs_list:
        fa_base = extract(f)
        if mc == 'catenary':
            fa, s, e = strip_begin_end(fa_base)
            start = s[3]
            end = e[3] + 1
        else:
            fa = fa_base
            start = 0
            end = len(fa_base[3])

        fair_ten_1.append(fa_base[0][start:end])
        fair_ten_2.append(fa_base[1][start:end])
        fair_ten_3.append(fa_base[2][start:end])

        anch_ten_1.append(fa[3])
        anch_ten_2.append(fa_base[4][start:end])
        anch_ten_3.append(fa_base[5][start:end])

    fair_ten_1_ = [list(), list(), list()]
    fair_ten_2_ = [list(), list(), list()]
    fair_ten_3_ = [list(), list(), list()]
    anch_ten_1_ = [list(), list(), list()]
    anch_ten_2_ = [list(), list(), list()]
    anch_ten_3_ = [list(), list(), list()]

    seq_list = [seq_list_t1, seq_list_t2, seq_list_t3]
    for i in range(3):
        seq = ''
        for j in range(len(seq_list_t1)):
            fair_ten_1_[i] += fair_ten_1[seq_list[i][j] - 1]
            fair_ten_2_[i] += fair_ten_2[seq_list[i][j] - 1]
            fair_ten_3_[i] += fair_ten_3[seq_list[i][j] - 1]
            anch_ten_1_[i] += anch_ten_1[seq_list[i][j] - 1]
            anch_ten_2_[i] += anch_ten_2[seq_list[i][j] - 1]
            anch_ten_3_[i] += anch_ten_3[seq_list[i][j] - 1]

            seq += str(seq_list[i][j])
        if i == 1:
            seq_t2.append(seq)
        elif i == 2:
            seq_t3.append(seq)

    forces_t1 = np.array([anch_ten_1_[0], anch_ten_2_[0], anch_ten_3_[0]])
    max_f_t1 = np.amax(forces_t1)
    loc_max_f_t1 = np.where(forces_t1 == max_f_t1)[1]

    col_max_a1_single.append(forces_t1[0, loc_max_f_t1][0])
    col_max_a2_single.append(forces_t1[1, loc_max_f_t1][0])
    col_max_a3_single.append(forces_t1[2, loc_max_f_t1][0])

    forces_catenary_shared = np.array([anch_ten_1_[0], anch_ten_2_[1], anch_ten_3_[2]])

    f_net_x = forces_catenary_shared[0, :] - np.cos(np.radians(60)) * (
            forces_catenary_shared[1, :] + forces_catenary_shared[2, :])
    f_net_y = np.sin(np.radians(60)) * (forces_catenary_shared[1, :] - forces_catenary_shared[2, :])
    f_net = np.sqrt(f_net_x ** 2 + f_net_y ** 2)

    f_2a_net_x1 = np.cos(np.radians(60)) * (forces_catenary_shared[0, :] + forces_catenary_shared[1, :])
    f_2a_net_y1 = np.sin(np.radians(60)) * (forces_catenary_shared[0, :] - forces_catenary_shared[1, :])
    f_2a_net_1 = np.sqrt(f_2a_net_x1 ** 2 + f_2a_net_y1 ** 2)

    f_2a_net_x2 = np.cos(np.radians(60)) * (forces_catenary_shared[1, :] + forces_catenary_shared[2, :])
    f_2a_net_y2 = np.sin(np.radians(60)) * (forces_catenary_shared[1, :] - forces_catenary_shared[2, :])
    f_2a_net_2 = np.sqrt(f_2a_net_x2 ** 2 + f_2a_net_y2 ** 2)

    f_2a_net_x3 = np.cos(np.radians(60)) * (forces_catenary_shared[2, :] + forces_catenary_shared[0, :])
    f_2a_net_y3 = np.sin(np.radians(60)) * (forces_catenary_shared[2, :] - forces_catenary_shared[0, :])
    f_2a_net_3 = np.sqrt(f_2a_net_x3 ** 2 + f_2a_net_y3 ** 2)

    f_2a_net = np.array([f_2a_net_1, f_2a_net_2, f_2a_net_3])

    max_f_shared = np.max(f_net)
    col_max_shared.append(max_f_shared)
    loc_max_shared = np.where(f_net == max_f_shared)[0][0]
    col_max_t1_shared.append(anch_ten_1_[0][loc_max_shared])
    col_max_t2_shared.append(anch_ten_2_[1][loc_max_shared])
    col_max_t3_shared.append(anch_ten_3_[2][loc_max_shared])

    max_f_2a = np.max(f_2a_net)
    loc_max_2a = np.where(f_2a_net == max_f_2a)
    col_max_shared_2a.append(max_f_2a)

    mask_fx_min = f_net_x < 0
    mask_fy_min = f_net_y < 0
    mask_both = np.logical_and(mask_fx_min, mask_fy_min)

    f_direction = np.degrees(np.arctan(f_net_y / f_net_x))
    f_direction[mask_fx_min] = 180 + np.degrees(np.arctan(f_net_y[mask_fx_min] / f_net_x[mask_fx_min]))
    f_direction[mask_fy_min] = 360 + np.degrees(np.arctan(f_net_y[mask_fy_min] / f_net_x[mask_fy_min]))
    f_direction[mask_both] = 180 + np.degrees(np.arctan(f_net_y[mask_both] / f_net_x[mask_both]))

    col_dir_shared.append(f_direction[loc_max_shared])

    x = np.arange(0, len(anch_ten_1_[0]) * 0.025, 0.025)
    if len(anch_ten_1_[0]) != len(x):
        x = x[:-1]
    plt.figure(figsize=(25, 15))
    plt.plot(x / 60, np.multiply(anch_ten_1_[0], 1e-6), label='Anchor 1')
    plt.plot(x / 60, np.multiply(anch_ten_2_[0], 1e-6), label='Anchor 2')
    plt.plot(x / 60, np.multiply(anch_ten_3_[0], 1e-6), label='Anchor 3')
    plt.title('Forces on the anchors of one FWT')
    plt.xlabel('Time [min]')
    plt.ylabel('Force [MN]')
    plt.xlim((0, 60))
    plt.legend()
    plt.grid()
    plt.savefig(f'results/{wd}m/{ec}/{mc}/direction{d}_misallignment{ma}_anchors_single.png')
    plt.close()

    plt.figure(figsize=(25, 15))
    plt.plot(x / 60, np.multiply(fair_ten_1_[0], 1e-6), label='Fairlead 1')
    plt.plot(x / 60, np.multiply(fair_ten_2_[0], 1e-6), label='Fairlead 2')
    plt.plot(x / 60, np.multiply(fair_ten_3_[0], 1e-6), label='Fairlead 3')
    plt.title('Forces on the fairleads of one FWT')
    plt.xlabel('Time [min]')
    plt.ylabel('Force [MN]')
    plt.xlim((0, 60))
    plt.legend()
    plt.grid()
    plt.savefig(f'results/{wd}m/{ec}/{mc}/direction{d}_misallignment{ma}_fairleads_single.png')
    plt.close()

    plt.figure(figsize=(25, 15))
    plt.plot(x / 60, np.multiply(anch_ten_1_[0], 1e-6), label='T1, A1')
    plt.plot(x / 60, np.multiply(anch_ten_2_[1], 1e-6), label='T2, T2')
    plt.plot(x / 60, np.multiply(anch_ten_3_[2], 1e-6), label='T3, A3')
    plt.plot(x / 60, f_net * 1e-6, label='Net force')
    plt.title('Forces acting on the shared anchor incl. net force')
    plt.xlabel('Time [min]')
    plt.ylabel('Force [MN]')
    plt.xlim((0, 60))
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.savefig(f'results/{wd}m/{ec}/{mc}/direction{d}_misallignment{ma}_anchors_shared.png')
    plt.close()

    plt.figure(figsize=(25, 15))
    if loc_max_2a[0][0] == 0:
        plt.plot(x / 60, np.multiply(anch_ten_1_[0], 1e-6), label='T1, A1')
        plt.plot(x / 60, np.multiply(anch_ten_2_[1], 1e-6), label='T2, T2')
        plt.plot(x / 60, f_2a_net_1 * 1e-6, label='Net force')
    elif loc_max_2a[0][0] == 1:
        plt.plot(x / 60, np.multiply(anch_ten_2_[1], 1e-6), label='T2, A2')
        plt.plot(x / 60, np.multiply(anch_ten_3_[2], 1e-6), label='T3, A3')
        plt.plot(x / 60, f_2a_net_2 * 1e-6, label='Net force')
    else:
        plt.plot(x / 60, np.multiply(anch_ten_3_[2], 1e-6), label='T3, A3')
        plt.plot(x / 60, np.multiply(anch_ten_1_[0], 1e-6), label='T1, T1')
        plt.plot(x / 60, f_2a_net_3 * 1e-6, label='Net force')
    plt.title('Forces acting on the dual shared anchor incl. net force')
    plt.xlabel('Time [min]')
    plt.ylabel('Force [MN]')
    plt.xlim((0, 60))
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.savefig(f'results/{wd}m/{ec}/{mc}/direction{d}_misallignment{ma}_anchors_shared_2.png')
    plt.close()

    ax = WindroseAxes.from_ax()
    ax.bar(f_direction, f_net * 1e-6, normed=True, opening=0.8, edgecolor="white")
    ax.set_legend()
    plt.title('Direction of the net force on the shared anchor')
    plt.savefig(f'results/{wd}m/{ec}/{mc}/direction{d}_misallignment{ma}_shared_rose.png')
    plt.close()

    sr = 1 / 0.025
    N_1 = len(anch_ten_1_[0])
    n_1 = np.arange(N_1)
    freq_1 = n_1 / (N_1 / sr)
    X_1 = np.fft.fft(anch_ten_1_[0])

    N_2 = len(anch_ten_2_[0])
    n_2 = np.arange(N_2)
    freq_2 = n_2 / (N_2 / sr)
    X_2 = np.fft.fft(anch_ten_2_[0])

    N_3 = len(anch_ten_3_[0])
    n_3 = np.arange(N_3)
    freq_3 = n_3 / (N_3 / sr)
    X_3 = np.fft.fft(anch_ten_3_[0])

    plt.figure(figsize=(25, 15))
    plt.stem(freq_1, np.abs(X_1), 'b', markerfmt=" ", basefmt="-b", label='A1')
    plt.stem(freq_2, np.abs(X_2), 'r', markerfmt=" ", basefmt="-b", label='A2')
    plt.stem(freq_3, np.abs(X_3), 'tab:cyan', markerfmt=" ", basefmt="-b", label='A3')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 0.02)
    plt.ylim(0, 15e9)
    plt.title('FFT of force spectrum')
    plt.legend()
    plt.grid()
    plt.savefig(f'results/{wd}m/{ec}/{mc}/direction{d}_misallignment{ma}_FFT.png')
    plt.close()

    toc = perf_counter()
    print(f'Run {wd}-{ec}-{mc}-{d}-{ma} finished in {round(toc - tic, 2)}s')

data = {
    'Water depth': col_wd,
    'Env. condition': col_ec,
    'Mooring config': col_mc,
    'Direction': col_d,
    'Misalignment': col_ma,
    'Max A1, single': col_max_a1_single,
    'Max A2, single': col_max_a2_single,
    'Max A3, single': col_max_a3_single,
    'Max shared mooring': col_max_shared,
    'Dir shared mooring': col_dir_shared,
    'Max shared mooring 2 anchors': col_max_shared_2a,
    'Max A1,T1 shared': col_max_t1_shared,
    'Max A2,T2 shared': col_max_t2_shared,
    'Max A3,T3 shared': col_max_t3_shared,
    'Sequence runs T2': seq_t2,
    'Sequence runs T3': seq_t3,
}

df = pd.DataFrame(data=data)
df.to_csv('export_forces.csv')
