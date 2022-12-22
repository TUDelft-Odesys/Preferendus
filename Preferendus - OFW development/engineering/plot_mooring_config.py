"""
(c) Harold van Heukelum, 2022
"""

import os
import pathlib

import matplotlib.pyplot as plt
import moorpy as mp
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
results_dir = None

list_dir = os.listdir(results_dir)
list_dir.sort()
try:
    list_dir.remove('.DS_Store')
except ValueError:  # DS_Store is not always present
    pass

results = [list_dir[item] for item in range(0, len(list_dir), 6)]

floater_data = {
    'xcm': '0',
    'ycm': '0',
    'zcm': '-14.94',
    'mass': 'yyyy',
    'volume': '20206',
    'ix': '1.251E+10',
    'iy': '1.251E+10',
    'iz': '2.367E+10',
    'cd': '0',
    'ca': '0',
    'AWP': 446.70
}

mooring_data = {
    'n lines': '3',
    'depth fairleads': '14',
    'R fairleads': '58'
}

col_wd = list()
col_mc = list()
col_d = list()
col_angle_wo_force = list()
col_angle_w_force = list()


def progress(percent=0, width=40):
    """Function to print progress bars"""
    left = width * percent // 100
    right = width - left

    tags = "#" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"

    print("\r[", tags, spaces, "]", percents, sep="", end="", flush=True)


class Boss:
    """
    Class extract mooring data and plot it
    """

    def __init__(self):
        self.floater_data = floater_data
        self.mooring_data = mooring_data

        self.ms = None

    def save_configs(self, shared=False):
        """Create and save plots of moorings"""
        progress()
        for it, f in enumerate(results):
            progress(int(it / len(results) * 100))
            if '_ma-8' in f or '_ma8' in f or 'SLC' in f:
                continue

            wd = f[7:10]  # water depth
            col_wd.append(wd)

            if 'taut' in f:  # mooring config
                mc = 'taut'
            else:
                mc = 'catenary'
            col_mc.append(mc)

            if '_d60' in f:  # direction
                d = 60
            elif '_d30' in f:
                d = 30
            else:
                d = 0
            col_d.append(d)

            # try making directories if they are not yet there
            try:
                os.mkdir(f'results/{wd}m/plots_single_mooring/')
                os.mkdir(f'results/{wd}m/plots_shared_mooring/')
            except FileExistsError:
                pass

            if not shared:
                if mc == 'taut':
                    self.build_system_taut(f)
                else:
                    self.build_system_catenary(f)
                plt.savefig(f'results/{wd}m/plots_single_mooring/{mc}_d{d}_ini.png')
                plt.close()

                fig, ax = self.plot_mooring(mc)
                plt.savefig(f'results/{wd}m/plots_single_mooring/{mc}_d{d}.png')

                ax.view_init(elev=90, azim=-90)
                plt.savefig(f'results/{wd}m/plots_single_mooring/{mc}_d{d}_top.png')
                plt.close()

            else:
                angle = np.radians(120)
                if mc == 'taut':
                    self.build_system_taut(f, shared, float(angle), float(d))
                else:
                    self.build_system_catenary(f, shared, float(angle), float(np.radians(d)))
                plt.savefig(f'results/{wd}m/plots_shared_mooring/{mc}_d{d}_shared_ini.png')
                plt.close()

                fig, ax = self.plot_mooring(mc)
                plt.savefig(f'results/{wd}m/plots_shared_mooring/{mc}_d{d}_shared.png')

                ax.view_init(elev=90, azim=-90)
                plt.savefig(f'results/{wd}m/plots_shared_mooring/{mc}_d{d}_shared_top.png')
                plt.close()

        progress(100)
        print()
        return

    def plot_mooring(self, mc):
        """Make plots of moorings"""
        ms = self.ms
        for body in ms.bodyList:
            body.f6Ext = np.array([0, 0, 0, 0, 0, 0])

        ms.solveEquilibrium3()  # equilibrate
        fig, ax = ms.plot(shadow=False)  # plot the system in original configuration

        if mc == 'taut':
            point_a1 = ms.pointList[0].r
            point_f1 = ms.pointList[3].r
            angle_ml = np.arctan(
                -1 * point_a1[2] / np.sqrt((point_a1[0] - point_f1[0]) ** 2 + (point_a1[1] - point_f1[1]) ** 2))
        else:
            angle_ml = 0
        col_angle_wo_force.append(angle_ml)

        for body in ms.bodyList:
            body.f6Ext = np.array([3e6, 0, 0, 0, 0, 0])  # apply an external force on the body
        ms.solveEquilibrium3()  # equilibrate
        fig, ax = ms.plot(ax=ax, color='blue', shadow=False, linelabels=True)

        if mc == 'taut':
            point_a1 = ms.pointList[0].r
            point_f1 = ms.pointList[3].r
            angle_ml = np.arctan(
                -1 * point_a1[2] / np.sqrt((point_a1[0] - point_f1[0]) ** 2 + (point_a1[1] - point_f1[1]) ** 2))
        else:
            angle_ml = 0
        col_angle_w_force.append(angle_ml)

        plt.title('Mooring Configuration')
        return fig, ax

    def build_base(self):
        """Create the basis of the mooring system in MoorPy"""
        ms = mp.System()
        ms.addLineType(type_string='chain', d=0.333, massden=685, EA=3.27e9)  # 15MW semi
        ms.addLineType(type_string='rope', d=0.211, massden=23, EA=3.89e8)  # poly rope (DeepRope)
        return ms

    def build_system_catenary(self, file, shared=False, angle=0., direction=0.):
        """
        Build the catenary mooring system
        """

        if shared:
            iterations = 3
        else:
            iterations = 1

        system = self.build_base()

        for i in range(iterations):
            connects, lines = self.load_mooring_config(file)

            if shared:
                radius = np.sqrt(float(connects[0][2]) ** 2 + float(connects[0][3]) ** 2)
                x = -1 * np.cos(i * angle + direction) * float(radius)
                y = -1 * np.sin(i * angle + direction) * float(radius)
                r6 = np.array([x, y, 0, 0, 0, 0])
                arr = [x, y, 0]
            else:
                r6 = np.array([0, 0, 0, 0, 0, 0])
                arr = [0, 0, 0]

            system.addBody(mytype=0, r6=r6, m=20090931.2, v=20206,
                           rCG=np.array([0, 0, -14.94]),
                           AWP=floater_data['AWP'])

            system.addPoint(mytype=1, r=np.float_(connects[0][2:5]) + arr)
            system.addPoint(mytype=-1, r=np.float_(connects[1][2:5]) + arr)
            system.bodyList[i].attachPoint([2, 8, 14][i], np.float_(connects[1][2:5]))

            system.addPoint(mytype=1, r=np.float_(connects[2][2:5]) + arr)
            system.addPoint(mytype=-1, r=np.float_(connects[3][2:5]) + arr)
            system.bodyList[i].attachPoint([4, 10, 16][i], np.float_(connects[3][2:5]))

            system.addPoint(mytype=1, r=np.float_(connects[4][2:5]) + arr)
            system.addPoint(mytype=-1, r=np.float_(connects[5][2:5]) + arr)
            system.bodyList[i].attachPoint([6, 12, 18][i], np.float_(connects[5][2:5]))

            for line in lines:
                system.addLine(lUnstr=float(line[2]), type_string=line[1], nSegs=int(line[3]),
                               pointA=int(line[4]) + i * 6,
                               pointB=int(line[5]) + i * 6)

        system.depth = -1 * float(connects[0][4])

        system.initialize(plots=1)
        self.ms = system
        return

    def build_system_taut(self, file, shared=False, angle=0., direction=0.):
        """
        Build the taut mooring system
        """
        if shared:
            iterations = 3
        else:
            iterations = 1

        system = self.build_base()

        for i in range(iterations):
            connects, lines = self.load_mooring_config(file)

            if shared:
                radius = np.sqrt(float(connects[0][2]) ** 2 + float(connects[0][3]) ** 2)
                x = -1 * np.cos(i * angle + direction) * float(radius)
                y = -1 * np.sin(i * angle + direction) * float(radius)
                r6 = np.array([x, y, 0, 0, 0, 0])
                arr = [x, y, 0]
            else:
                r6 = np.array([0, 0, 0, 0, 0, 0])
                arr = [0, 0, 0]

            system.addBody(mytype=0, r6=r6, m=20090931.2, v=20206,
                           rCG=np.array([0, 0, -14.94]),
                           AWP=floater_data['AWP'])

            system.addPoint(mytype=1, r=np.float_(connects[0][2:5]) + arr)
            system.addPoint(mytype=0, r=np.float_(connects[1][2:5]) + arr)
            system.addPoint(mytype=0, r=np.float_(connects[2][2:5]) + arr)
            system.addPoint(mytype=-1, r=np.float_(connects[3][2:5]) + arr)
            system.bodyList[i].attachPoint([4, 16, 28][i], np.float_(connects[3][2:5]))

            system.addPoint(mytype=1, r=np.float_(connects[4][2:5]) + arr)
            system.addPoint(mytype=0, r=np.float_(connects[5][2:5]) + arr)
            system.addPoint(mytype=0, r=np.float_(connects[6][2:5]) + arr)
            system.addPoint(mytype=-1, r=np.float_(connects[7][2:5]) + arr)
            system.bodyList[i].attachPoint([8, 20, 32][i], np.float_(connects[7][2:5]))

            system.addPoint(mytype=1, r=np.float_(connects[8][2:5]) + arr)
            system.addPoint(mytype=0, r=np.float_(connects[9][2:5]) + arr)
            system.addPoint(mytype=0, r=np.float_(connects[10][2:5]) + arr)
            system.addPoint(mytype=-1, r=np.float_(connects[11][2:5]) + arr)
            system.bodyList[i].attachPoint([12, 24, 36][i], np.float_(connects[11][2:5]))

            for line in lines:
                system.addLine(lUnstr=float(line[2]), type_string=line[1], nSegs=int(line[3]),
                               pointA=int(line[4]) + i * 12,
                               pointB=int(line[5]) + i * 12)

        system.depth = -1 * float(connects[0][4])

        system.initialize(plots=1)
        self.ms = system
        return

    def load_mooring_config(self, file):
        """
        Load the mooring config file (.MD) and extract the important data
        """
        with open(results_dir + file, 'r') as t:
            text = t.readlines()

        connects = list()
        if 'taut' in file:
            for i in range(13, 25):
                connects.append(text[i].split())
        else:
            for i in range(13, 19):
                connects.append(text[i].split())

        lines = list()
        if 'taut' in file:
            for i in range(29, 38):
                lines.append(text[i].split())
        else:
            for i in range(23, 26):
                lines.append(text[i].split())
        return connects, lines


if __name__ == '__main__':
    cls = Boss()
    print('Run single turbines configs')
    cls.save_configs()

    data = {
        'Water depth': col_wd,
        'Mooring config': col_mc,
        'Direction': col_d,
        'Angle ML wo force [rad]': col_angle_wo_force,
        'Angle ML w force (3e6) [rad]': col_angle_w_force,
    }
    df = pd.DataFrame(data=data)
    df.to_csv('export_ML_angles.csv')

    print('Run shared turbines configs')
    cls.save_configs(shared=True)
