
import os
import argparse
import csv
import util
import matplotlib.pyplot as plt
import torch
import numpy as np

parser = argparse.ArgumentParser(description='CombinedDepth')
parser.add_argument('--data_root', type=str, default='data')
args = parser.parse_args()


def lla2enu(gps):
    converter = util.GPS()
    enu = []
    for g in gps:
        lla = [float(v) for v in g.split()]
        enu.append(converter.lla2enu(*lla))
    return np.array(enu)


def show_position(position):
    plt.plot(position[:, 0], position[:, 1])
    plt.show()


def smooth_position(position):
    x = torch.from_numpy(position)
    optimizer = torch.optim.Adam(position)
    n = position.shape[0]
    for i in range(100):
        v0 = x[0:n-1]
        v1 = x[1:n]
        loss = (v1 - v0).abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        show_position(torch.to_numpy(x))

    return torch.to_numpy(x)


def gps_to_position():
    for root, dirs, files in os.walk(args.data_root):
        # if files.contains('*.jpg'):
        if 'train.csv' in files:
            with open(os.path.join(root, 'train.csv')) as csv_file:
                csv_reader = csv.reader(csv_file)
                keys = next(csv_reader)
                gps_id = keys.index('GPS')
                position_id = keys.index('Pos')
                gps = []
                for row in csv_reader:
                    gps.append(row[gps_id])
                position = lla2enu(gps)
                show_position(position)
                position = torch.tensor(np.array(position))
                smooth_position(position)
                for i in range(100):

                    show_position(position)
                for k, row in csv_reader:
                    row[position_id] = position[k]
                csv.save(csv_reader, os.path.join(root, 'train.csv'))


if __name__ == '__main__':
    gps_to_position()
