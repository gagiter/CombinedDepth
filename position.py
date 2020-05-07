
import os
import argparse
import util
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
# from PIL.ExifTags import TAGS, GPSTAGS

parser = argparse.ArgumentParser(description='CombinedDepth')
parser.add_argument('--data_root', type=str, default='data')
args = parser.parse_args()


def save_exifs(root, names, exifs):
    pass


def add_position(exifs, position):
    return None


def read_exifs(root, names):
    exifs = []
    for name in names:
        image_path = os.path.join(root, name)
        exif = Image.open(image_path)._getexif()
        exifs.append(exif)
    return exifs


def read_gps(exifs):
    gps = []
    for meta in exifs:
        gps.append(util.Exif.read_gps(meta))
    return np.array(gps)


def localize(gps):
    converter = util.GPS()
    enu = []
    for i in range(gps.shape[0]):
        enu.append(converter.lla2enu(gps[i][0], gps[i][1], gps[i][2]))
    return np.array(enu)


def show_position(position):
    plt.clf()
    plt.plot(position[:, 0], position[:, 1], 'bs')
    plt.plot(position[:, 0], position[:, 1], 'r--')
    plt.show(block=False)


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
        names = [f for f in files if f.endswith('.jpg')]
        if len(names) == 0:
            continue
        exifs = read_exifs(root, names)
        gps = read_gps(exifs)
        position = localize(gps)
        show_position(position)
        print(root)
    # plt.show()
        # position = torch.tensor(np.array(position))
        # smooth_position(position)
        # exifs = add_position(exifs, position)
        # save_exifs(root, names, exifs)



if __name__ == '__main__':
    gps_to_position()
