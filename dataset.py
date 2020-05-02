
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import csv
import math
import random
from PIL import ImageOps
import util
import numpy as np



class Data(Dataset):
    def __init__(self, data_root, target_pixels=350000,
                 target_width=640, target_height=480, use_number=0,
                 mode='train', device='cpu', shuffle=True):
        super(Data, self).__init__()
        self.data_root = data_root
        self.target_pixels = target_pixels
        self.target_width = target_width
        self.target_height = target_height
        self.use_number = use_number
        self.mode = mode
        self.device = device
        self.data = []
        self.index = -1
        self.shuffle = shuffle
        self.load_data()

    def load_data(self):
        csv_name = self.mode + '.csv'
        for root, dirs, files in os.walk(self.data_root):
            if csv_name in files:
                with open(os.path.join(root, csv_name)) as csv_file:
                    csv_reader = csv.reader(csv_file)
                    keys = next(csv_reader)
                    previous_id = keys.index('previous')
                    next_id = keys.index('next')
                    for row in csv_reader:
                        if row[previous_id] != 'None' and row[next_id] != 'None':
                            item = dict(zip(keys, row))
                            self.data.append((root, item))

    def __len__(self):
        return len(self.data) if self.use_number == 0 else \
            min(self.use_number, len(self.data))

    def __getitem__(self, idx):
        root = self.data[idx][0]
        item = self.data[idx][1]
        image = os.path.join(root, item['image'])
        depth = None if item['depth'] == 'None' else os.path.join(root, item['depth'])
        refs = dict()
        for ref_name in ['previous', 'next']:  # , 'previous' ['stereo', 'previous', 'next']: # , 'previous', 'next'
            refs[ref_name] = os.path.join(root, item[ref_name])

        out = dict()
        image = Image.open(image)
        image = ImageOps.exif_transpose(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.target_pixels > 0:
            scale_factor = math.sqrt(self.target_pixels / (image.width * image.height))
            height = int(image.height * scale_factor)
            width = int(image.width * scale_factor)
        else:
            height = self.target_height
            width = self.target_width
        image = TF.resize(image, (height, width))
        image = TF.to_tensor(image)
        image = TF.normalize(image, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        out['image'] = image

        if depth is not None:
            depth = Image.open(depth)
            depth = ImageOps.exif_transpose(depth)
            depth = TF.resize(depth, (height, width), Image.NEAREST)
            depth = TF.to_tensor(depth).float()
            depth /= 256.0
            mask = depth > 0.1
            # mask &= depth < 80.0
            depth[mask] = 1.0 / depth[mask]
            out['depth'] = depth

        for ref_name in refs:
            refs[ref_name] = Image.open(refs[ref_name])
            refs[ref_name] = ImageOps.exif_transpose(refs[ref_name])
            refs[ref_name] = TF.resize(refs[ref_name], (height, width))
            if refs[ref_name].mode != 'RGB':
                refs[ref_name] = refs[ref_name].convert('RGB')
            refs[ref_name] = TF.to_tensor(refs[ref_name])
            out[ref_name] = refs[ref_name]

        if item['GPS'] != 'None' and item['GPS_previous'] != 'None' and item['GPS_next'] != 'None':
            gps = util.GPS()
            lla_image = [float(i) for i in item['GPS'].split()]
            gps.set_origin(*lla_image)
            lla_previous = [float(i) for i in item['GPS_previous'].split()]
            distance_previous = np.linalg.norm(np.array(gps.lla2enu(*lla_previous)), keepdims=True)
            lla_next = [float(i) for i in item['GPS_next'].split()]
            distance_next = np.linalg.norm(np.array(gps.lla2enu(*lla_next)), keepdims=True)
            out['distance_previous'] = torch.from_numpy(distance_previous.astype(np.float32))
            out['distance_next'] = torch.from_numpy(distance_next.astype(np.float32))

        for out_item in out:
            out[out_item] = out[out_item].to(self.device)

        return out

    def __next__(self):
        if self.shuffle:
            self.index = random.randint(0, len(self) - 1)
        else:
            self.index = (self.index + 1) % len(self)
        item = self[self.index % len(self)]
        for key in item:
            item[key] = item[key].unsqueeze(0)
        return item

