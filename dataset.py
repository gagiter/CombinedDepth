
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF
import csv
import torch


class Data(Dataset):
    def __init__(self, args, mode='train'):
        super(Data, self).__init__()
        self.args = args
        self.mode = mode
        self.data = []
        self.load_data()

    def load_data(self):
        csv_name = self.mode + '.csv'
        for root, dirs, files in os.walk(self.args.data_root):
            if csv_name in files:
                with open(os.path.join(root, csv_name)) as csv_file:
                    csv_reader = csv.reader(csv_file)
                    keys = next(csv_reader)
                    for row in csv_reader:
                        item = dict(zip(keys, row))
                        self.data.append((root, item))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        root = self.data[idx][0]
        item = self.data[idx][1]
        image = os.path.join(root, item['image'])
        depth = None if item['depth'] == 'None' else os.path.join(root, item['depth'])
        stereo = None if item['stereo'] == 'None' else os.path.join(root, item['stereo'])
        previous = None if item['previous'] == 'None' else os.path.join(root, item['previous'])
        # next = None if item['next'] == 'None' else os.path.join(root, item['next'])

        image = Image.open(image)
        # image = TF.resize(image, 256, interpolation=)
        image = TF.center_crop(image, (256, 512))
        image = TF.to_tensor(image)

        depth = Image.open(depth)
        # depth = TF.resize(depth, 256)
        depth = TF.center_crop(depth, (256, 512))
        depth = TF.to_tensor(depth).float()
        depth /= (256.0 * 80.0)
        mask = depth > 0.0
        mask &= depth < 1.0
        depth *= mask

        return {
            'image': image.to(self.args.device),
            'depth': depth.to(self.args.device),
            'mask': mask.to(self.args.device)
        }
