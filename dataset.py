
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import csv


class Data(Dataset):
    def __init__(self, data_root, mode='train', device='cpu'):
        super(Data, self).__init__()
        self.data_root = data_root
        self.mode = mode
        self.device = device
        self.data = []
        self.load_data()

    def load_data(self):
        csv_name = self.mode + '.csv'
        for root, dirs, files in os.walk(self.data_root):
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
        ref = dict()
        for ref_name in ['stereo']:  # , 'previous' ['stereo', 'previous', 'next']:
            ref[ref_name] = None if item[ref_name] == 'None' \
                else os.path.join(root, item[ref_name])

        out = dict()
        image = Image.open(image)
        image = TF.center_crop(image, (256, 512))
        image = TF.to_tensor(image)
        out['image'] = image

        if depth:
            depth = Image.open(depth)
            depth = TF.center_crop(depth, (256, 512))
            depth = TF.to_tensor(depth).float()
            depth /= 256.0
            mask = depth > 0.00001
            depth[mask] = 1.0 / depth[mask]
            out['depth'] = depth

        for ref_key in ref:
            if ref[ref_key] is not None:
                ref[ref_key] = Image.open(ref[ref_key])
                ref[ref_key] = TF.center_crop(ref[ref_key], (256, 512))
                ref[ref_key] = TF.to_tensor(ref[ref_key])
                out[ref_key] = ref[ref_key]

        for out_item in out:
            out[out_item] = out[out_item].to(self.device)

        return out
