import os
import numpy as np
from torch.utils.data import Dataset


class FlatDirectoryImageDataset(Dataset):

    def __setup_files(self):
        file_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list

        for file_name in file_names:
            possible_file = os.path.join(self.data_dir, file_name)
            if os.path.isfile(possible_file):
                files.append(possible_file)

        return files

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.files = self.__setup_files()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image

        img_file = self.files[idx]

        if img_file[-4:] == ".npy":
            img = np.load(img_file)
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))

        else:
            img = Image.open(self.files[idx])
        if self.transform is not None:
            img = self.transform(img)

        if img.shape[0] >= 4:
            img = img[:3, :, :]

        return img


class FoldersDistributedDataset(Dataset):
    def __setup_files(self):
        dir_names = os.listdir(self.data_dir)
        files = []

        for dir_name in dir_names:
            file_path = os.path.join(self.data_dir, dir_name)
            file_names = os.listdir(file_path)
            for file_name in file_names:
                possible_file = os.path.join(file_path, file_name)
                if os.path.isfile(possible_file):
                    files.append(possible_file)

        return files

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.files = self.__setup_files()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image

        img_name = self.files[idx]
        if img_name[-4:] == ".npy":
            img = np.load(img_name)
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))
        else:
            img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if img.shape[0] >= 4:
            img = img[:3, :, :]

        return img


def get_transform(new_size=None):
    from torchvision.transforms import ToTensor, Normalize, Compose, Resize

    if new_size is not None:
        image_transform = Compose([
            Resize(new_size),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    else:
        image_transform = Compose([
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    return image_transform


def get_data_loader(dataset, batch_size, num_workers):
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dl
