from torchvision import transforms
from PIL import Image
from . import dataset
from base.base_data_loader import BaseDataLoader
from torch.utils.data import DataLoader

class GoProDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers):
        transform = transforms.Compose([
            transforms.Resize([360, 640], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = dataset.GoProDataset(data_dir, transform=transform, height=360, width=640, fine_size=256)

        super(GoProDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CustomDataLoader(DataLoader):
    def __init__(self, data_dir):
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
        ])
        self.dataset = dataset.CustomDataset(data_dir, transform=transform)

        super(CustomDataLoader, self).__init__(self.dataset)