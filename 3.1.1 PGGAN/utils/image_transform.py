import torchvision.transforms as Transforms
import os
import random
import numpy as np
from PIL import Image

class NumpyResize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return np.array(img.resize(self.size, resample=Image.BILINEAR))

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class NumpyFlip(object):

    def __init__(self, p=0.5):
        self.p = p
        random.seed(None)

    def __call__(self, img):
        if random.random() < self.p:
            return np.flip(img, 1).copy()
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class NumpyToTensor(object):

    def __init__(self):
        return

    def __call__(self, img):
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)

        return Transforms.functional.to_tensor(img)


def pil_loader(path):
    imgExt = os.path.splitext(path)[1]
    if imgExt == ".npy":
        img = np.load(path)[0]
        return np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def standardTransform(size):
    return Transforms.Compose([NumpyResize(size),
                               Transforms.ToTensor(),
                               Transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))])