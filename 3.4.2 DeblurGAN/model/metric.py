import math
import torch


def PSNR(deblurred, sharp):
    mse = torch.mean((deblurred - sharp) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 10 * math.log10(PIXEL_MAX ** 2 / mse)
