import os
import torch


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr_lambda(lr_lambda):
    if lr_lambda == 'origin_lr_scheduler':
        return lambda epoch: (1 - (epoch - 150) / 150) if epoch > 150 else 1
    else:
        raise NotImplementedError('lr_lambda [%s] is not found' % lr_lambda)


def get_lr_scheduler(lr_scheduler_config, optimizer):
    lr_scheduler_class = getattr(torch.optim.lr_scheduler, lr_scheduler_config['type'])
    if lr_scheduler_config['type'] == 'LambdaLR':
        lr_lambda = get_lr_lambda(lr_scheduler_config['args']['lr_lambda'])
        return lr_scheduler_class(optimizer, lr_lambda)
    else:
        return lr_scheduler_class(optimizer, **lr_scheduler_config['args'])


def denormalize(image_tensor):
    return (image_tensor + 1) / 2.0
