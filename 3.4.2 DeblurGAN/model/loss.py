import torch
import torch.nn.functional as F
import torch.autograd as autograd
from .layer_utils import CONV3_3_IN_VGG_19


def perceptual_loss(deblurred, sharp):
    model = CONV3_3_IN_VGG_19
    deblurred_feature_map = model.forward(deblurred)
    sharp_feature_map = model.forward(sharp).detach()
    loss = F.mse_loss(deblurred_feature_map, sharp_feature_map)
    return loss


def wgan_gp_loss(type, **kwargs):
    if type == 'G':
        deblurred_discriminator_out = kwargs['deblurred_discriminator_out']
        return -deblurred_discriminator_out.mean()

    elif type == 'D':
        gp_lambda = kwargs['gp_lambda']
        interpolates = kwargs['interpolates']
        interpolates_discriminator_out = kwargs['interpolates_discriminator_out']
        sharp_discriminator_out = kwargs['sharp_discriminator_out']
        deblurred_discriminator_out = kwargs['deblurred_discriminator_out']

        wgan_loss = deblurred_discriminator_out.mean() - sharp_discriminator_out.mean()

        gradients = autograd.grad(outputs=interpolates_discriminator_out, inputs=interpolates,
                                  grad_outputs=torch.ones(interpolates_discriminator_out.size()).cuda(),
                                  retain_graph=True,
                                  create_graph=True)[0]
        gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

        return wgan_loss, (gp_lambda * gradient_penalty)
