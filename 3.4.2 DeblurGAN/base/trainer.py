import random
import numpy as np
import torch
from torchvision.utils import make_grid
from base.base_trainer import BaseTrainer
from utils.util import denormalize


class Trainer(BaseTrainer):

    def __init__(self, config, generator, discriminator, loss, metrics, optimizer, 
                 lr_scheduler, data_loader, train_logger=None):
        super(Trainer, self).__init__(config, generator, discriminator,
                                      loss, metrics, optimizer, lr_scheduler, train_logger)

        self.data_loader = data_loader
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()

        total_generator_loss = 0
        total_discriminator_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, sample in enumerate(self.data_loader):
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            blurred = sample['blurred'].to(self.device)
            sharp = sample['sharp'].to(self.device)

            deblurred = self.generator(blurred)

            with torch.no_grad():
                denormalized_blurred = denormalize(blurred)
                denormalized_sharp = denormalize(sharp)
                denormalized_deblurred = denormalize(deblurred)

            if batch_idx % 100 == 0:
                self.writer.add_image('blurred', make_grid(denormalized_blurred.cpu()))
                self.writer.add_image('sharp', make_grid(denormalized_sharp.cpu()))
                self.writer.add_image('deblurred', make_grid(denormalized_deblurred.cpu()))

            critic_updates = 5
            discriminator_loss = 0
            torch.autograd.set_detect_anomaly(True)
            
            for i in range(critic_updates):
                self.discriminator_optimizer.zero_grad()
                gp_lambda = self.config['others']['gp_lambda']
                alpha = random.random()
                interpolates = alpha * sharp + (1 - alpha) * deblurred
                interpolates_discriminator_out = self.discriminator(interpolates)
                sharp_discriminator_out = self.discriminator(sharp)
                deblurred_discriminator_out = self.discriminator(deblurred)
                kwargs = {
                    'gp_lambda': gp_lambda,
                    'interpolates': interpolates,
                    'interpolates_discriminator_out': interpolates_discriminator_out,
                    'sharp_discriminator_out': sharp_discriminator_out,
                    'deblurred_discriminator_out': deblurred_discriminator_out
                }
                
                wgan_loss_d, gp_d = self.adversarial_loss('D', **kwargs)
                discriminator_loss_per_update = wgan_loss_d + gp_d

                self.writer.add_scalar('wgan_loss_d', wgan_loss_d.item())
                self.writer.add_scalar('gp_d', gp_d.item())

                discriminator_loss_per_update.backward(retain_graph=True)
                self.discriminator_optimizer.step()
                discriminator_loss += discriminator_loss_per_update.item()

            discriminator_loss /= critic_updates
            self.writer.add_scalar('discriminator_loss', discriminator_loss)
            total_discriminator_loss += discriminator_loss

            self.generator_optimizer.zero_grad()

            content_loss_lambda = self.config['others']['content_loss_lambda']
            kwargs = {
                'deblurred_discriminator_out': deblurred_discriminator_out
            }
            adversarial_loss_g = self.adversarial_loss('G', **kwargs)
            content_loss_g = self.content_loss(deblurred, sharp) * content_loss_lambda
            
            generator_loss = adversarial_loss_g.detach() + content_loss_g

            self.writer.add_scalar('adversarial_loss_g', adversarial_loss_g.item())
            self.writer.add_scalar('content_loss_g', content_loss_g.item())
            self.writer.add_scalar('generator_loss', generator_loss.item())

            generator_loss.backward()
            self.generator_optimizer.step()
            total_generator_loss += generator_loss.item()

            total_metrics += self._eval_metrics(denormalized_deblurred, denormalized_sharp)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] generator_loss: {:.6f} discriminator_loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        generator_loss.item(),
                        discriminator_loss
                    )
                )

        log = {
            'generator_loss': total_generator_loss / len(self.data_loader),
            'discriminator_loss': total_discriminator_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }
        self.generator_lr_scheduler.step()
        self.discriminator_lr_scheduler.step()

        return log

