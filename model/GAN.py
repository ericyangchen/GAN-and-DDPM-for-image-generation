import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        self.n_classes = args.n_classes
        self.ngf, self.nc, self.nz = args.ngf, args.nc, args.latent_dim

        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                self.nz + self.n_classes, self.ngf * 8, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        batch_size, nz, w, h = noise.shape
        _, n_classes = labels.shape
        labels = labels.view(batch_size, n_classes, w, h)

        inputs = torch.cat((noise, labels), 1)

        outputs = self.main(inputs)

        return outputs


class Discriminator(nn.Module):

    def __init__(self, args, label_embed_size=4096):
        super(Discriminator, self).__init__()
        self.n_classes = args.n_classes
        self.ndf, self.nc, self.nz = args.ndf, args.nc, args.latent_dim

        self.label_embedding = nn.Embedding(self.n_classes, label_embed_size)

        self.main = nn.Sequential(
            nn.Conv2d(self.nc + self.n_classes, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, images, labels):
        batch_size, c, w, h = images.shape

        label_embed = self.label_embedding(labels)
        _, n_classes, label_embed_size = label_embed.shape

        label_embed = label_embed.view(batch_size, n_classes, w, h)

        inputs = torch.cat((images, label_embed), dim=1)

        outputs = self.main(inputs)

        outputs = outputs.view(-1, 1).squeeze(1)

        return outputs
