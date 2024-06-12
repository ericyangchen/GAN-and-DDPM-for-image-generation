import torch
import torch.nn as nn
from diffusers import UNet2DModel
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)


class DDPM_NoisePredictor(nn.Module):
    def __init__(self, n_classes=24, label_embed_size=4096):
        super().__init__()

        self.label_embedding = nn.Embedding(n_classes, label_embed_size)

        self.model = UNet2DModel(
            sample_size=64,
            in_channels=3 + n_classes,
            out_channels=3,
            time_embedding_type="positional",
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, images, timestep, labels):
        batch_size, c, w, h = images.shape

        label_embed = self.label_embedding(labels)

        # reshape label_embed to match the shape of images
        _, n_classes, label_embed_size = label_embed.shape
        label_embed = label_embed.view(batch_size, n_classes, w, h)

        inputs = torch.cat((images, label_embed), dim=1)

        outputs = self.model(inputs, timestep).sample

        return outputs
