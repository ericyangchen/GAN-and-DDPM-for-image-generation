import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from evaluator import evaluation_model

def evaluate_GAN(args, generator, test_dataset):
    evaluator = evaluation_model()

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    generated_images = []

    total_accuracy = 0.0

    for i, labels in enumerate(test_loader):
        batch_size = labels.size(0)
        labels = labels.to(args.device)

        noise = torch.randn(batch_size, args.latent_dim, 1, 1).to(args.device)

        # generate images
        images = generator(noise, labels)

        # calculate accuracy
        accuracy = evaluator.eval(images, labels)
        total_accuracy += accuracy

        denormalized_images = (images / 2 + 0.5).clamp(0, 1)

        generated_images.extend(denormalized_images)

    # average accuracy
    accuracy = total_accuracy / len(test_loader)

    return accuracy, generated_images


def evaluate_DDPM(args, ddpm, ddpm_scheduler, test_dataset):
    evaluator = evaluation_model()

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    generated_images = []

    total_accuracy = 0.0

    for i, labels in enumerate(test_loader):
        batch_size = labels.size(0)
        labels = labels.to(args.device)

        # reverse process: start from random noise image
        images = torch.randn(batch_size, 3, 64, 64).to(args.device)
        for i, timestep in enumerate(ddpm_scheduler.timesteps):
            with torch.no_grad():
                pred_noise = ddpm(images, timestep, labels)
            images = ddpm_scheduler.step(pred_noise, timestep, images).prev_sample

        # calculate accuracy
        accuracy = evaluator.eval(images, labels)
        total_accuracy += accuracy

        denormalized_images = (images / 2 + 0.5).clamp(0, 1)

        generated_images.extend(denormalized_images)

    # average accuracy
    accuracy = total_accuracy / len(test_loader)

    return accuracy, generated_images


def show_DDPM_denoising_process(args, ddpm, ddpm_scheduler):
    # one-hot label with [“red sphere,” “yellow cube,” “cyan cylinder”]
    label = torch.zeros(24, dtype=torch.long)
    label[9] = 1
    label[7] = 1
    label[22] = 1
    label = label.unsqueeze(0).to(args.device)

    batch_size = 1

    total_timesteps = len(ddpm_scheduler.timesteps)

    denoising_process_images = []

    current_image = torch.randn(batch_size, 3, 64, 64).to(args.device)

    for i, timestep in enumerate(ddpm_scheduler.timesteps):
        with torch.no_grad():
            pred_noise = ddpm(current_image, timestep, label)
        current_image = ddpm_scheduler.step(
            pred_noise, timestep, current_image
        ).prev_sample

        if i % 100 == 0 or i == total_timesteps - 1:
            denormalized_image = (current_image[0] / 2 + 0.5).clamp(0, 1)
            denoising_process_images.append(denormalized_image)

    return denoising_process_images
