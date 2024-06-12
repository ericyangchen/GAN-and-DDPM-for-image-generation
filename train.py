import os
import pytz
from tqdm import tqdm
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from utils import set_random_seed
from dataset.dataloader import IclevrDataset
from model.GAN import Generator, Discriminator
from model.DDPM import DDPM_NoisePredictor
from evaluator import evaluation_model
from evaluate import evaluate_GAN, evaluate_DDPM


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def weights_clipping(m):
    for p in m.parameters():
        p.data.clamp_(-0.01, 0.01)


def train_GAN(args):
    # init tensorboard
    writer = SummaryWriter(log_dir=f"{args.output_dir}/logs")

    # load dataset
    train_dataset = IclevrDataset(
        mode="train",
        json_root=args.data_path,
        image_root=f"{args.data_path}/images",
        num_cpus=8,
    )

    # dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32
    )

    # Initialize generator and discriminator
    generator = Generator(args).to(args.device)
    discriminator = Discriminator(args).to(args.device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Loss function
    criterion = torch.nn.BCELoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999)
    )

    # define real/fake labels
    real_label = 1.0
    fake_label = 0.0

    for epoch in range(args.epochs):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0

        generator.train()
        discriminator.train()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for i, (images, conditions) in enumerate(progress_bar):
            batch_size = images.size(0)

            real_imgs = images.to(args.device)
            conditions = conditions.to(args.device)

            #############################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            #############################################################
            discriminator.zero_grad()

            ## Train with all-real batch
            label = torch.full((batch_size,), real_label, dtype=torch.float).to(
                args.device
            )
            output = discriminator(real_imgs, conditions).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            D_x = output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(batch_size, args.latent_dim, 1, 1).to(args.device)
            fake = generator(noise, conditions)
            label.fill_(fake_label)
            output = discriminator(fake.detach(), conditions).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()

            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake

            # Update D
            optimizer_D.step()

            #############################################################
            # (2) Update G network: maximize log(D(G(z)))
            #############################################################
            generator.zero_grad()

            label.fill_(real_label)
            output = discriminator(fake, conditions).view(-1)

            errG = criterion(output, label)
            errG.backward()

            D_G_z2 = output.mean().item()

            # Update G
            optimizer_G.step()

            # Log the losses
            g_loss_epoch += errG.item()
            d_loss_epoch += errD.item()

            progress_bar.set_description(
                f"Epoch {epoch}/{args.epochs} [Batch {i}/{len(train_loader)}] [D loss: {errD.item()}] [G loss: {errG.item()}]"
            )

        # Log the average losses to TensorBoard
        writer.add_scalar("Loss/Discriminator", d_loss_epoch / len(train_loader), epoch)
        writer.add_scalar("Loss/Generator", g_loss_epoch / len(train_loader), epoch)

        # evaluate accuracy
        if epoch % 10 == 0:
            with torch.no_grad():
                test_dataset = IclevrDataset(
                    mode="test",
                    json_root=args.data_path,
                    image_root=f"{args.data_path}/images",
                    num_cpus=8,
                )
                accuracy, generated_images = evaluate_GAN(args, generator, test_dataset)

                # visualize images
                image_visualizations = make_grid(generated_images, nrow=8)

                # tensorboard
                writer.add_scalar("Evaluation Accuracy", accuracy, epoch)
                writer.add_image("Generated Images", image_visualizations, epoch)

        # save model
        if epoch % 10 == 0:
            torch.save(
                generator.state_dict(), f"{args.output_dir}/generator_epoch{epoch}.pth"
            )
            torch.save(
                discriminator.state_dict(),
                f"{args.output_dir}/discriminator_epoch{epoch}.pth",
            )
        elif epoch == args.epochs - 1:
            torch.save(generator.state_dict(), f"{args.output_dir}/generator.pth")
            torch.save(
                discriminator.state_dict(), f"{args.output_dir}/discriminator.pth"
            )

    print("Training completed!")

    # evaluate
    for mode in ["test", "new_test"]:
        test_dataset = IclevrDataset(
            mode=mode,
            json_root=args.data_path,
            image_root=f"{args.data_path}/images",
            num_cpus=8,
        )

        accuracy, generated_images = evaluate_GAN(args, generator, test_dataset)

        print(f"Accuracy for {mode}: {accuracy}")

        # visualize images
        image_visualizations = make_grid(generated_images, nrow=8)

        save_image(image_visualizations, f"{args.output_dir}/{mode}_gan_result.png")


def train_DDPM(args):
    # init tensorboard
    writer = SummaryWriter(log_dir=f"{args.output_dir}/logs")

    # load dataset
    train_dataset = IclevrDataset(
        mode="train",
        json_root=args.data_path,
        image_root=f"{args.data_path}/images",
        num_cpus=8,
    )

    # dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    # Initialize DDPM
    ddpm = DDPM_NoisePredictor(n_classes=args.n_classes).to(args.device)

    # Optimizer
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=args.lr)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_loader) * args.epochs),
    )

    # scheduler
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timestamps, beta_schedule="squaredcos_cap_v2"
    )

    # Loss function
    criterion = torch.nn.MSELoss()

    # training
    for epoch in range(args.epochs):
        epoch_loss = 0.0

        ddpm.train()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for i, (images, labels) in enumerate(progress_bar):
            batch_size = images.size(0)
            images, labels = images.to(args.device), labels.to(args.device)

            # sample noise
            noises = torch.randn_like(images).to(args.device)

            timesteps = torch.randint(
                0,
                ddpm_scheduler.config.num_train_timesteps,
                (batch_size,),
                dtype=torch.int64,
            ).to(args.device)

            noisy_images = ddpm_scheduler.add_noise(images, noises, timesteps)

            pred_noises = ddpm(noisy_images, timesteps, labels)

            loss = criterion(pred_noises, noises)
            loss.backward()

            epoch_loss += loss.item()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_description(
                f"Epoch {epoch}/{args.epochs} [Batch {i}/{len(train_loader)}] [Loss: {loss.item():.4f}]"
            )

        # Log the average loss to TensorBoard
        writer.add_scalar("Loss/DDPM", epoch_loss / len(train_loader), epoch)

        # evaluate accuracy
        if epoch % 10 == 0:
            with torch.no_grad():
                test_dataset = IclevrDataset(
                    mode="test",
                    json_root=args.data_path,
                    image_root=f"{args.data_path}/images",
                    num_cpus=8,
                )
                accuracy, generated_images = evaluate_DDPM(
                    args, ddpm, ddpm_scheduler, test_dataset
                )

                # visualize images
                image_visualizations = make_grid(generated_images, nrow=8)

                # tensorboard
                writer.add_scalar("Evaluation Accuracy", accuracy, epoch)
                writer.add_image("Generated Images", image_visualizations, epoch)

        # save model
        if epoch % 10 == 0:
            torch.save(
                {
                    "ddpm": ddpm.state_dict(),
                    "ddpm_scheduler": ddpm_scheduler,
                },
                f"{args.output_dir}/ddpm_epoch{epoch}.pth",
            )
        elif epoch == args.epochs - 1:
            torch.save(
                {
                    "ddpm": ddpm.state_dict(),
                    "ddpm_scheduler": ddpm_scheduler,
                },
                f"{args.output_dir}/ddpm.pth",
            )

    print("Training completed!")

    # evaluate
    for mode in ["test", "new_test"]:
        test_dataset = IclevrDataset(
            mode=mode,
            json_root=args.data_path,
            image_root=f"{args.data_path}/images",
            num_cpus=8,
        )

        accuracy, generated_images = evaluate_DDPM(
            args, ddpm, ddpm_scheduler, test_dataset
        )

        print(f"Accuracy for {mode}: {accuracy}")

        # visualize images
        image_visualizations = make_grid(generated_images, nrow=8)

        save_image(image_visualizations, f"{args.output_dir}/{mode}_ddpm_result.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generative Models with iclevr Dataset"
    )

    # fmt: off
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--data_path", type=str, default="data", help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    parser.add_argument("--model", type=str, default="GAN", choices=["GAN", "DDPM"])
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    
    parser.add_argument("--n_classes", type=int, default=24, help="Number of object label classes")
    
    # For GAN
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent dimension")
    parser.add_argument("--nc", type=int, default=3, help="Number of color channels")
    parser.add_argument("--ngf", type=int, default=64, help="Feature map size for generator")
    parser.add_argument("--ndf", type=int, default=64, help="Feature map size for discriminator")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam")
    
    # For DDPM
    parser.add_argument("--num_train_timestamps", type=int, default=1000, help="Number of training timesteps")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of learning rate warmup steps")
    # fmt: on

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # load args
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # set random seed
    if args.seed is not None:
        set_random_seed(args.seed)

    # create output directory
    if args.output_dir is not None:
        if args.debug:
            args.output_dir = f"{args.output_dir}/{args.model}-debug"
        else:
            tz = pytz.timezone("Asia/Taipei")
            now = datetime.now(tz).strftime("%Y%m%d-%H%M")
            args.output_dir = f"{args.output_dir}/{args.model}-{now}"

        # create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # create logs directory
        os.makedirs(f"{args.output_dir}/logs", exist_ok=True)

    # start training
    print(f"Start training {args.model}:")
    print(f"Output directory: {args.output_dir}\n")

    if args.model == "GAN":
        train_GAN(args)
    elif args.model == "DDPM":
        train_DDPM(args)
