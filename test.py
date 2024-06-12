import os
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from diffusers import DDPMScheduler

from utils import set_random_seed
from dataset.dataloader import IclevrDataset
from model.GAN import Generator
from model.DDPM import DDPM_NoisePredictor
from evaluator import evaluation_model
from evaluate import evaluate_GAN, evaluate_DDPM, show_DDPM_denoising_process


def test_GAN(generator, args):
    evaluator = evaluation_model()

    for mode in ["test", "new_test"]:
        test_dataset = IclevrDataset(
            mode=mode,
            json_root=args.data_path,
            image_root=f"{args.data_path}/images",
            num_cpus=8,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
        )

        accuracy, generated_images = evaluate_GAN(args, generator, test_dataset)
        print("---------------------------------")
        print(f"Accuracy for {mode}: {round(accuracy * 100, 2)}%")

        # visualize images
        image_visualizations = make_grid(generated_images, nrow=8)

        save_image(image_visualizations, f"{args.output_dir}/{mode}_gan_result.png")


def test_DDPM(args, ddpm, ddpm_scheduler):
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

        print("---------------------------------")
        print(f"Accuracy for {mode}: {round(accuracy * 100, 2)}%")

        # visualize images
        image_visualizations = make_grid(generated_images, nrow=8)

        save_image(image_visualizations, f"{args.output_dir}/{mode}_ddpm_result.png")

    # show denoising process
    denoising_process_images = show_DDPM_denoising_process(args, ddpm, ddpm_scheduler)

    denoising_process_images_grid = make_grid(
        denoising_process_images, nrow=len(denoising_process_images)
    )

    save_image(
        denoising_process_images_grid, f"{args.output_dir}/ddpm_denoising_process.png"
    )


def main(args):
    if args.model == "GAN":
        generator = Generator(args).to(args.device)
        generator.load_state_dict(torch.load(f"{args.model_path}"))
        test_GAN(generator, args)

    elif args.model == "DDPM":
        ddpm = DDPM_NoisePredictor(args.n_classes).to(args.device)
        ddpm.load_state_dict(torch.load(f"{args.model_path}"))

        ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_timestamps,
            beta_schedule="squaredcos_cap_v2",
        )

        test_DDPM(args, ddpm, ddpm_scheduler)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generative Models with iclevr Dataset"
    )

    # fmt: off
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--data_path", type=str, default="data", help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--model_path", type=str, required=True, help="Model directory")

    parser.add_argument("--model", type=str, default="GAN", choices=["GAN", "DDPM"])
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    
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

    main(args)
