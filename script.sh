# install env
conda create python=3.9 -n dlp-lab6 -y
conda activate dlp-lab6

pip install numpy tqdm pyyaml pytz tensorboard pandas scipy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers


# env startup
conda activate dlp-lab6
cd /home_nfs/ericyangchen/DLP/lab6/src

# tensorboard
tensorboard --logdir="/home_nfs/ericyangchen/DLP/lab6/src/outputs" --port 9090


# train
python train.py --model GAN  --batch_size 32 --ngf 300 --ndf 100 
python train.py --model DDPM --batch_size 128 



# test
output_dir=outputs/GAN-old; 
python test.py --model GAN \
    --output_dir $output_dir \
    --model_path $output_dir/generator_epoch770.pth \
    --ngf 300 \
    --ndf 100 \
    --seed 2024 






# Result
## GAN
```
python test.py --model GAN --output_dir outputs/GAN \
    --model_path outputs/GAN/generator_epoch660.pth \
    --seed 2024
```

## DDPM
```
python test.py --model DDPM --output_dir outputs/DDPM \
    --model_path outputs/DDPM/ddpm_epoch200.pth \
    --seed 2024
```
