#!/bin/bash
#SBATCH --job-name=train_job          
#SBATCH -p lrz-v100x2              
#SBATCH --gres=gpu:1                  
#SBATCH --time=100               
#SBATCH -o output_log.out
#SBATCH -e error_log.err   

enroot create --name ssl_container ./code/container/kernel-infonce.sqsh
enroot start -m code/ ssl_container << 'EOF'

pip install ipdb

cd ../code/spectral_contrastive_learning

python eval/eval_run.py --dataset cifar10 --dir ./log/spectral/in-progress-2024-11-18spectral-resnet18-mlp1000-norelu-cifar10-lr003-mu1-log_freq:100/ --arch resnet18_cifar_variant1 --batch_size 256 --epochs 100 --specific_ckpts 100.pth --nomlp

EOF
